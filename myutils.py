import torch
import torch.nn as nn
import random
import numpy as np
from einops import rearrange
from skimage.transform import resize
from colmap import qvec2rotmat


def mk_chunk(items, batch):
    ls = []
    num = len(items) // batch
    remain = len(items) % batch
    for i in range(num):
        ls.append(items[i * batch:(i + 1) * batch])

    if remain > 0:
        ls.append(items[-remain:])

    return ls


def ch_linear(linear, ch):
    dim_in = linear.in_features
    dim_out = linear.out_features
    linear_new = nn.Linear(ch, dim_out, dtype=linear.weight.dtype, device=linear.weight.device)

    linear_new.weight.detach().zero_()
    linear_new.bias.detach().zero_()

    linear_new.weight[:, :dim_in].detach().copy_(linear.weight)
    linear_new.bias.detach().copy_(linear.bias)
    return linear_new


def ch_channel(transformer, ch_in):
    proj: nn.Linear = transformer.patch_embed.proj
    ch_in_org = proj.in_features
    proj_new = ch_linear(proj, ch_in)
    del transformer.patch_embed.proj
    transformer.patch_embed.proj = proj_new

    print(f'channel change from {ch_in_org} to {ch_in}')


def ch_cnn(cnn, ch):
    dim_in = cnn.weight.shape[1]
    dim_out = cnn.weight.shape[0]
    cnn_new = nn.Conv2d(
        in_channels=ch,
        out_channels=dim_out,
        kernel_size=2,
        stride=2,
        dtype=cnn.weight.dtype,
        device=cnn.weight.device,
    )

    cnn_new.weight.detach().zero_()
    cnn_new.bias.detach().zero_()

    cnn_new.weight[:, :dim_in].detach().copy_(cnn.weight)
    cnn_new.bias.detach().copy_(cnn.bias)
    return cnn_new


def ch_channel_cnn(transformer, ch_in):
    proj: nn.Conv2d = transformer.patch_embed.proj
    ch_in_org = proj.weight.shape[1]
    proj_new = ch_cnn(proj, ch_in)
    del transformer.patch_embed.proj
    transformer.patch_embed.proj = proj_new

    print(f'channel change from {ch_in_org} to {ch_in}')


def convert_1D_to_2D(W, height, width):
    *r, l = W.shape
    assert height * width == l

    mask_neg = W < 0
    W = np.stack([W // width, W % height], axis=-1)
    W[mask_neg, :] = -1
    W = W.reshape(*r, height, width, 2)
    return W


def convert_2D_to_1D(W):
    *r, h, w, c = W.shape
    W = W.reshape(*r, h * w, c)

    # Convert to 1D
    W = w * W[..., 0] + W[..., 1]
    W[W < 0] = -1
    return W


def downsample_W_1D(W_1D, size, height_org, width_org):
    W_2D = convert_1D_to_2D(W_1D, height_org, width_org)
    W_2D_down = downsample_W_2D(W_2D, size)
    W_1D = convert_2D_to_1D(W_2D_down)
    return W_1D


def downsample_W_2D(W_2D, size):
    W_2D = W_2D.astype('float')

    *r, h, w, c = W_2D.shape
    b = np.prod(r)

    scale = None
    h_new = None
    w_new = None

    if h > w:
        scale = size / w
        h_new = int(round(h * scale))
        w_new = size
    else:
        scale = size / h
        h_new = size
        w_new = int(round(w * scale))

    W_2D = W_2D.reshape(b, h, w, c)
    W_2D = rearrange(W_2D, 'b h w c -> h w (b c)', b=b)
    W_2D = resize(
        W_2D,
        (h_new, w_new),
        order=0,
        preserve_range=True,
        anti_aliasing=False,
    )
    W_2D = rearrange(W_2D, 'h w (b c) -> b h w c', b=b)
    W_2D = W_2D.reshape(*r, h_new, w_new, c)

    W_2D = W_2D * scale
    W_2D[W_2D < 0] = -1

    W_2D = W_2D.astype('int32')

    return W_2D


def convert_to_canonical(flow, mask):
    f1, f2, resy, resx = mask.shape

    flow[..., :, :, 0] += np.arange(resx)
    flow[..., :, :, 1] += np.arange(resy)[:, np.newaxis]
    flow = flow.astype(np.int32)

    W = np.ones((f1, f2, resy, resx, 2), dtype='int32') * -1
    valid_mask = (flow[..., 0] >= 0) & (flow[..., 0] < resx) & (flow[..., 1] >= 0) & (
        flow[..., 1] < resy) & mask > 0

    W[valid_mask] = flow[valid_mask][:, [1, 0]]
    W = resx * W[..., 0] + W[..., 1]
    W[W < 0] = -1
    W_1D = W.reshape(f1, f2, -1)

    return W_1D


def random_sample_with_idx(population, k):
    assert len(population) >= k
    idxs = list(range(len(population)))
    random.shuffle(idxs)
    idxs = idxs[:k]
    out = [population[i] for i in idxs]
    return out, idxs


class Resize_K(object):

    def __init__(self, w_src, h_src, size):
        self.w_src = w_src
        self.h_src = h_src
        short = min(w_src, h_src)

        self.w_dst = int(w_src / short * size)
        self.h_dst = int(h_src / short * size)

    def __call__(self, K):
        # process intrinsic
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        fx, fy, cx, cy, *_ = resize_intrinsic_param(
            fx,
            fy,
            cx,
            cy,
            self.w_src,
            self.h_src,
            self.w_dst,
            self.h_dst,
        )
        K[0, 0], K[1, 1], K[0, 2], K[1, 2] = fx, fy, cx, cy

        return K


class CenterCrop_K(object):

    def __init__(self, w_src, h_src, size):
        self.w_src = w_src
        self.h_src = h_src
        self.w_dst = size
        self.h_dst = size

    def __call__(self, K):

        w_diff_half = (self.w_src - self.w_dst) // 2
        h_diff_half = (self.h_src - self.h_dst) // 2

        x_min = w_diff_half
        x_max = self.w_src - w_diff_half

        y_min = h_diff_half
        y_max = self.h_src - h_diff_half

        cx, cy = K[0, 2], K[1, 2]
        cx, cy, w, h = crop_intrinsic_param(
            cx,
            cy,
            self.w_src,
            self.h_src,
            x_min,
            x_max,
            y_min,
            y_max,
        )

        # assert w == w_dst and h == h_dst
        K[0, 2], K[1, 2] = cx, cy

        return K


def resize_intrinsic_param(fx, fy, cx, cy, w_src, h_src, w_dst, h_dst):

    fx = fx * w_dst / w_src
    fy = fy * h_dst / h_src

    cx = cx * w_dst / w_src
    cy = cy * h_dst / h_src

    return fx, fy, cx, cy, w_dst, h_dst


def crop_intrinsic_param(cx, cy, w, h, x_min, x_max, y_min, y_max):

    w_hat = w - (x_min + (w - x_max))
    h_hat = h - (y_min + (h - y_max))
    cx_hat = cx - x_min
    cy_hat = cy - y_min

    assert w_hat > 0 and h_hat > 0 and cx_hat > 0 and cy_hat > 0

    return cx_hat, cy_hat, w_hat, h_hat


def K2vec(K):
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    return np.array([fx, fy, cx, cy])


def param2extrinsic(p):
    ET = np.eye(4, 4)
    ET[:3, :3] = qvec2rotmat(p.qvec)
    ET[:3, 3] = p.tvec
    return ET


def get_relative_pose(w2cs):
    abs_w2cs = w2cs
    abs_c2ws = [np.linalg.inv(w2c) for w2c in w2cs]

    cam_to_origin = 0
    target_cam_c2w = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, -cam_to_origin],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ])

    abs2rel = target_cam_c2w @ abs_w2cs[0]
    rel_c2ws = [
        target_cam_c2w,
    ] + [abs2rel @ abs_c2w for abs_c2w in abs_c2ws[1:]]
    rel_c2ws = np.array(rel_c2ws, dtype=np.float32)
    return rel_c2ws


def ray_condition_np(K, c2w, H, W):
    # c2w: B, V, 4, 4
    # K: B, V, 4

    B, F = K.shape[0], K.shape[1]

    j, i = np.meshgrid(
        np.linspace(0, H - 1, H),
        np.linspace(0, W - 1, W),
        indexing='ij',
    )

    i = i.reshape(1, 1, H * W).repeat(B, axis=0).repeat(F, axis=1) + 0.5
    j = j.reshape(1, 1, H * W).repeat(B, axis=0).repeat(F, axis=1) + 0.5

    fx, fy, cx, cy = np.split(K, 4, axis=-1)  # B,V, 1

    zs = np.ones_like(i)  # [B, HxW]
    xs = (i - cx) / fx * zs
    ys = (j - cy) / fy * zs

    directions = np.stack((xs, ys, zs), axis=-1)  # B, V, HW, 3
    directions = directions / np.linalg.norm(directions, axis=-1, keepdims=True)  # B, V, HW, 3

    rays_d = directions @ c2w[..., :3, :3].transpose(0, 1, 3, 2)  # B, V, 3, HW
    rays_o = c2w[..., :3, 3]  # B, V, 3
    rays_o = rays_o[:, :, None].repeat(rays_d.shape[-2], axis=-2)  # B, V, 3, HW
    # c2w @ dirctions
    rays_dxo = np.cross(rays_o, rays_d)
    plucker = np.concatenate([rays_dxo, rays_d], axis=-1)
    plucker = plucker.reshape(B, c2w.shape[1], H, W, 6)  # B, V, H, W, 6
    # plucker = plucker.permute(0, 1, 4, 2, 3)
    return plucker


def convert_jsonformat_to_colmapformat(matrix, applied_transform):
    if applied_transform.shape[0] == 3:
        applied_transform = np.concatenate([applied_transform, np.array([[0, 0, 0, 1]])], axis=0)

    matrix = np.linalg.inv(matrix)
    matrix = matrix @ applied_transform
    matrix[:, 1:3] *= -1
    return matrix
