import torch
import torch.nn.functional as F
import numpy as np
import math

def gradient_loss(s, penalty='l2'):
    # s shape: (N, C, H, W) for 2D
    dy = torch.abs(s[:, :, 1:, :] - s[:, :, :-1, :]) 
    dx = torch.abs(s[:, :, :, 1:] - s[:, :, :, :-1]) 

    if penalty == 'l2':
        dy = dy * dy
        dx = dx * dx

    d = torch.mean(dx) + torch.mean(dy)
    return d / 2.0  

def app_gradient_loss(mask, s, penalty='l2'):
    dy = torch.abs(s[:, :, 1:, :] - s[:, :, :-1, :])
    dx = torch.abs(s[:, :, :, 1:] - s[:, :, :, :-1])

    if penalty == 'l2':
        dy = dy * dy
        dx = dx * dx

    d = torch.mean(mask * (dx + dy))
    return d / 2.0

def ncc_loss(I, J, win=None):
    ndims = len(list(I.size())) - 2
    assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

    if win is None:
        win = [9] * ndims

    sum_filt = torch.ones([1, 1, *win], device=I.device, dtype=I.dtype)

    pad_no = math.floor(win[0]/2)

    if ndims == 1:
        stride = (1,)
        padding = (pad_no,)
    elif ndims == 2:
        stride = (1,1)
        padding = (pad_no, pad_no)
    else:
        stride = (1,1,1)
        padding = (pad_no, pad_no, pad_no)

    I_var, J_var, cross = compute_local_sums(I, J, sum_filt, stride, padding, win)

    I_var = torch.clamp(I_var, min=1e-5)
    J_var = torch.clamp(J_var, min=1e-5)

    cc = cross*cross / (I_var*J_var + 1e-5)

    return 1 - torch.mean(cc)


def compute_local_sums(I, J, filt, stride, padding, win):
    ndims = len(win)
    I2 = I * I
    J2 = J * J
    IJ = I * J

    if ndims == 1:
        conv_func = F.conv1d
    elif ndims == 2:
        conv_func = F.conv2d
    elif ndims == 3:
        conv_func = F.conv3d
    else:
        raise ValueError(f"Unsupported dimension: {ndims}")

    I_sum = conv_func(I, filt, stride=stride, padding=padding)
    J_sum = conv_func(J, filt, stride=stride, padding=padding)
    I2_sum = conv_func(I2, filt, stride=stride, padding=padding)
    J2_sum = conv_func(J2, filt, stride=stride, padding=padding)
    IJ_sum = conv_func(IJ, filt, stride=stride, padding=padding)

    win_size = int(np.prod(win))
    u_I = I_sum / win_size
    u_J = J_sum / win_size

    cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
    I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
    J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

    return I_var, J_var, cross

def dice_coef(y_true, y_pred):
    # y_true, y_pred: (N, C, H, W)
    smooth = 1.
    a = torch.sum(y_true * y_pred, (2, 3))
    b = torch.sum(y_true ** 2, (2, 3))
    c = torch.sum(y_pred ** 2, (2, 3))
    dice = (2 * a + smooth) / (b + c + smooth)
    return torch.mean(dice)

def dice_loss(y_true, y_pred):
    d = dice_coef(y_true, y_pred)
    return 1 - d

def att_dice(y_true, y_pred):
    dice = dice_coef(y_true, y_pred).detach()
    loss = (1 - dice) ** 3
    return loss

def masked_dice_loss(y_true, y_pred, mask):
    smooth = 1.
    a = torch.sum(y_true * y_pred * mask, (2, 3))
    b = torch.sum((y_true + y_pred) * mask, (2, 3))
    dice = (2 * a) / (b + smooth)
    return 1 - torch.mean(dice)

def MSE(y_true, y_pred):
    return torch.mean((y_true - y_pred) ** 2)

def MAE(y_true, y_pred):
    return torch.mean(torch.abs(y_true - y_pred))

def partical_MAE(y_true, y_pred, mask, Lambda=0.5):
    return torch.mean(torch.abs(y_true - y_pred) * mask) * Lambda

def mix_ce_dice(y_true, y_pred):
    return crossentropy(y_true, y_pred) + 1 - dice_coef(y_true, y_pred)

def crossentropy(y_pred, y_true):
    smooth = 1e-6
    return -torch.mean(y_true * torch.log(y_pred + smooth))

def correspondence(atlas_img, atlas_lab, unlab_img, warp_lab, corr):
    smooth = 1e-6
    # 2D: dims (N, C, H, W), sum over (2,3)
    atlas_img_proto = torch.mean(torch.sum(atlas_img * atlas_lab, dim=(2, 3)) / (torch.sum(atlas_lab, dim=(2, 3)) + 1e-6), dim=0)
    unlab_index = torch.where(torch.sum(warp_lab[:, 1:, :, :], dim=1, keepdim=True) > 0)
    unlab_img_roi = unlab_img[:, :, unlab_index[2], unlab_index[3]]
    unlab_img_roi = torch.transpose(unlab_img_roi, 1, 0).flatten(start_dim=1, end_dim=2)

    diff = [(unlab_img_roi - atlas_img_proto[i]) ** 2 for i in range(atlas_img_proto.shape[0])]
    diff = torch.cat(diff, dim=0)
    loss = diff * corr
    loss = torch.sum(loss)
    return loss

def mask_crossentropy(y_pred, y_true, mask):
    smooth = 1e-6
    return -torch.mean(mask * y_true * torch.log(y_pred + smooth))

def B_crossentropy(y_pred, y_true):
    smooth = 1e-6
    return -torch.mean(y_true * torch.log(y_pred + smooth) + (1 - y_true) * torch.log(1 - y_pred + smooth))