"""Euclidean operations utils functions."""

import torch


def euc_sqdistance(x, y, eval_mode=False):
    """Compute euclidean squared distance between tensors.

    Args:
        x: torch.Tensor of shape (N1 x d)
        y: torch.Tensor of shape (N2 x d)
        eval_mode: boolean

    Returns:
        torch.Tensor of shape N1 x 1 with pairwise squared distances if eval_mode is false
        else torch.Tensor of shape N1 x N2 with all-pairs distances

    """
    x2 = torch.sum(x * x, dim=-1, keepdim=True)
    y2 = torch.sum(y * y, dim=-1, keepdim=True)
    if eval_mode:
        y2 = y2.t()
        xy = x @ y.t()
    else:
        assert x.shape[0] == y.shape[0]
        xy = torch.sum(x * y, dim=-1, keepdim=True)
    return x2 + y2 - 2 * xy

def euc_dist(x, y, eval_mode=False):
    return torch.abs(x - y)

def givens_rotations(r, x):
    """Givens rotations.

    Args:
        r: torch.Tensor of shape (N x d), rotation parameters
        x: torch.Tensor of shape (N x d), points to rotate

    Returns:
        torch.Tensor os shape (N x d) representing rotation of x by r
    """
    givens = r.view((r.shape[0], -1, 2))
    givens = givens / torch.norm(givens, p=2, dim=-1, keepdim=True)
    x = x.view((r.shape[0], -1, 2))
    x_rot = givens[:, :, 0:1] * x + givens[:, :, 1:] * torch.cat((-x[:, :, 1:], x[:, :, 0:1]), dim=-1)
    return x_rot.view((r.shape[0], -1))


def threed_rotate(alpha, beta, gamma, x):
    norm =  torch.sqrt(alpha ** 2 +  beta ** 2 + gamma ** 2)
    alpha = alpha / norm
    beta = beta / norm
    gamma = gamma / norm

    rot_00 = torch.cos(alpha) * torch.cos(beta)
    rot_01 = torch.cos(alpha) * torch.sin(beta) * torch.sin(gamma) - torch.sin(alpha) * torch.cos(gamma)
    rot_02 = torch.cos(alpha) * torch.sin(beta) * torch.cos(gamma) + torch.sin(alpha) * torch.sin(gamma)
    row_0 = torch.stack([rot_00, rot_01, rot_02], -1)
    rot_10 = torch.sin(alpha) * torch.cos(beta)
    rot_11 = torch.sin(alpha) * torch.sin(beta) * torch.sin(gamma) + torch.cos(alpha) * torch.cos(gamma)
    rot_12 = torch.sin(alpha) * torch.sin(beta) * torch.cos(gamma) - torch.cos(alpha) * torch.sin(gamma)
    row_1 = torch.stack([rot_10, rot_11, rot_12], -1)
    rot_20 = - torch.sin(beta)
    rot_21 = torch.cos(beta) * torch.sin(gamma)
    rot_22 = torch.cos(beta) * torch.cos(gamma)
    row_2 = torch.stack([rot_20, rot_21, rot_22], -1)
    rot = torch.stack([row_0, row_1, row_2], -2)
    # rot = rot.transpose(-1, 1)
    # print("alpha size", alpha.size())

    # print("rot size", rot.size())

    x = x.view((alpha.shape[0], -1, 3))
    x = x.unsqueeze(dim=-2)
    # print("x size", x.size())

    x_rot = torch.matmul(x, rot)
    # print("x_rot size", x_rot.size())
    return x_rot.view((alpha.shape[0], -1))

def threedim_rotate(alpha, beta, gamma, x):
    alpha = alpha.view((alpha.shape[0], -1, 2))
    alpha = alpha / torch.norm(alpha, p=2, dim=-1, keepdim=True)

    beta = beta.view((beta.shape[0], -1, 2))
    beta = beta / torch.norm(beta, p=2, dim=-1, keepdim=True)

    gamma = gamma.view((gamma.shape[0], -1, 2))
    gamma = gamma / torch.norm(gamma, p=2, dim=-1, keepdim=True)

    rot = torch.stack([alpha, beta, gamma], -2)
    x = x.view((alpha.shape[0], -1, 3))
    x = x.unsqueeze(dim=-2)
    x_rot = torch.matmul(x, rot)
    return x_rot.view((alpha.shape[0], -1))




def givens_reflection(r, x):
    """Givens reflections.

    Args:
        r: torch.Tensor of shape (N x d), rotation parameters
        x: torch.Tensor of shape (N x d), points to reflect

    Returns:
        torch.Tensor os shape (N x d) representing reflection of x by r
    """
    givens = r.view((r.shape[0], -1, 2))
    givens = givens / torch.norm(givens, p=2, dim=-1, keepdim=True)
    x = x.view((r.shape[0], -1, 2))
    x_ref = givens[:, :, 0:1] * torch.cat((x[:, :, 0:1], -x[:, :, 1:]), dim=-1) + givens[:, :, 1:] * torch.cat(
        (x[:, :, 1:], x[:, :, 0:1]), dim=-1)
    return x_ref.view((r.shape[0], -1))
