from dataclasses import dataclass

import torch
from pytorch3d.ops import knn_points
from torch import nn

from utils.cloth_and_material import FaceNormals
from utils.common import gather


@dataclass
class Config:
    weight_start: float = 1e+3
    weight_max: float = 1e+5
    start_rampup_iteration: int = 50000
    n_rampup_iterations: int = 100000
    eps: float = 1e-3


def create(mcfg):
    return Criterion(weight_start=mcfg.weight_start, weight_max=mcfg.weight_max,
                     start_rampup_iteration=mcfg.start_rampup_iteration, n_rampup_iterations=mcfg.n_rampup_iterations,
                     eps=mcfg.eps)


def find_nn(x, obstacle_x, obstacle_faces, f_normals_f):
    # x: cloth position (n_verts, 3)
    # obstacle_x: obstacle position (n_verts, 3)
    # obstacle_faces: obstacle faces (n_faces, 3)
    obstacle_face_curr_pos = gather(obstacle_x, obstacle_faces, 0, 1, 1).mean(
        dim=-2)  # (n_faces, 3), position of every face center
    _, nn_idx, _ = knn_points(x.unsqueeze(0), obstacle_face_curr_pos.unsqueeze(0),
                              return_nn=True)
    nn_idx = nn_idx[0]

    # Compute distances in the new step
    obstacle_fn = f_normals_f(obstacle_x.unsqueeze(0), obstacle_faces.unsqueeze(0))[0]

    nn_points = gather(obstacle_face_curr_pos, nn_idx, 0, 1, 1)
    nn_normals = gather(obstacle_fn, nn_idx, 0, 1, 1)

    nn_points = nn_points[:, 0]
    nn_normals = nn_normals[:, 0]
    return nn_points, nn_normals


def push_away(x, nn_points, nn_normals, eps):
    device = x.device
    distance = ((x - nn_points) * nn_normals).sum(dim=-1)
    interpenetration = torch.maximum(eps - distance, torch.FloatTensor([0]).to(device))
    x = x + interpenetration[:, None] * nn_normals
    return x


def collision_handling(x, obstacle_x, obstacle_faces, f_normals_f, eps=4e-3):
    # x: cloth position (n_verts, 3)
    # obstacle_x: obstacle position (n_verts, 3)
    # obstacle_faces: obstacle faces (n_faces, 3)
    nn_points, nn_normals = find_nn(x, obstacle_x, obstacle_faces, f_normals_f)
    x = push_away(x, nn_points, nn_normals, eps=eps)
    return x


class Criterion(nn.Module):
    def __init__(self, weight_start, weight_max, start_rampup_iteration, n_rampup_iterations, eps=1e-3):
        super().__init__()
        self.weight_start = weight_start
        self.weight_max = weight_max
        self.start_rampup_iteration = start_rampup_iteration
        self.n_rampup_iterations = n_rampup_iterations
        self.eps = eps
        self.f_normals_f = FaceNormals()
        self.name = 'collision_penalty'
        self.nn_idx = None
        self.weight = self.weight_start

    def get_weight(self, iter):
        iter = iter - self.start_rampup_iteration
        iter = max(iter, 0)
        progress = iter / self.n_rampup_iterations
        progress = min(progress, 1.)
        weight = self.weight_start + (self.weight_max - self.weight_start) * progress
        return weight

    def calc_loss(self, example, use_cache=False):
        obstacle_next_pos = example['obstacle'].target_pos
        obstacle_curr_pos = example['obstacle'].pos
        obstacle_faces = example['obstacle'].faces_batch.T

        curr_pos = example['cloth'].pos
        next_pos = example['cloth'].pred_pos

        # Find correspondences in current step
        if use_cache:
            assert self.nn_idx is not None
            nn_idx = self.nn_idx
        else:
            obstacle_face_curr_pos = gather(obstacle_curr_pos, obstacle_faces, 0, 1, 1).mean(dim=-2)
            _, nn_idx, _ = knn_points(curr_pos.unsqueeze(0), obstacle_face_curr_pos.unsqueeze(0),
                                      return_nn=True)
            nn_idx = nn_idx[0]
            self.nn_idx = nn_idx

        # Compute distances in the new step
        obstacle_face_next_pos = gather(obstacle_next_pos, obstacle_faces, 0, 1, 1).mean(dim=-2)
        obstacle_fn = self.f_normals_f(obstacle_next_pos.unsqueeze(0), obstacle_faces.unsqueeze(0))[0]

        nn_points = gather(obstacle_face_next_pos, nn_idx, 0, 1, 1)
        nn_normals = gather(obstacle_fn, nn_idx, 0, 1, 1)

        nn_points = nn_points[:, 0]
        nn_normals = nn_normals[:, 0]
        device = next_pos.device

        distance = ((next_pos - nn_points) * nn_normals).sum(dim=-1)
        interpenetration = torch.maximum(self.eps - distance, torch.FloatTensor([0]).to(device))

        interpenetration = interpenetration.pow(3)
        loss = interpenetration.sum(-1)

        return loss, interpenetration

    def forward(self, sample, use_cache=False):
        B = sample.num_graphs
        iter_num = sample['cloth'].iter[0].item()
        weight = self.get_weight(iter_num)
        self.weight = weight

        loss_list = []
        per_vert_list = []
        for i in range(B):
            loss_, per_vert_ = self.calc_loss(sample.get_example(i), use_cache=use_cache)
            loss_list.append(loss_)
            per_vert_list.append(per_vert_)

        loss = sum(loss_list) / B * weight
        per_vert = sum(per_vert_list) / B * weight

        return dict(loss=loss, per_vert=per_vert)
