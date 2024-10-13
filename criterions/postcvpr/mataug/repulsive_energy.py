from dataclasses import dataclass

from pytorch3d.loss import chamfer_distance
import torch
from torch import nn

from utils.cloth_and_material import FaceNormals
from utils.common import gather


@dataclass
class Config:
    weight: float = 1.
    threshold: float = 0.1


def create(mcfg):
    return Criterion(weight=mcfg.weight, threshold=mcfg.threshold)


class Criterion(nn.Module):
    def __init__(self, weight, threshold):
        super().__init__()
        self.weight = weight
        self.threshold = threshold
        self.name = 'repulsive_energy'

    def calc_single(self, example):
        pred_pos = example['cloth'].pred_pos
        faces = example['cloth'].faces_batch.T
        f_connectivity = example['cloth'].f_connectivity
        f_connectivity_edges = example['cloth'].f_connectivity_edges

        dist = torch.square(pred_pos[None] - pred_pos[:, None]).sum(-1) + 1e-8
        mask = torch.ones_like(dist)
        mask[dist > self.threshold * self.threshold] = 0
        mask[f_connectivity_edges[:, 0], f_connectivity_edges[:, 1]] = 0
        mask[f_connectivity_edges[:, 1], f_connectivity_edges[:, 0]] = 0
        mask[torch.arange(mask.shape[0]), torch.arange(mask.shape[0])] = 0

        masked_dist = -torch.log(dist) * mask
        loss = torch.sum(masked_dist) * self.weight

        return loss

    def forward(self, sample):
        loss_list = []
        B = sample.num_graphs
        for i in range(B):
            loss_sample = self.calc_single(sample.get_example(i))
            loss_list.append(loss_sample)

        loss = sum(loss_list) / B

        return dict(loss=loss)
