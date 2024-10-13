from dataclasses import dataclass

from torch import nn


@dataclass
class Config:
    weight: float = 1.


def create(mcfg):
    return Criterion()


class Criterion(nn.Module):
    def __init__(self):
        super().__init__()

        self.name = 'ext_force_energy'

    def forward(self, sample):
        cloth_sample = sample['cloth']
        pred_pos = cloth_sample.pred_pos
        ext_acc = cloth_sample.wind
        device = pred_pos.device

        B = sample.num_graphs

        v_mass = cloth_sample.v_mass
        U = -ext_acc * v_mass * pred_pos

        loss = U.sum() / B

        return dict(loss=loss, per_vert=U)
