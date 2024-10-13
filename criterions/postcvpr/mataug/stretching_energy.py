from dataclasses import dataclass

import einops
import torch
from torch import nn

from utils.cloth_and_material import gather_triangles, get_shape_matrix
from utils.common import make_pervertex_tensor_from_lens


@dataclass
class Config:
    weight: float = 1.
    thickness: float = 4.7e-4


def create(mcfg):
    return Criterion(weight=mcfg.weight, thickness=mcfg.thickness)


def deformation_gradient(triangles, Dm_inv):
    Ds = get_shape_matrix(triangles)

    return Ds @ Dm_inv


def green_strain_tensor(F):
    device = F.device
    I = torch.eye(2, dtype=F.dtype).to(device)

    Ft = F.permute(0, 2, 1)
    return 0.5 * (Ft @ F - I)


class Criterion(nn.Module):
    def __init__(self, weight, thickness):
        super().__init__()
        self.weight = weight
        self.thickness = thickness
        self.name = 'stretching_energy'

    def create_stack(self, triangles_list, param):
        lens = [x.shape[0] for x in triangles_list]
        stack = make_pervertex_tensor_from_lens(lens, param)[:, 0]
        return stack

    def forward(self, sample):
        per_vert = torch.zeros_like(sample['cloth'].pred_pos)
        Dm_inv = sample['cloth'].Dm_inv
        uv_matrices = sample['cloth'].uv_matrices
        device = Dm_inv.device
        B = sample.num_graphs

        if not hasattr(sample['cloth'], 'energy_type') or sample['cloth'].energy_type[0] == 'stvk':
            # print('stvk')
            f_area = sample['cloth'].f_area[None, ..., 0]

            triangles_list = []
            for i in range(B):
                example = sample.get_example(i)
                v = example['cloth'].pred_pos
                f = example['cloth'].faces_batch.T

                triangles = gather_triangles(v.unsqueeze(0), f)[0]
                triangles_list.append(triangles)

            lame_mu_stack = self.create_stack(triangles_list, sample['cloth'].lame_mu)
            lame_lambda_stack = self.create_stack(triangles_list, sample['cloth'].lame_lambda)
            triangles = torch.cat(triangles_list, dim=0)

            F = deformation_gradient(triangles, Dm_inv)
            G = green_strain_tensor(F)

            I = torch.eye(2).to(device)
            I = einops.repeat(I, 'm n -> k m n', k=G.shape[0])

            G_trace = G.diagonal(dim1=-1, dim2=-2).sum(-1)  # trace

            S = lame_mu_stack[:, None, None] * G + 0.5 * lame_lambda_stack[:, None, None] * G_trace[:, None, None] * I
            energy_density_matrix = S.permute(0, 2, 1) @ G
            energy_density = energy_density_matrix.diagonal(dim1=-1, dim2=-2).sum(-1)  # trace
            f_area = f_area[0]

            energy = f_area * self.thickness * energy_density
            loss = energy.sum() / B

            per_vert.scatter_add_(0, f, energy[:, None].repeat(1, 3))
            per_vert = torch.mean(per_vert, 1)

        elif sample['cloth'].energy_type[0] == 'spring':
            # print('spring')
            v = sample['cloth'].pred_pos
            rest_v = sample['cloth'].rest_pos

            edge_idx = sample['cloth'].f_connectivity_edges
            edges = v[edge_idx[:, 0]] - v[edge_idx[:, 1]]
            edge_lengths = torch.norm(edges, dim=-1)


            rest_edges = rest_v[edge_idx[:, 0]] - rest_v[edge_idx[:, 1]]
            edge_lengths_true = torch.norm(rest_edges, dim=-1)

            edge_difference = edge_lengths - edge_lengths_true
            stiffness = sample['cloth'].stiffness
            loss = stiffness * edge_difference ** 2
            loss = torch.sum(loss, dim=-1)
            loss = torch.mean(loss)

        elif sample['cloth'].energy_type[0] == 'baraff':
            # print('baraff')
            k_stretch = sample['cloth'].k_stretch
            k_shear = sample['cloth'].k_shear
            v = sample['cloth'].pred_pos
            f = sample['cloth'].faces_batch.T
            f_area = sample['cloth'].f_area  # (F, 1)

            dX = torch.stack(
                [
                    v[f[:, 1]] - v[f[:, 0]],
                    v[f[:, 2]] - v[f[:, 0]],
                ], dim=1,
            )
            w = torch.einsum("bcd,bce->bed", dX, uv_matrices)
            # w = torch.transpose(w, 1, 2)

            stretch = torch.norm(w, dim=-1) - 1
            stretch_loss = f_area * stretch ** 2
            stretch_loss = torch.sum(stretch_loss)
            stretch_loss = k_stretch * stretch_loss
            # stretch_loss = torch.mean(stretch_loss)

            # stretch_error = (
            #         f_area[:, None] * torch.abs(stretch) * (0.5 / total_area)
            # )
            # stretch_error = torch.mean(torch.sum(stretch_error, dim=-1))

            shear = torch.sum(torch.mul(w[:, 0], w[:, 1]), dim=-1)
            shear_loss = shear ** 2
            shear_loss = shear_loss * f_area[:, 0]
            shear_loss = torch.sum(shear_loss)
            shear_loss = k_shear * shear_loss
            # shear_loss = torch.mean(shear_loss)
            # shear_error = f_area * torch.abs(shear) * (1 / total_area)
            # shear_error = torch.mean(torch.sum(shear_error, dim=-1))

            loss = stretch_loss + shear_loss


        return dict(loss=loss, per_vert=per_vert)
