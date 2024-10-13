from dataclasses import dataclass
import torch
from torch import nn

from utils_self_inter.find_penetration_triangles_one_batch import (
    compute_self_intersection_one_batch,
)
from torch.nn.utils.rnn import pad_sequence
import igl
import sys
import numpy as np

@dataclass
class Config:
    weight: float = 100.0
    norm: str = "l1"
    return_average: bool = True
    area: bool = False


def create(mcfg):
    return Criterion(weight=mcfg.weight, norm=mcfg.norm, return_average=mcfg.return_average)

def writeOBJ(file, V, F, Vt=None, Ft=None):
    if not Vt is None:
        assert len(F) == len(
            Ft
        ), "Inconsistent data, mesh and UV map do not have the same number of faces"

    with open(file, "w") as file:
        # Vertices
        for v in V:
            line = "v " + " ".join([str(_) for _ in v]) + "\n"
            file.write(line)
        # UV verts
        if not Vt is None:
            for v in Vt:
                line = "vt " + " ".join([str(_) for _ in v]) + "\n"
                file.write(line)
        # 3D Faces / UV faces
        if Ft:
            F = [
                [str(i + 1) + "/" + str(j + 1) for i, j in zip(f, ft)]
                for f, ft in zip(F, Ft)
            ]
        else:
            F = [[str(i + 1) for i in f] for f in F]
        for f in F:
            line = "f " + " ".join(f) + "\n"
            file.write(line)

class Criterion(nn.Module):
    def __init__(self, weight, norm='l1', area=False, return_average=True):
        super().__init__()
        self.name = "self_intersection_energy"
        self.weight = weight
        self.norm = norm
        self.area = area

    def calc_single(self, example):
        v = example["cloth"].pred_pos
        faces = example["cloth"].closed_faces
        f_adj_dict = example["cloth"].f_adj_dict
        boundary_paths = example["cloth"].boundary_paths
        garment_name = example.garment_name
        device = v.device

        if garment_name == 'tshirt_unzipped':
            # print("tshirt_unzipped skipped")
            return torch.tensor(0.0).to(device)

        for boundary_path in boundary_paths:
            centre_v = torch.mean(v[boundary_path], dim=0, keepdim=True)
            v = torch.vstack([v, centre_v])

        numpy_pos = (v.detach().cpu().numpy().copy()).astype(np.float64)
        # faces = faces.detach().cpu().numpy().copy()

        # obj_file = "/userhome/cs/wsn1226/HOOD/HOOD/criterions/postcvpr/mataug/test.obj"
        # boundary_paths = [(torch.tensor(l)).to(device) for l in boundary_paths]

        # writeOBJ(obj_file, numpy_pos, faces)
        # sys.exit(0)
        (
            ff,
            new_v_indices,
            new_v_bary,
            inner_f_one_batch,
        ) = compute_self_intersection_one_batch(numpy_pos, faces, f_adj_dict)
        if inner_f_one_batch:
            ff = torch.from_numpy(ff)
            ff_append = torch.zeros(1, 3, dtype = ff.dtype)
            ff = torch.vstack([ff, ff_append]).to(device)

            new_v_indices = torch.from_numpy(new_v_indices).to(device)
            new_v_bary = torch.from_numpy(new_v_bary).to(device)

            unassembled_v = v[new_v_indices]
            new_bary = new_v_bary.unsqueeze(2).repeat(1, 1, 3)
            new_v = torch.sum(unassembled_v * new_bary, dim = 1)
            final_v = torch.vstack([v, new_v])

            list_of_tensors = [torch.tensor(l) for l in inner_f_one_batch]
            padded_tensor = pad_sequence(list_of_tensors, batch_first=True, padding_value = ff.shape[0] - 1)

            inner_vert_ids = ff[padded_tensor]
            p0 = final_v[inner_vert_ids[:, :, 0]]
            p1 = final_v[inner_vert_ids[:, :, 1]]
            p2 = final_v[inner_vert_ids[:, :, 2]]
            if not self.area:
                cross12 = torch.cross(p1, p2, dim = -1)
                if self.norm == 'l1':
                    loss = self.weight * torch.sum(torch.abs(torch.sum(p0 * cross12, dim = [-1, -2])))
                elif self.norm == 'l2':
                    loss = self.weight * torch.sum(torch.square(torch.sum(p0 * cross12, dim = [-1, -2])))
                else:
                    raise ValueError(f"Unknown norm {self.norm}")
            else:
                v01 = p1 - p0
                v12 = p2 - p1
                loss = torch.sum(torch.norm(torch.cross(v01, v12, dim=-1), dim=-1), [-1, -2])
            # if loss>100:
                # writeOBJ(obj_file, numpy_pos, faces)
        else:
            loss = torch.tensor(0.0).to(device)
        # print(loss)
        return loss

    def forward(self, sample):
        loss_list = []
        B = sample.num_graphs
        for i in range(B):
            loss_sample = self.calc_single(sample.get_example(i))
            loss_list.append(loss_sample)

        loss = sum(loss_list) / B

        return dict(loss=loss)
