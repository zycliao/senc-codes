import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional
from copy import deepcopy

import numpy as np
import torch
from huepy import yellow
from omegaconf import II
from omegaconf.dictconfig import DictConfig
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from tensorboardX import SummaryWriter
import wandb

from runners.utils.collector import SampleCollector
from runners.utils.collision import CollisionPreprocessor
from runners.utils.material import RandomMaterial
from utils.cloth_and_material import FaceNormals, ClothMatAug
from utils.common import move2device, save_checkpoint, add_field_to_pyg_batch, NodeType
from utils.defaults import DEFAULTS
from criterions.postcvpr.collision_penalty import collision_handling
from utils.mesh_io import writePC2, save_obj_mesh
from runners.postcvpr import Runner as BaseRunner
from runners.postcvpr import create_optimizer, run_epoch

from pytorch3d.ops import knn_points


@dataclass
class MaterialConfig:
    density_min: float = 0.20022            # minimal density to sample from (used to compute nodal masses)
    density_max: float = 0.20022            # maximal density to sample from (used to compute nodal masses)
    lame_mu_min: float = 23600.0            # minimal shear modulus to sample from
    lame_mu_max: float = 23600.0            # maximal shear modulus to sample from
    lame_lambda_min: float = 44400          # minimal Lame's lambda to sample from
    lame_lambda_max: float = 44400          # maximal Lame's lambda to sample from
    bending_coeff_min: float = 3.96e-05     # minimal bending coefficient to sample from
    bending_coeff_max: float = 3.96e-05     # maximal bending coefficient to sample from
    bending_multiplier: float = 1.          # multiplier for bending coefficient
    energy_type_prob: Optional[str] = None

    stiffness_min: float = 8
    stiffness_max: float = 30
    k_stretch_min: float = 15
    k_stretch_max: float = 50
    k_shear_min: float = 0.3
    k_shear_max: float = 10
    wind_min: float = 0
    wind_max: float = 5

    density_override: Optional[float] = None        # if set, overrides the sampled density (used in validation)
    lame_mu_override: Optional[float] = None        # if set, overrides the sampled shear modulus (used in validation)
    lame_lambda_override: Optional[float] = None    # if set, overrides the sampled Lame's lambda (used in validation)
    bending_coeff_override: Optional[float] = None  # if set, overrides the sampled bending coefficient (used in validation)
    stiffness_override: Optional[float] = None      # if set, overrides the sampled stiffness (used in validation)
    k_stretch_override: Optional[float] = None      # if set, overrides the k_stretch (used in validation)
    k_shear_override: Optional[float] = None        # if set, overrides the k_shear (used in validation)
    wind_override: Optional[tuple] = None           # if set, overrides the wind (used in validation)
    # energy_type_override: Optional[str] = None      # if set, overrides the energy type (used in validation)
    energy_type_prob_override: Optional[str] = None  # if set, overrides the energy type (used in validation)


@dataclass
class OptimConfig:
    lr: float = 1e-4                # initial learning rate
    decay_rate: float = 1e-1        # decay multiplier for the scheduler
    decay_min: float = 0            # minimal decay
    decay_steps: int = 5_000_000    # number of steps for one decay step
    step_start: int = 0             # step to start from (used to resume training)


@dataclass
class Config:
    optimizer: OptimConfig = OptimConfig()
    material: MaterialConfig = MaterialConfig()
    warmup_steps: int = 100                 # number of steps to warm up the normalization statistics
    increase_roll_every: int = 5000         # we start from predicting only one step, then increase the number of steps each `increase_roll_every` steps
    roll_max: int = 5                       # maximum number of steps to predict
    push_eps: float = 2e-3                  # threshold for collision solver, we apply it once before the first step
    grad_clip: Optional[float] = 1.         # if set, clips the gradient norm to this value
    overwrite_pos_every_step: bool = False  # if true, the canonical poses of each garment are not cached
    self_col: bool = False
    use_one_ring: bool = False  # if true, uses one ring neighbors for the space graph

    # In the paper, the difference between the initial and regular time steps is explained with alpha coeffitient in the inertia loss term
    initial_ts: float = 1 / 3   # time between the first two steps in training, used to allow the model to faster reach static equiliblium
    regular_ts: float = 1 / 30  # time between the regular steps in training and validation

    force_names: tuple = ('ext_force_energy', 'gravitational_energy')

    device: str = II('device')


def is_debug():
    gettrace = getattr(sys, 'gettrace', None)
    return gettrace is not None and gettrace()


class Runner(BaseRunner):
    def __init__(self, model: nn.Module, criterion_dict: Dict[str, nn.Module], mcfg: DictConfig, create_wandb=False):
        super().__init__(model, criterion_dict, mcfg, create_wandb=create_wandb)

        self.model = model
        self.criterion_dict = criterion_dict
        self.mcfg = mcfg

        self.cloth_obj = ClothMatAug(None, always_overwrite_mass=True, self_col=mcfg.self_col)
        self.normals_f = FaceNormals()

        self.sample_collector = SampleCollector(mcfg)
        self.collision_solver = CollisionPreprocessor(mcfg)
        self.random_material = RandomMaterial(mcfg.material)

        # if create_wandb:
        #     wandb.login()
        #     self.wandb_run = wandb.init(project='HOOD')
        self.writer = None
        self.debug = is_debug()

    def _rollout(self, sequence, n_steps, progressbar=False, bare=False, material_dict=None):
        trajectory = []
        obstacle_trajectory = []

        metrics_dict = defaultdict(list)

        pbar = range(0, n_steps)
        if progressbar:
            pbar = tqdm(pbar)

        prev_out_dict = None
        for i in pbar:
            state = self.collect_sample_wholeseq(sequence, i, prev_out_dict, material_dict=material_dict)

            if i == 0:
                state = self.collision_solver.solve(state)

            state = add_field_to_pyg_batch(state, 'pred_pos', state['cloth'].pos, 'cloth', 'pos')
            state['cloth'].pred_pos.requires_grad = True
            # internal_force_names = ['stretching_energy', 'bending_energy']
            torch.set_grad_enabled(True)
            energy = 0
            if material_dict is not None and 'ext_force' in material_dict:
                state['cloth'].wind = material_dict['ext_force'][i:i+1]
            for k in self.mcfg.force_names:
            # for k in ['ext_force_energy', 'gravitational_energy']:
                energy += self.criterion_dict[k](state)['loss']
            denergy_dx = torch.autograd.grad(energy, state['cloth'].pred_pos)[0]
            acc = -denergy_dx / state['cloth'].v_mass
            state['cloth'].pred_pos.grad = None
            state = add_field_to_pyg_batch(state, 'explicit_acc', acc, 'cloth', 'pos')
            torch.set_grad_enabled(False)

            with torch.no_grad():
                state = self.add_space_neighbors(state)
                state = self.model(state, is_training=False)

            trajectory.append(state['cloth'].pred_pos.detach().cpu().numpy())
            obstacle_trajectory.append(state['obstacle'].target_pos.detach().cpu().numpy())

            # save obj
            # if i == 230:

                # save_obj_mesh('230.obj', trajectory[-1], sequence['cloth'].faces_batch.T.cpu().numpy())
                # exit()

            if not bare:
                loss_dict, per_vert_dict = self.criterion_pass(state)
                for k, v in loss_dict.items():
                    metrics_dict[k].append(v.item())
                for k, v in per_vert_dict.items():
                    metrics_dict[k].append(v.detach().cpu().numpy())
            prev_out_dict = state.clone()
        return trajectory, obstacle_trajectory, metrics_dict

    def add_space_neighbors(self, sample):
        pos = sample['cloth'].pos[None]
        t0 = time.time()
        K = 10
        with torch.no_grad():
            dist, idx, _ = knn_points(pos, pos, K=K + 1, return_sorted=False, return_nn=False)
            dist = torch.sqrt(dist)
            # idx = idx[0, :, 1:]
            # dist = dist[0, :, 1:]
            idx = idx[0]
            dist = dist[0]

            node_indices = torch.arange(idx.shape[0])[:, None].repeat(1, K + 1).to(idx.device)
            mask = torch.logical_and(dist < 0.02, dist > 1e-7)
            idx = idx[mask]
            node_indices = node_indices[mask]
            new_edges = torch.stack([node_indices, idx], dim=0)
            new_edges = torch.sort(new_edges, dim=0)[0]
            new_edges = new_edges.unique(dim=1)

            # remove edges that already exist
            if self.mcfg.use_one_ring:
                new_edges_filtered = new_edges
            else:
                old_edges = sample['cloth', 'mesh_edge', 'cloth'].edge_index
                old_edges = torch.sort(old_edges, dim=0)[0]
                radix = 30000
                new_edges_unique = new_edges[0] * radix + new_edges[1]
                old_edges_unique = old_edges[0] * radix + old_edges[1]
                mask = torch.logical_not(torch.isin(new_edges_unique, old_edges_unique))
                new_edges_filtered = new_edges[:, mask]

            # when adding this, need to update _inc_dict and _slice_dict
            # ['cloth', 'mesh_edge', 'cloth'] doesn't need because it is created in the Dataset
            # Dataloader will automatically make it a part of the batch, and udpat the _inc_dict and _slice_dict
            sample['cloth', 'space_edge', 'cloth'].edge_index = new_edges_filtered
            device = new_edges_filtered.device
            sample._inc_dict['cloth', 'space_edge', 'cloth'] = \
                dict(edge_index=torch.zeros([1, 2, 1], dtype=torch.long).to(device))
            sample._slice_dict['cloth', 'space_edge', 'cloth'] = \
                dict(edge_index=torch.tensor([0, new_edges_filtered.shape[1]], dtype=torch.long).to(device))

            t = time.time() - t0
            # print(f"KNN time: {t:.5f}")
        return sample

    def forward(self, sample, roll_steps=1, optimizer=None, scheduler=None) -> dict:

        # for the first 5000 steps, we randomly chose between initial and regular timesteps so that model does not overfit
        # Then, we always use initial timestep for the first frame and regular timestep for the rest of the frames
        random_ts = (roll_steps == 1)

        # add
        t0 = time.time()
        sample = self.add_cloth_obj(sample)
        # print(f'add_cloth_obj: {time.time() - t0:.6f}')

        prev_out_sample = None
        for i in range(roll_steps):
            sample_step = self.collect_sample(sample, i, prev_out_sample, random_ts=random_ts)

            if i == 0:
                sample_step = self.collision_solver.solve(sample_step)

            # after iteration >= 10000, i.e., roll_steps >= 3, explicit_acc appears, which will cause error later
            # not sure the reason yet
            if 'explicit_acc' in sample_step._slice_dict['cloth']:
                sample_step._slice_dict['cloth'].pop('explicit_acc')
            if 'explicit_acc' in sample_step._inc_dict['cloth']:
                sample_step._inc_dict['cloth'].pop('explicit_acc')

            sample_step = add_field_to_pyg_batch(sample_step, 'pred_pos', sample_step['cloth'].pos, 'cloth', 'pos')
            sample_step['cloth'].pred_pos.requires_grad = True
            # internal_force_names = ['stretching_energy', 'bending_energy']
            energy = 0
            for k in self.mcfg.force_names:
                energy += self.criterion_dict[k](sample_step)['loss']
            denergy_dx = torch.autograd.grad(energy, sample_step['cloth'].pred_pos)[0]
            acc = -denergy_dx / sample_step['cloth'].v_mass
            sample_step['cloth'].pred_pos.grad = None
            sample_step = add_field_to_pyg_batch(sample_step, 'explicit_acc', acc, 'cloth', 'pos')


            sample_step = self.add_space_neighbors(sample_step)
            sample_step = self.model(sample_step)
            loss_dict, _ = self.criterion_pass(sample_step)
            prev_out_sample = sample_step.detach()

            self.optimizer_step(loss_dict, optimizer, scheduler)

        ld_to_write = {k: v.item() for k, v in loss_dict.items()}
        end = time.time()
        # print(f'forward time: {end - t0:.6f}')
        return ld_to_write

