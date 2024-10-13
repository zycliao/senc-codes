import os
import pickle
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
from matplotlib import pyplot as plt

from runners.utils.collector import SampleCollector
from runners.utils.collision import CollisionPreprocessor
from runners.utils.material import RandomMaterial
from utils.cloth_and_material import FaceNormals, ClothMatAug
from utils.common import move2device, save_checkpoint, add_field_to_pyg_batch, NodeType
from utils.defaults import DEFAULTS
from criterions.postcvpr.collision_penalty import collision_handling
from utils.mesh_io import writePC2, save_obj_mesh


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

    use_one_ring: bool = False            # if true, uses one ring neighbors for the space graph

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

    # In the paper, the difference between the initial and regular time steps is explained with alpha coeffitient in the inertia loss term
    initial_ts: float = 1 / 3   # time between the first two steps in training, used to allow the model to faster reach static equiliblium
    regular_ts: float = 1 / 30  # time between the regular steps in training and validation

    device: str = II('device')


def is_debug():
    gettrace = getattr(sys, 'gettrace', None)
    return gettrace is not None and gettrace()


class Runner(nn.Module):
    def __init__(self, model: nn.Module, criterion_dict: Dict[str, nn.Module], mcfg: DictConfig, create_wandb=False):
        super().__init__()

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

    def setup_writer(self, log_dir):
        self.writer = SummaryWriter(log_dir=log_dir)

    def rollout_material(self, sequence, prev_out_dict=None, material_dict=None, start_step=0, n_steps=-1, bare=False, record_time=False,
                         ext_force=None):
        """
        Generates a trajectory for the full sequence, use this function for inference

        :param sequence: torch geometric batch with the sequence
        :param material_dict: dict with material properties, the value should be torch.Tensor
        :param n_steps: number of steps to predict, if -1, predicts for the full sequence
        :param bare: if true, the loss terms are not computed, use for faster validation
        :param record_time: if true, records the time spent on the forward pass and adds it to trajectories_dicts['metrics']
        :return: trajectories_dicts: dict with the following fields:
            'pred': np.ndarray (NxVx3) predicted cloth trajectory
            'obstacle': np.ndarray (NxWx3) body trajectory
            'cloth_faces': np.ndarray (Fx3) cloth faces
            'obstacle_faces': np.ndarray (Fox3) body faces
            'metrics': dictionary with by-frame loss values
        """
        with torch.no_grad():
            sequence = add_field_to_pyg_batch(sequence, 'iter', [0], 'cloth', reference_key=None)

        n_samples = sequence['obstacle'].pos.shape[1]
        if n_steps > 0:
            n_samples = min(n_samples, n_steps)

        trajectories_dicts = defaultdict(lambda: defaultdict(list))

        if record_time:
            st_time = time.time()

        # trajectory, obstacle_trajectory, metrics_dict = self._rollout(sequence, n_samples,
        #                                                               progressbar=True, bare=bare)

        trajectory = []
        obstacle_trajectory = []

        metrics_dict = defaultdict(list)

        for i in range(start_step, start_step+n_samples):
            state = self.collect_sample_wholeseq(sequence, i, prev_out_dict, material_dict=material_dict)
            state = self.model(state, is_training=False, ext_force=ext_force)

            trajectory.append(state['cloth'].pred_pos)
            obstacle_trajectory.append(state['obstacle'].target_pos.detach().cpu().numpy())

            if not bare:
                loss_dict, _ = self.criterion_pass(state)
                for k, v in loss_dict.items():
                    metrics_dict[k].append(v.item())
            prev_out_dict = state.clone()

        self.state = state
        if record_time:
            total_time = time.time() - st_time
            metrics_dict['time'] = total_time

        trajectories_dicts['pred'] = trajectory
        trajectories_dicts['obstacle'] = obstacle_trajectory
        trajectories_dicts['metrics'] = dict(metrics_dict)
        trajectories_dicts['cloth_faces'] = sequence['cloth'].faces_batch.T.cpu().numpy()
        trajectories_dicts['obstacle_faces'] = sequence['obstacle'].faces_batch.T.cpu().numpy()

        for s in ['obstacle']:
            trajectories_dicts[s] = np.stack(trajectories_dicts[s], axis=0)
        trajectories_dicts['pred'] = trajectories_dicts['pred'][0]
        return trajectories_dicts, prev_out_dict

    def forward_simulation(self, sequence, material_dict=None, start_step=0, n_steps=-1, init_cloth_pos=None,
                           explicit_solve_collision=False, record_time=False, verbose=False):
        """
        Generates a trajectory for the full sequence, use this function for inference

        :param sequence: torch geometric batch with the sequence
        :param material_dict: dict with material properties, the value should be torch.Tensor
        :param n_steps: number of steps to predict, if -1, predicts for the full sequence
        :param bare: if true, the loss terms are not computed, use for faster validation
        :param record_time: if true, records the time spent on the forward pass and adds it to trajectories_dicts['metrics']
        :return: trajectories_dicts: dict with the following fields:
            'pred': np.ndarray (NxVx3) predicted cloth trajectory
            'obstacle': np.ndarray (NxWx3) body trajectory
            'cloth_faces': np.ndarray (Fx3) cloth faces
            'obstacle_faces': np.ndarray (Fox3) body faces
            'metrics': dictionary with by-frame loss values
        """
        sequence = add_field_to_pyg_batch(sequence, 'iter', [0], 'cloth', reference_key=None)
        sequence = self.add_cloth_obj(sequence, material_dict)
        vertex_type = sequence['cloth'].vertex_type
        pinned_mask = vertex_type == NodeType.HANDLE

        n_samples = sequence['obstacle'].pos.shape[1]
        if n_steps > 0:
            n_samples = min(n_samples, n_steps)

        trajectories_dicts = defaultdict(lambda: defaultdict(list))

        if record_time:
            st_time = time.time()

        obstacle_trajectory = []
        trajectory = []

        # trajectory, obstacle_trajectory, metrics_dict = self._rollout(sequence, n_samples,
        #                                                               progressbar=True, bare=bare)



        metrics_dict = defaultdict(list)

        progressbar = True
        pbar = range(start_step, start_step+n_samples)
        if progressbar:
            pbar = tqdm(pbar)

        if init_cloth_pos is not None:
            pred_pos_init = init_cloth_pos
        else:
            pred_pos_init = sequence['cloth'].pos[:, 0].clone().detach()

        prev_out_dict = None
        for i in pbar:
            with torch.no_grad():
                if i > 0:
                    if 'pred_pos' in sequence._slice_dict['cloth']:
                        sequence._slice_dict['cloth'].pop('pred_pos')
                    if 'pred_velocity' in sequence._slice_dict['cloth']:
                        sequence._slice_dict['cloth'].pop('pred_velocity')
                    if 'pred_pos' in sequence._inc_dict['cloth']:
                        sequence._inc_dict['cloth'].pop('pred_pos')
                    if 'pred_velocity' in sequence._inc_dict['cloth']:
                        sequence._inc_dict['cloth'].pop('pred_velocity')
                state = self.collect_sample_wholeseq(sequence, i, prev_out_dict)

            pred_pos_init = collision_handling(pred_pos_init, state['obstacle'].target_pos, state['obstacle'].faces_batch.T,
                                               self.normals_f, eps=4e-3)

            pred_pos = torch.tensor(pred_pos_init, dtype=torch.float32, device=self.mcfg.device, requires_grad=True)
            optimizer = torch.optim.Adam([pred_pos], lr=0.0003)

            min_loss = np.inf
            min_iter = -1
            for i_iter in range(3000):
                optimizer.zero_grad()

                pinned_pos = pred_pos * torch.logical_not(pinned_mask) + state['cloth'].target_pos * pinned_mask
                if i_iter == 0:
                    state = add_field_to_pyg_batch(state, 'pred_pos', pinned_pos, 'cloth', 'pos')
                else:
                    state['cloth'].pred_pos = pinned_pos

                if i_iter == 0:
                    use_cache = False
                else:
                    use_cache = True

                loss_dict, per_vert_dict = self.criterion_pass(state, use_cache=use_cache, no_self=True)
                loss = 0
                print_info = f"Iter {i_iter}, "
                for k, v in loss_dict.items():
                    loss += v
                    print_info += f"{k}: {v.item():.5f}, "
                loss.backward()

                pred_pos.grad = torch.where(torch.isnan(pred_pos.grad),
                                            torch.zeros_like(pred_pos.grad), pred_pos.grad)

                optimizer.step()

                if verbose:
                    if i_iter % 50 == 0:
                        print(print_info)

                loss_val = loss.detach().cpu().item()
                # print(f'iter {i_iter}, loss {loss_val}')
                if loss_val < min_loss:
                    min_loss = loss_val
                    min_iter = i_iter
                if i_iter - min_iter > 30:
                    break

            if verbose:
                print("Start to optimize self collision")
            # optimize self collision
            min_loss = np.inf
            min_iter = -1
            for i_iter in range(1000):
                optimizer.zero_grad()

                pinned_pos = pred_pos * torch.logical_not(pinned_mask) + state['cloth'].target_pos * pinned_mask
                if i_iter == 0:
                    state = add_field_to_pyg_batch(state, 'pred_pos', pinned_pos, 'cloth', 'pos')
                else:
                    state['cloth'].pred_pos = pinned_pos

                if i_iter == 0:
                    use_cache = False
                else:
                    use_cache = True

                loss_dict, per_vert_dict = self.criterion_pass(state, use_cache=use_cache, no_self=False)
                loss = 0
                print_info = f"Iter {i_iter}, "
                for k, v in loss_dict.items():
                    loss += v
                    print_info += f"{k}: {v.item():.5f}, "
                loss.backward()

                self_loss = loss_dict['self_intersection_energy_loss'].detach().cpu().item()

                pred_pos.grad = torch.where(torch.isnan(pred_pos.grad),
                                            torch.zeros_like(pred_pos.grad), pred_pos.grad)

                optimizer.step()

                if verbose:
                    if i_iter % 10 == 0:
                        print(print_info)

                if self_loss < min_loss:
                    min_loss = self_loss
                    min_iter = i_iter
                if i_iter - min_iter > 30 or self_loss < 1e-3:
                    break

            with torch.no_grad():
                pred_pos = pred_pos * torch.logical_not(pinned_mask) + state['cloth'].target_pos * pinned_mask

                if explicit_solve_collision:
                    pred_pos = collision_handling(pred_pos, state['obstacle'].target_pos, state['obstacle'].faces_batch.T,
                                                  self.normals_f, eps=4e-3)
                state['cloth'].pred_pos = pred_pos
                pred_velocity = state['cloth'].pred_pos - state['cloth'].pos
                pred_pos_init = 2 * pred_pos - state['cloth'].pos
                state = add_field_to_pyg_batch(state, 'pred_velocity', pred_velocity, 'cloth', 'pos')

                if i < start_step + n_samples - 1:
                    sequence['cloth'].pos[:, i + 1] = pred_pos

                trajectory.append(state['cloth'].pred_pos.detach().cpu().numpy())
                obstacle_trajectory.append(state['obstacle'].target_pos.detach().cpu().numpy())
                for k, v in loss_dict.items():
                    metrics_dict[k].append(v.item())
                for k, v in per_vert_dict.items():
                    metrics_dict[k].append(v.detach().cpu().numpy())

                prev_out_dict = state.clone()


        if record_time:
            total_time = time.time() - st_time
            metrics_dict['time'] = total_time

        trajectories_dicts['pred'] = trajectory
        trajectories_dicts['obstacle'] = obstacle_trajectory
        trajectories_dicts['metrics'] = dict(metrics_dict)
        trajectories_dicts['cloth_faces'] = sequence['cloth'].faces_batch.T.cpu().numpy()
        trajectories_dicts['obstacle_faces'] = sequence['obstacle'].faces_batch.T.cpu().numpy()

        for s in ['pred', 'obstacle']:
            trajectories_dicts[s] = np.stack(trajectories_dicts[s], axis=0)
        return trajectories_dicts


    def forward_simulation_lbfgs(self, sequence, material_dict=None, start_step=0, n_steps=-1, init_cloth_pos=None,
                           bare=False, record_time=False, var_wind=None):
        """
        Generates a trajectory for the full sequence, use this function for inference

        :param sequence: torch geometric batch with the sequence
        :param material_dict: dict with material properties, the value should be torch.Tensor
        :param n_steps: number of steps to predict, if -1, predicts for the full sequence
        :param bare: if true, the loss terms are not computed, use for faster validation
        :param record_time: if true, records the time spent on the forward pass and adds it to trajectories_dicts['metrics']
        :return: trajectories_dicts: dict with the following fields:
            'pred': np.ndarray (NxVx3) predicted cloth trajectory
            'obstacle': np.ndarray (NxWx3) body trajectory
            'cloth_faces': np.ndarray (Fx3) cloth faces
            'obstacle_faces': np.ndarray (Fox3) body faces
            'metrics': dictionary with by-frame loss values
        """
        sequence = add_field_to_pyg_batch(sequence, 'iter', [0], 'cloth', reference_key=None)
        sequence = self.add_cloth_obj(sequence, material_dict)
        vertex_type = sequence['cloth'].vertex_type
        pinned_mask = vertex_type == NodeType.HANDLE

        n_samples = sequence['obstacle'].pos.shape[1]
        if n_steps > 0:
            n_samples = min(n_samples, n_steps)

        trajectories_dicts = defaultdict(lambda: defaultdict(list))

        if record_time:
            st_time = time.time()

        # trajectory, obstacle_trajectory, metrics_dict = self._rollout(sequence, n_samples,
        #                                                               progressbar=True, bare=bare)

        trajectory = []
        obstacle_trajectory = []

        metrics_dict = defaultdict(list)

        progressbar = True
        pbar = range(start_step, start_step+n_samples)
        if progressbar:
            pbar = tqdm(pbar)

        if init_cloth_pos is not None:
            pred_pos_init = init_cloth_pos
        else:
            pred_pos_init = sequence['cloth'].pos[:, 0].clone().detach()

        prev_out_dict = None
        for i in pbar:
            if var_wind is not None:
                sequence['cloth'].wind = var_wind[i: i+1]

            pred_pos = torch.tensor(pred_pos_init, dtype=torch.float32, device=self.mcfg.device, requires_grad=True)
            optimizer = torch.optim.LBFGS([pred_pos], lr=1, max_iter=1000, history_size=100, tolerance_grad=1e-6,
                                          tolerance_change=1e-10)

            with torch.no_grad():
                if i > 0:
                    if 'pred_pos' in sequence._slice_dict['cloth']:
                        sequence._slice_dict['cloth'].pop('pred_pos')
                    if 'pred_velocity' in sequence._slice_dict['cloth']:
                        sequence._slice_dict['cloth'].pop('pred_velocity')
                    if 'pred_pos' in sequence._inc_dict['cloth']:
                        sequence._inc_dict['cloth'].pop('pred_pos')
                    if 'pred_velocity' in sequence._inc_dict['cloth']:
                        sequence._inc_dict['cloth'].pop('pred_velocity')
                state_ = self.collect_sample_wholeseq(sequence, i, prev_out_dict)
                self.lbfgs_var = {'state': state_, 'loss_dict': {}, 'per_vert_dict': {}}



            for i_iter in range(3):
                def closure():
                    optimizer.zero_grad()
                    state_ = self.lbfgs_var['state']
                    pinned_pos = pred_pos * torch.logical_not(pinned_mask) + state_['cloth'].target_pos * pinned_mask
                    state_ = add_field_to_pyg_batch(state_, 'pred_pos', pinned_pos, 'cloth', 'pos')
                    self.lbfgs_var['state'] = state_

                    loss_dict, per_vert_dict = self.criterion_pass(self.lbfgs_var['state'])
                    self.lbfgs_var['loss_dict'] = loss_dict
                    self.lbfgs_var['per_vert_dict'] = per_vert_dict
                    loss = 0
                    print_info = f"Iter {i_iter}, "
                    for k, v in loss_dict.items():
                        loss += v
                        print_info += f"{k}: {v.item():.5f}, "
                    loss.backward()
                    return loss

                optimizer.step(closure)


            state = self.lbfgs_var['state']
            with torch.no_grad():
                pred_pos = pred_pos * torch.logical_not(pinned_mask) + state['cloth'].target_pos * pinned_mask
                state['cloth'].pred_pos = pred_pos
                pred_velocity = state['cloth'].pred_pos - state['cloth'].pos
                pred_pos_init = 2 * pred_pos - state['cloth'].pos
                state = add_field_to_pyg_batch(state, 'pred_velocity', pred_velocity, 'cloth', 'pos')

                if i < start_step + n_samples - 1:
                    sequence['cloth'].pos[:, i + 1] = pred_pos

                # trajectory.append(state['cloth'].pred_pos.detach().cpu().numpy())
                trajectory.append(state['cloth'].pos.detach().cpu().numpy())

                obstacle_trajectory.append(state['obstacle'].target_pos.detach().cpu().numpy())
                for k, v in self.lbfgs_var['loss_dict'].items():
                    metrics_dict[k].append(v.item())
                for k, v in self.lbfgs_var['per_vert_dict'].items():
                    metrics_dict[k].append(v.detach().cpu().numpy())

                prev_out_dict = state.clone()


        if record_time:
            total_time = time.time() - st_time
            # metrics_dict['time'] = total_time

        trajectories_dicts['pred'] = trajectory
        trajectories_dicts['obstacle'] = obstacle_trajectory
        trajectories_dicts['metrics'] = dict(metrics_dict)
        trajectories_dicts['cloth_faces'] = sequence['cloth'].faces_batch.T.cpu().numpy()
        trajectories_dicts['obstacle_faces'] = sequence['obstacle'].faces_batch.T.cpu().numpy()

        for s in ['pred', 'obstacle']:
            trajectories_dicts[s] = np.stack(trajectories_dicts[s], axis=0)
        return trajectories_dicts


    def sequence_criterion(self, pred_sequence, sequence, material_dict=None, start_step=0, n_steps=-1, gt_sequence=None):
        """
        Generates a trajectory for the full sequence, use this function for inference

        :param sequence: torch geometric batch with the sequence
        :param material_dict: dict with material properties, the value should be torch.Tensor
        :param n_steps: number of steps to predict, if -1, predicts for the full sequence
        :param bare: if true, the loss terms are not computed, use for faster validation
        :param record_time: if true, records the time spent on the forward pass and adds it to trajectories_dicts['metrics']
        :return: trajectories_dicts: dict with the following fields:
            'pred': np.ndarray (NxVx3) predicted cloth trajectory
            'obstacle': np.ndarray (NxWx3) body trajectory
            'cloth_faces': np.ndarray (Fx3) cloth faces
            'obstacle_faces': np.ndarray (Fox3) body faces
            'metrics': dictionary with by-frame loss values
        """
        torch.set_grad_enabled(False)
        sequence = add_field_to_pyg_batch(sequence, 'iter', [0], 'cloth', reference_key=None)
        sequence = self.add_cloth_obj(sequence, material_dict)

        n_samples = sequence['obstacle'].pos.shape[1]
        if n_steps > 0:
            n_samples = min(n_samples, n_steps)

        metrics_dict = defaultdict(list)
        pbar = range(start_step, start_step+n_samples)

        if not torch.is_tensor(pred_sequence):
            pred_sequence = torch.tensor(pred_sequence, dtype=torch.float32, device=self.mcfg.device)

        if gt_sequence is not None:
            if not torch.is_tensor(gt_sequence):
                gt_sequence = torch.tensor(gt_sequence, dtype=torch.float32, device=self.mcfg.device)
            dist = torch.mean(torch.norm(pred_sequence - gt_sequence, dim=-1))
            metrics_dict['dist'] = dist.cpu().item()

        prev_out_dict = None
        for i in pbar:
            pred_pos = pred_sequence[i]

            if i > 0:
                if 'pred_pos' in sequence._slice_dict['cloth']:
                    sequence._slice_dict['cloth'].pop('pred_pos')
                if 'pred_velocity' in sequence._slice_dict['cloth']:
                    sequence._slice_dict['cloth'].pop('pred_velocity')
                if 'pred_pos' in sequence._inc_dict['cloth']:
                    sequence._inc_dict['cloth'].pop('pred_pos')
                if 'pred_velocity' in sequence._inc_dict['cloth']:
                    sequence._inc_dict['cloth'].pop('pred_velocity')
            state = self.collect_sample_wholeseq(sequence, i, prev_out_dict)

            state = add_field_to_pyg_batch(state, 'pred_pos', pred_pos, 'cloth', 'pos')

            loss_dict, per_vert_dict = self.criterion_pass(state)

            state['cloth'].pred_pos = pred_pos
            pred_velocity = state['cloth'].pred_pos - state['cloth'].pos
            # pred_pos_init = 2 * pred_pos - state['cloth'].pos
            state = add_field_to_pyg_batch(state, 'pred_velocity', pred_velocity, 'cloth', 'pos')

            if i < start_step + n_samples - 1:
                sequence['cloth'].pos[:, i + 1] = pred_pos

            for k, v in loss_dict.items():
                metrics_dict[k].append(v.item())
            for k, v in per_vert_dict.items():
                metrics_dict[k].append(v.detach().cpu().numpy())
            prev_out_dict = state.clone()

        torch.set_grad_enabled(True)
        return dict(metrics_dict)


    def valid_rollout(self, sequence, n_steps=-1, bare=False, record_time=False, material_dict=None):
        """
        Generates a trajectory for the full sequence, use this function for inference

        :param sequence: torch geometric batch with the sequence
        :param n_steps: number of steps to predict, if -1, predicts for the full sequence
        :param bare: if true, the loss terms are not computed, use for faster validation
        :param record_time: if true, records the time spent on the forward pass and adds it to trajectories_dicts['metrics']
        :return: trajectories_dicts: dict with the following fields:
            'pred': np.ndarray (NxVx3) predicted cloth trajectory
            'obstacle': np.ndarray (NxWx3) body trajectory
            'cloth_faces': np.ndarray (Fx3) cloth faces
            'obstacle_faces': np.ndarray (Fox3) body faces
            'metrics': dictionary with by-frame loss values
        """
        sequence = add_field_to_pyg_batch(sequence, 'iter', [0], 'cloth', reference_key=None)
        sequence = self.add_cloth_obj(sequence, material_dict=material_dict)

        n_samples = sequence['obstacle'].pos.shape[1]
        if n_steps > 0:
            n_samples = min(n_samples, n_steps)

        trajectories_dicts = defaultdict(lambda: defaultdict(list))

        if record_time:
            st_time = time.time()

        trajectory, obstacle_trajectory, metrics_dict = self._rollout(sequence, n_samples,
                                                                      progressbar=True, bare=bare,
                                                                      material_dict=material_dict)

        if record_time:
            total_time = time.time() - st_time
            metrics_dict['time'] = total_time

        trajectories_dicts['pred'] = trajectory
        trajectories_dicts['obstacle'] = obstacle_trajectory
        trajectories_dicts['metrics'] = dict(metrics_dict)
        trajectories_dicts['cloth_faces'] = sequence['cloth'].faces_batch.T.cpu().numpy()
        trajectories_dicts['obstacle_faces'] = sequence['obstacle'].faces_batch.T.cpu().numpy()

        for s in ['pred', 'obstacle']:
            trajectories_dicts[s] = np.stack(trajectories_dicts[s], axis=0)
        return trajectories_dicts

    def _rollout(self, sequence, n_steps, progressbar=False, bare=False, material_dict=None):
        trajectory = []
        obstacle_trajectory = []

        metrics_dict = defaultdict(list)

        pbar = range(0, n_steps)
        if progressbar:
            pbar = tqdm(pbar)

        prev_out_dict = None
        total_time = []
        for i in pbar:
            state = self.collect_sample_wholeseq(sequence, i, prev_out_dict, material_dict=material_dict)

            if i == 0:
                state = self.collision_solver.solve(state)

            t0 = time.time()
            with torch.no_grad():
                state = self.model(state, is_training=False)
            t1 = time.time()
            total_time.append(t1 - t0)

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

    def calc_metric(self, pred_seq, sequence, material_dict=None, progressbar=True):
        trajectory = []
        obstacle_trajectory = []

        metrics_dict = defaultdict(list)
        sequence = add_field_to_pyg_batch(sequence, 'iter', [0], 'cloth', reference_key=None)
        gt_len = sequence['cloth'].pos.shape[1]
        pred_len = pred_seq.shape[0]
        # assert gt_len == pred_len, f'gt_len {gt_len}, pred_len {pred_len}'

        pbar = range(0, pred_len)
        if progressbar:
            pbar = tqdm(pbar)

        prev_out_dict = None
        for i in pbar:
            state = self.collect_sample_wholeseq(sequence, i, prev_out_dict, material_dict=material_dict)

            if i == 0:
                state = self.collision_solver.solve(state)

            # with torch.no_grad():
            #     state = self.model(state, is_training=False)

            add_field_to_pyg_batch(state, 'pred_pos', pred_seq[i], 'cloth', 'pos')
            add_field_to_pyg_batch(state, 'pred_velocity', pred_seq[i] - state['cloth'].pos, 'cloth', 'pos')

            trajectory.append(state['cloth'].pred_pos.detach().cpu().numpy())
            obstacle_trajectory.append(state['obstacle'].target_pos.detach().cpu().numpy())

            loss_dict, per_vert_dict = self.criterion_pass(state)
            for k, v in loss_dict.items():
                metrics_dict[k].append(v.item())
            for k, v in per_vert_dict.items():
                metrics_dict[k].append(v.detach().cpu().numpy())
            prev_out_dict = state.clone()
        return metrics_dict

    def collect_sample_wholeseq(self, sequence, index, prev_out_dict, material_dict=None):

        """
        Collects a sample from the sequence, given the previous output and the index of the current step
        This function is only used in validation
        For training, see `collect_sample`

        :param sequence: torch geometric batch with the sequence
        :param index: index of the current step
        :param prev_out_dict: previous output of the model

        """
        sample_step = sequence.clone()
        sample_step = self.add_cloth_obj(sample_step, material_dict)

        # gather infor for the current step
        sample_step = self.sample_collector.sequence2sample(sample_step, index)

        # move to device
        sample_step = move2device(sample_step, self.mcfg.device)

        # coly fields from the previous step (pred_pos -> pos, pos->prev_pos)
        sample_step = self.sample_collector.copy_from_prev(sample_step, prev_out_dict)
        ts = self.mcfg.regular_ts

        # in the first step, the obstacle and positions of the pinned vertices are static
        if index == 0:
            sample_step = self.sample_collector.target2pos(sample_step)
            sample_step = self.sample_collector.pos2prev(sample_step)
            ts = self.mcfg.initial_ts
        # in the second step, we set velocities for both the cloth and the obstacle to zero
        elif index == 1:
            sample_step = self.sample_collector.pos2prev(sample_step)

        sample_step = self.sample_collector.add_velocity(sample_step, prev_out_dict)
        sample_step = self.sample_collector.add_timestep(sample_step, ts)
        return sample_step

    def set_random_material(self, sample, material_dict=None):
        """
        Add material parameters to the cloth object and the sample
        :param sample:
        :return:
        """
        sample, self.cloth_obj = self.random_material.add_material(sample, self.cloth_obj, material_dict=material_dict)
        return sample

    def add_cloth_obj(self, sample, material_dict=None):
        """
        - Updates self.cloth_obj with the cloth object in the sample
        - Samples the material properties of the cloth object and adds them to the sample
        - Adds info about the garment to the sample, which is later used by the the GNN and to compute objective terms (see utils.cloth_and_material.ClothMatAug for details)
        """
        sample = self.set_random_material(sample, material_dict=material_dict)
        sample = self.cloth_obj.set_batch(sample, overwrite_pos=self.mcfg.overwrite_pos_every_step)
        sample['cloth'].cloth_obj = self.cloth_obj
        return sample

    def criterion_pass(self, sample_step, use_cache=False, no_self=False):
        """
        Pass the sample through all the loss terms in self.criterion_dict
        and gathers their values in a dictionary
        """
        sample_step.cloth_obj = self.cloth_obj
        loss_dict = dict()
        per_vert_dict = dict()
        for criterion_name, criterion in self.criterion_dict.items():
            if no_self and criterion_name == 'self_intersection_energy':
                continue
            if criterion_name == 'collision_penalty':
                ld = criterion(sample_step, use_cache=use_cache)
            else:
                ld = criterion(sample_step)
            for k, v in ld.items():
                if k == 'per_vert':
                    per_vert_dict[f"{criterion_name}_{k}"] = v
                elif k == 'loss':
                    loss_dict[f"{criterion_name}_{k}"] = v

        return loss_dict, per_vert_dict

    def collect_sample(self, sample, idx, prev_out_dict=None, random_ts=False):
        """
        Collects a sample from the sequence, given the previous output and the index of the current step
        This function is only used in training
        For validation, see `collect_sample_wholeseq`

        :param sample: pytroch geometric batch from the dataloader
        :param idx: index of the current step
        :param prev_out_dict: previous output of the model
        :param random_ts: if True, the time step is randomly chosen between the initial and the regular time step
        :return: the sample for the current step
        """

        sample_step = sample.clone()

        # coly fields from the previous step (pred_pos -> pos, pos->prev_pos)
        sample_step = self.sample_collector.copy_from_prev(sample_step, prev_out_dict)
        ts = self.mcfg.regular_ts

        # copy positions from the lookup steps
        if idx != 0:
            sample_step = self.sample_collector.lookup2target(sample_step, idx - 1)

        # in the first step, the obstacle and positions of the pinned vertices are static
        if idx == 0:
            is_init = np.random.rand() > 0.5
            sample_step = self.sample_collector.pos2target(sample_step)
            if is_init or not random_ts:
                sample_step = self.sample_collector.pos2prev(sample_step)
                ts = self.mcfg.initial_ts
        # for the second frame, we set velocity to zero
        elif idx == 1:
            sample_step = self.sample_collector.pos2prev(sample_step)

        sample_step = self.sample_collector.add_velocity(sample_step, prev_out_dict)
        sample_step = self.sample_collector.add_timestep(sample_step, ts)
        return sample_step

    def optimizer_step(self, loss_dict, optimizer=None, scheduler=None):
        if optimizer is not None:
            loss = sum(loss_dict.values())
            optimizer.zero_grad()
            loss.backward()
            if self.mcfg.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.mcfg.grad_clip)

            optimizer.step()
            if scheduler is not None:
                scheduler.step()

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

            sample_step = self.model(sample_step)
            loss_dict, _ = self.criterion_pass(sample_step)
            prev_out_sample = sample_step.detach()

            self.optimizer_step(loss_dict, optimizer, scheduler)

        ld_to_write = {k: v.item() for k, v in loss_dict.items()}
        end = time.time()
        # print(f'forward time: {end - t0:.6f}')
        return ld_to_write

    def validation(self, val_sequences, global_step, result_dir=None):
        if result_dir is not None:
            os.makedirs(result_dir, exist_ok=True)
        self.model.eval()
        torch.set_grad_enabled(False)
        all_metrics = {}
        for i, sequence in enumerate(val_sequences):
            sequence = deepcopy(sequence)
            move2device(sequence, self.mcfg.device)
            trajectories_dicts = self.valid_rollout(sequence, material_dict=sequence['cloth'].material_dict)
            pred = trajectories_dicts['pred']
            if 'self_intersection_energy_loss' in trajectories_dicts['metrics']:
                w = self.criterion_dict['self_intersection_energy'].weight
                v = trajectories_dicts['metrics']['self_intersection_energy_loss']
                trajectories_dicts['metrics']['self_intersection_energy_loss'] = [k * 100 / w for k in v]
            if 'repulsive_energy_loss' in trajectories_dicts['metrics']:
                w = self.criterion_dict['repulsive_energy'].weight
                v = trajectories_dicts['metrics']['repulsive_energy_loss']
                trajectories_dicts['metrics']['repulsive_energy_loss'] = [k / w for k in v]

            if hasattr(sequence['cloth'], 'gt'):
                has_gt = True
                gt = sequence['cloth'].gt.cpu().numpy()
                gt = np.transpose(gt, [1, 0, 2])
            else:
                has_gt = False


            if result_dir is not None and not self.debug:
                garment_name = str(sequence['cloth'].garment_name)
                motion_name = str(sequence['cloth'].motion_name)
                writePC2(os.path.join(result_dir, f'step_{global_step}_pred_{motion_name}_{garment_name}.pc2'), pred)
                writePC2(os.path.join(result_dir, f"body_{motion_name}_{garment_name}.pc2"), trajectories_dicts['obstacle'])
                if has_gt:
                    writePC2(os.path.join(result_dir, f'gt_{motion_name}_{garment_name}.pc2'), gt)

                if 'self_intersection_energy_loss' in trajectories_dicts['metrics']:
                    self_loss = trajectories_dicts['metrics']['self_intersection_energy_loss']
                    plt.figure(figsize=(10, 5))
                    plt.plot(np.arange(len(self_loss)), self_loss, label='Self Intersection Loss')

                    # 添加标题和标签
                    plt.title('self intersection loss')
                    plt.xlabel('Frame')
                    plt.ylabel('Loss')
                    plt.legend()
                    plt.savefig(os.path.join(result_dir, f'loss_curve_{motion_name}_{garment_name}.png'), dpi=300)

            per_vert_keys = [k for k in trajectories_dicts['metrics'].keys() if k.endswith('_per_vert')]
            for k in per_vert_keys:
                trajectories_dicts['metrics'].pop(k)
            if has_gt:
                dist = np.mean(np.linalg.norm(pred - gt, axis=-1), 1)
                trajectories_dicts['metrics']['dist'] = dist * 1000.  # unit is mm
            if result_dir is not None and not self.debug:
                with open(os.path.join(result_dir, f'metric_{motion_name}_{garment_name}.pkl'), 'wb') as f:
                    pickle.dump(trajectories_dicts['metrics'], f)

            for k, v in trajectories_dicts['metrics'].items():
                if k not in all_metrics:
                    all_metrics[k] = np.array(v)
                else:
                    all_metrics[k] = np.concatenate([all_metrics[k], v], axis=0)

        log_info = ''
        for k, v in all_metrics.items():
            v = np.mean(np.array(v))

            if not self.debug and self.writer is not None:
                self.writer.add_scalar(f'val/{k}', v, global_step)
            log_info += f'Itr {global_step}. val/{k}: {v:.5f}\n'
            # self.wandb_run.log(trajectories_dicts['metr ics'], step=self.global_step)
        print(log_info)
        if result_dir is not None and not self.debug:
            with open(os.path.join(result_dir, f'val_{global_step}.txt'), 'w') as f:
                f.write(log_info)
        self.model.train()
        torch.set_grad_enabled(True)

    def validation_save_result(self, val_sequences, global_step, result_dir=None):
        if result_dir is not None:
            os.makedirs(result_dir, exist_ok=True)
        self.model.eval()
        torch.set_grad_enabled(False)
        for i, sequence in enumerate(val_sequences):
            sequence = deepcopy(sequence)
            move2device(sequence, self.mcfg.device)
            trajectories_dicts = self.valid_rollout(sequence, material_dict=sequence['cloth'].material_dict, bare=True)
            pred = trajectories_dicts['pred']

            garment_name = str(sequence['cloth'].garment_name)
            motion_name = str(sequence['cloth'].motion_name)
            writePC2(os.path.join(result_dir, f'step_{global_step}_pred_{motion_name}_{garment_name}.pc2'), pred)
            writePC2(os.path.join(result_dir, f"body_{motion_name}_{garment_name}.pc2"), trajectories_dicts['obstacle'])

        self.model.train()
        torch.set_grad_enabled(True)


    def speed_test(self, val_sequences, result_dir=None):
        if result_dir is not None:
            os.makedirs(result_dir, exist_ok=True)
        self.model.eval()
        torch.set_grad_enabled(False)

        log_info = ''
        all_time = 0
        total_samples = 0
        for i, sequence in enumerate(val_sequences):
            sequence = deepcopy(sequence)
            garment_name = str(sequence['cloth'].garment_name)
            motion_name = str(sequence['cloth'].motion_name)

            trajectories_dicts = self.valid_rollout(sequence, material_dict=sequence['cloth'].material_dict,
                                                    bare=True, record_time=True)
            t = trajectories_dicts['metrics']['time']
            n_samples = sequence['obstacle'].pos.shape[1]
            speed = n_samples / t
            total_samples += n_samples
            all_time += t

            log_info += f'{motion_name}_{garment_name}: {t:.5f}, speed: {speed:.5f} fps \n'

        log_info += f'Average speed: {total_samples / all_time:.5f} fps \n'
        print(log_info)
        if result_dir is not None and not self.debug:
            with open(os.path.join(result_dir, f'speed_test.txt'), 'w') as f:
                f.write(log_info)
        self.model.train()
        torch.set_grad_enabled(True)


def create_optimizer(training_module: Runner, mcfg: DictConfig):
    optimizer = Adam(training_module.parameters(), lr=mcfg.lr)

    def sched_fun(step):
        decay = mcfg.decay_rate ** (step // mcfg.decay_steps) + 1e-2
        decay = max(decay, mcfg.decay_min)
        return decay

    scheduler = LambdaLR(optimizer, sched_fun)
    scheduler.last_epoch = mcfg.step_start

    return optimizer, scheduler


def run_epoch(training_module: Runner, aux_modules: dict, dataloader: DataLoader,
              n_epoch: int, run_dir, cfg: DictConfig, global_step=None, val_sequences=None, only_validation=False):
    global_step = global_step or len(dataloader) * n_epoch

    optimizer = aux_modules['optimizer']
    scheduler = aux_modules['scheduler']

    prbar = tqdm(dataloader, desc=cfg.config)

    checkpoints_dir = os.path.join(run_dir, 'checkpoints')
    print(yellow(f'run_epoch started, checkpoints will be saved in {checkpoints_dir}'))

    if not hasattr(cfg.experiment, 'min_validate'):
        min_validate = -1
    else:
        min_validate = cfg.experiment.min_validate

    if (only_validation or global_step == 0) and val_sequences is not None and global_step >= min_validate:
        training_module.validation(val_sequences, global_step, result_dir=os.path.join(run_dir, 'result'))
    if only_validation:
        exit()


    for sample in prbar:
        global_step += 1
        if cfg.experiment.max_iter is not None and global_step > cfg.experiment.max_iter:
            break

        sample = move2device(sample, cfg.device)

        # add `iter` field to the sample (used in some criterions with dynamically changing weight)
        B = sample.num_graphs
        sample = add_field_to_pyg_batch(sample, 'iter', [global_step] * B, 'cloth', reference_key=None)

        # number of autoregressive steps to simulate for the training sample
        roll_steps = 1 + (global_step // training_module.mcfg.increase_roll_every)
        roll_steps = min(roll_steps, training_module.mcfg.roll_max)

        # ld_to_write is a dictionary of loss values averaged across a training sample
        # you can feed it to tensorboard or wandb writer
        optimizer_to_pass = optimizer if global_step >= training_module.mcfg.warmup_steps else None
        scheduler_to_pass = scheduler if global_step >= training_module.mcfg.warmup_steps else None
        ld_to_write = training_module(sample, roll_steps=roll_steps, optimizer=optimizer_to_pass,
                                      scheduler=scheduler_to_pass)
        # training_module.wandb_run.log(ld_to_write, step=global_step)
        if not training_module.debug:
            for k, v in ld_to_write.items():
                training_module.writer.add_scalar(f'train/{k}', v, global_step)
        # save checkpoint every `save_checkpoint_every` steps
        if global_step % cfg.experiment.save_checkpoint_every == 0 and not training_module.debug:
            checkpoint_path = os.path.join(checkpoints_dir, f"step_{global_step:010d}.pth")
            save_checkpoint(training_module, aux_modules, cfg, checkpoint_path)
        if global_step % cfg.experiment.validate_every == 0 and val_sequences is not None and global_step >= min_validate:
            training_module.validation(val_sequences, global_step, result_dir=os.path.join(run_dir, 'result'))

    return global_step
