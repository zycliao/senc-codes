import os
import numpy as np
from utils.validation import Config as ValidationConfig
from utils.validation import load_runner_from_checkpoint, update_config_for_validation, create_one_sequence_dataloader
from utils.arguments import load_params
from utils.common import move2device, pickle_dump
from utils.defaults import DEFAULTS, HOOD_DATA
from pathlib import Path
import random
import torch

torch.random.manual_seed(0)
np.random.seed(0)
random.seed(0)

if __name__ == '__main__':
    # Set material paramenters, see configs/cvpr.yaml for the training ranges for each parameter
    config_dict = dict()
    config_dict['density'] = 0.20022

    config_dict['lame_mu'] = 23600.0
    config_dict['lame_lambda'] = 44400
    config_dict['bending_coeff'] = 3.962e-05


    # config_dict['lame_mu'] = 50000
    # config_dict['lame_lambda'] = 66400
    # config_dict['bending_coeff'] = 1e-7

    # config_dict['lame_mu'] = 31818.0273
    # config_dict['lame_lambda'] = 18165.1719
    # config_dict['bending_coeff'] = 9.1493e-06

    config_dict['lame_mu'] = 23600.0
    config_dict['lame_lambda'] = 44400
    config_dict['bending_coeff'] = 3.962e-05
    config_dict['energy_type_prob'] = 'stvk'
    config_dict['wind'] = (-5., -3., 5.)
    mat_name = 'wind2'

    garment_name = 'dress'

    save_name = f'{mat_name}_{garment_name}'

    config_name = 'postcvpr_space'
    # save_name = 'postcvpr_velocity_aug'

    # save_dir = "/root/data/cloth_recon/c3/hood_results"
    exp_dir = os.path.join(DEFAULTS.data_root, 'experiments', 'postcvpr_ext_force3_20240222_223849')
    i_iter = 100000
    save_dir = os.path.join(exp_dir, f'inference_{i_iter}')
    os.makedirs(save_dir, exist_ok=True)
    # checkpoint_path = os.path.join(exp_dir, 'checkpoints', f'step_{i_iter:010d}.pth')
    # assert os.path.exists(checkpoint_path)
    checkpoint_path = None


    # If True, the SMPL poses are slightly modified to avoid hand-body self-penetrations. The technique is adopted from the code of SNUG
    config_dict['separate_arms'] = False
    config_dict['keep_length'] = True
    # Paths to SMPL model and garments_dict file relative to $HOOD_DATA/aux_data
    config_dict['garment_dict_file'] = 'garments_dict.pkl'
    config_dict['smpl_model'] = 'smpl/SMPL_NEUTRAL.pkl'
    config_dict['collision_eps'] = 4e-3
    validation_config = ValidationConfig(**config_dict)



    # checkpoint_path = Path(DEFAULTS.data_root) / 'trained_models' / 'postcvpr.pth'

    # checkpoint_path = Path(DEFAULTS.data_root) / 'experiments' / 'postcvpr_velocity_aug_20231129_174704' / 'checkpoints' / 'step_0000098000.pth'
    # checkpoint_path = Path(DEFAULTS.data_root) / 'experiments' / 'postcvpr_sim_data_20231129_223020' / 'checkpoints' / 'step_0000300000.pth'
    # checkpoint_path = Path(DEFAULTS.data_root) / 'experiments' / 'postcvpr_sim_data_20231214_161648' / 'checkpoints' / 'step_0000110000.pth'
    # checkpoint_path = Path(DEFAULTS.data_root) / 'experiments' / 'postcvpr_explicit2_20231025_215127' / 'checkpoints' / 'step_0000128000.pth'
    # checkpoint_path = Path(DEFAULTS.data_root) / 'experiments' / '20230728_092347' / 'checkpoints' / 'step_0000150000.pth'
    # checkpoint_path = Path(DEFAULTS.data_root) / 'experiments' / 'postcvpr_explicit_20231021_163934' / 'checkpoints' / 'step_0000170000.pth'

    # load the config from .yaml file and load .py modules specified there
    modules, experiment_config = load_params(config_name)

    # modify the config to use it in validation
    experiment_config = update_config_for_validation(experiment_config, validation_config)

    # load Runner object and the .py module it is declared in
    runner_module, runner = load_runner_from_checkpoint(checkpoint_path, modules, experiment_config)

    # file with the pose sequence
    # sequence_path =  Path(HOOD_DATA) / 'temp/01_01.pkl'
    sequence_path = Path(DEFAULTS.vto_root) / 'smpl_parameters' / 'stretch.pkl'


    dataloader = create_one_sequence_dataloader(sequence_path, garment_name, modules, experiment_config)
    sequence = next(iter(dataloader))
    sequence = move2device(sequence, 'cuda:0')


    trajectories_dict = runner.valid_rollout(sequence,  bare=False, n_steps=300)
    # Save the sequence to disc
    out_path = Path(DEFAULTS.data_root) / 'temp' / f'{save_name}.pkl'
    print(f"Rollout saved into {out_path}")
    pickle_dump(dict(trajectories_dict), out_path)

    from utils.mesh_io import save_as_pc2

    save_as_pc2(out_path, save_dir, save_mesh=True, prefix=save_name)

    # save metrics
    # metric_save_path = os.path.join(save_dir, save_name + '_metrics.npz')
    # metric_dict = {k: v for k, v in trajectories_dict['metrics'].items() if k.endswith('_per_vert')}
    # import functools
    # total_per_vert = functools.reduce(lambda a, b: a + b, [np.array(v) for k, v in metric_dict.items()])
    # total_per_vert = reduce([v for k, v in metric_dict.items()])
    # metric_dict['total_per_vert'] = total_per_vert
    # np.savez(metric_save_path, **metric_dict)