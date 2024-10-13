import os
import shutil
import sys
from copy import deepcopy
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from utils.arguments import load_params, create_modules
from utils.mesh_io import read_pc2
from utils.validation import Config as ValidationConfig
from utils.validation import update_config_for_validation, create_one_sequence_dataloader
from utils.common import move2device, add_field_to_pyg_batch
from utils.defaults import *


def create_val_dataset(config, modules):
    config = deepcopy(config)
    dataset_config = config.dataloader.dataset
    dataset_config = dataset_config[list(dataset_config.keys())[0]]
    val_split_path = os.path.join(DEFAULTS['aux_data'], dataset_config['val_split_path'])
    gt_dir = os.path.join(NC_DIR, 'simulation_hood_full',)
    datasplit = pd.read_csv(val_split_path, dtype='str')
    garment_names = datasplit['garment']
    motion_names = datasplit['id']

    materials = np.load(os.path.join(NC_DIR, dataset_config['val_sim_dir'], 'materials.npz'))

    all_sequences = []
    print("Loading validation sequences...")
    for garment_name, motion_name in zip(tqdm(garment_names), motion_names):
        sequence_path = os.path.join(HOOD_DATA, f"vto_dataset/smpl_parameters/{motion_name}")

        config_dict = dict()
        config_dict['separate_arms'] = False
        config_dict['keep_length'] = True

        gt_path = os.path.join(gt_dir, garment_name, motion_name + '.pc2')
        if os.path.exists(gt_path):
            mat_idx = np.where(np.logical_and(materials['garment_names']==garment_name,
                                              materials['motion_names']==motion_name+'.pkl'))[0]
            assert len(mat_idx) == 1
            config_dict['energy_type_prob'] = 'stvk'
            config_dict['density'] = float(materials['density'][mat_idx])
            config_dict['lame_mu'] = float(materials['lame_mu'][mat_idx])
            config_dict['lame_lambda'] = float(materials['lame_lambda'][mat_idx])
            config_dict['bending_coeff'] = float(materials['bending_coeff'][mat_idx])
            config_dict['stiffness'] = 1
            config_dict['k_stretch'] = 1
            config_dict['k_shear'] = 1
            config_dict['wind'] = (0., 0., 0.)


            material_dict = {}
            material_dict['energy_type_prob'] = 'stvk'
            material_dict['density'] = float(materials['density'][mat_idx])
            material_dict['lame_mu'] = float(materials['lame_mu'][mat_idx])
            material_dict['lame_lambda'] = float(materials['lame_lambda'][mat_idx])
            material_dict['bending_coeff'] = float(materials['bending_coeff'][mat_idx])
            material_dict['stiffness'] = 1
            material_dict['k_stretch'] = 1
            material_dict['k_shear'] = 1
            material_dict['wind'] = (0., 0., 0.)

            for k, v in material_dict.items():
                if type(v) is float or type(v) is int:
                    material_dict[k] = torch.tensor([v]).float().to('cuda:0')
        else:
            material_dict = {}

        # config_dict['energy_type_prob'] = 'stvk'
        # config_dict['density'] = 0.20022
        # config_dict['lame_mu'] = 23600.0
        # config_dict['lame_lambda'] = 44400
        # config_dict['bending_coeff'] = 3.962e-05
        # config_dict['stiffness'] = 1
        # config_dict['k_stretch'] = 1
        # config_dict['k_shear'] = 1
        # config_dict['wind'] = (0., 0., 0.)
        #
        # material_dict['energy_type_prob'] = 'stvk'
        # material_dict['density'] = 0.20022
        # material_dict['lame_mu'] = 23600.0
        # material_dict['lame_lambda'] = 44400
        # material_dict['bending_coeff'] = 3.962e-05
        # material_dict['stiffness'] = 1
        # material_dict['k_stretch'] = 1
        # material_dict['k_shear'] = 1
        # material_dict['wind'] = (0., 0., 0.)

        for k, v in material_dict.items():
            if type(v) is float or type(v) is int:
                material_dict[k] = torch.tensor([v]).float().to('cuda:0')

        validation_config = ValidationConfig(**config_dict)
        seq_config = update_config_for_validation(config, validation_config)
        dataloader = create_one_sequence_dataloader(sequence_path, garment_name, modules, seq_config)
        sequence = next(iter(dataloader))
        sequence = move2device(sequence, 'cuda:0')

        if os.path.exists(gt_path):

            gt_cloth_seq = read_pc2(gt_path)
            gt_cloth_seq = np.transpose(gt_cloth_seq, [1, 0, 2])
            gt_cloth_seq = torch.from_numpy(gt_cloth_seq).float().to('cuda:0')
            sequence = add_field_to_pyg_batch(sequence, 'gt', gt_cloth_seq, 'cloth', 'pos')

        setattr(sequence['cloth'], 'garment_name', garment_name)
        setattr(sequence['cloth'], 'motion_name', motion_name)
        setattr(sequence['cloth'], 'material_dict', material_dict)

        all_sequences.append(sequence)
        if is_debug():
            break
        break
    return all_sequences

def load_partial_state_dict(model, checkpoint_path):
    # 加载checkpoint的state_dict
    checkpoint = torch.load(checkpoint_path)
    checkpoint_state_dict = checkpoint['training_module']

    # 获取模型的state_dict
    model_state_dict = model.state_dict()

    # 遍历模型的state_dict
    for name, param in model_state_dict.items():
        # 如果checkpoint中有对应的权重
        if name in checkpoint_state_dict:
            # 获取checkpoint中的权重
            checkpoint_param = checkpoint_state_dict[name]
            # 如果shape小于模型中的权重shape
            if checkpoint_param.shape < param.shape:
                print(f"Updating {name} from {checkpoint_param.shape} to {param.shape}")
                # 根据checkpoint的权重shape创建一个零tensor
                new_param = torch.zeros_like(param)
                # 根据尺寸，拷贝数据到新tensor
                assert new_param.shape[0] == checkpoint_param.shape[0]
                new_param[:, :checkpoint_param.shape[1]] = checkpoint_param
                # 更新模型的state_dict
                model_state_dict[name] = new_param
            else:
                # 如果shape相等，直接拷贝
                model_state_dict[name] = checkpoint_param
        else:
            print(f"Skipping {name} as it's not in the checkpoint or incompatible.")

    # 加载更新后的state_dict到模型
    model.load_state_dict(model_state_dict)

def is_debug():
    gettrace = getattr(sys, 'gettrace', None)
    return gettrace is not None and gettrace()

def backup_file(src, dst):
    os.makedirs(dst, exist_ok=True)
    all_files = os.listdir(src)
    SUFFIX = ['.py', '.sh', '.yaml', '.yml', '.txt', '.md', '.ipynb']
    def has_suffix(fname):
        for suffix in SUFFIX:
            if fname.endswith(suffix):
                return True
        return False
    for fname in all_files:
        fname_full = os.path.join(src, fname)
        fname_dst = os.path.join(dst, fname)
        if os.path.isdir(fname_full):
            backup_file(fname_full, fname_dst)
        elif has_suffix(fname):
            shutil.copy(fname_full, fname_dst)

def main():
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    torch.set_num_threads(1)
    modules, config = load_params()
    dataloader_m, runner, training_module, aux_modules = create_modules(modules, config)

    if config.experiment.validate_every > 0:
        val_sequences = create_val_dataset(config, modules)
    else:
        val_sequences = None
    if is_debug():
        config.dataloader.nums_workers = 0

    now = datetime.now()
    dt_string = now.strftime("%Y%m%d_%H%M%S")
    if hasattr(config, 'resume'):
        print('RESUME:', config.resume)
        run_dir = os.path.join(DEFAULTS.experiment_root, config.resume)
        checkpoint_dir = os.path.join(run_dir, 'checkpoints')
        ckpts = sorted(os.listdir(checkpoint_dir))
        if len(ckpts) > 0:
            config.experiment.checkpoint_path = os.path.join('experiments', config.resume, 'checkpoints', ckpts[-1])
            config.step_start = int(ckpts[-1].split('_')[1].split('.')[0]) + 1
    else:
        run_dir = os.path.join(DEFAULTS.experiment_root, config.config + '_' + dt_string)

    if hasattr(config, 'only_validation'):
        print('VALIDATION ONLY!')
        only_validation = True
    else:
        only_validation = False

    if not is_debug():
        os.makedirs(run_dir, exist_ok=True)
        training_module.setup_writer(run_dir)
        cur_dir = os.path.dirname(os.path.realpath(__file__))
        dst_dir = os.path.join(run_dir, 'code')
        ii = 0
        while os.path.exists(dst_dir):
            ii += 1
            dst_dir = os.path.join(run_dir, f'code_{ii}')
        backup_file(cur_dir, dst_dir)

    if config.experiment.checkpoint_path is not None:
        config.experiment.checkpoint_path = os.path.join(HOOD_DATA, config.experiment.checkpoint_path)
        assert os.path.exists(config.experiment.checkpoint_path), f'Checkpoint {config.experiment.checkpoint_path} does not exist!'
        print('LOADING:', config.experiment.checkpoint_path)
        sd = torch.load(config.experiment.checkpoint_path)

        if 'training_module' in sd:
            try:
                training_module.load_state_dict(sd['training_module'], strict=False)
            except RuntimeError as e:
                print("Loading partial state dict")
                load_partial_state_dict(training_module, config.experiment.checkpoint_path)

            if hasattr(config, 'resume'):  # we only load aux modules (optimizer and scheduler) if we are resuming
                for k, v in aux_modules.items():
                    if k in sd:
                        print(f'{k} LOADED!')
                        v.load_state_dict(sd[k])
                    else:
                        print(f'{k} NOT LOADED!')
        else:
            training_module.load_state_dict(sd)
        print('LOADED:', config.experiment.checkpoint_path)

    if config.detect_anomaly:
        torch.autograd.set_detect_anomaly(True)


    global_step = config.step_start

    torch.manual_seed(57)
    np.random.seed(57)
    for i in range(config.step_start, config.experiment.n_epochs):
        dataloader = dataloader_m.create_dataloader()
        global_step = runner.run_epoch(training_module, aux_modules, dataloader, i, run_dir, config,
                                       global_step=global_step, val_sequences=val_sequences, only_validation=only_validation)

        if config.experiment.max_iter is not None and global_step > config.experiment.max_iter:
            break


if __name__ == '__main__':
    main()
