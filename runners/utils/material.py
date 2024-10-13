import torch
import numpy as np
from utils.cloth_and_material import Material
from utils.common import random_between, random_between_log, relative_between_log, relative_between, \
    add_field_to_pyg_batch


class RandomMaterial:
    """
    Helper class to sample random material parameters
    """
    def __init__(self, mcfg):
        self.mcfg = mcfg
        
    def get_density(self, device, B):
        if self.mcfg.density_override is None:
            density = random_between(self.mcfg.density_min, self.mcfg.density_max, shape=[B]).to(
                device)
        else:
            density = torch.ones(B).to(device) * self.mcfg.density_override
            
        return density
    
    def get_lame_mu(self, device, B):
        if self.mcfg.lame_mu_override is None:
            lame_mu, lame_mu_input = random_between_log(self.mcfg.lame_mu_min, self.mcfg.lame_mu_max,
                                                        shape=[B], return_norm=True, device=device)
        else:
            lame_mu = torch.ones(B).to(device) * self.mcfg.lame_mu_override
            lame_mu_input = relative_between_log(self.mcfg.lame_mu_min, self.mcfg.lame_mu_max,
                                                 lame_mu)
            
        return lame_mu, lame_mu_input
    
    def get_lame_lambda(self, device, B):
        if self.mcfg.lame_lambda_override is None:
            lame_lambda, lame_lambda_input = random_between(self.mcfg.lame_lambda_min,
                                                            self.mcfg.lame_lambda_max,
                                                            shape=[B], return_norm=True, device=device)
        else:
            lame_lambda = torch.ones(B).to(device) * self.mcfg.lame_lambda_override
            lame_lambda_input = relative_between(self.mcfg.lame_lambda_min, self.mcfg.lame_lambda_max,
                                                 lame_lambda)
            
        return lame_lambda, lame_lambda_input

    def get_stiffness(self, device, B):
        if self.mcfg.stiffness_override is None:
            stiffness, stiffness_input = random_between_log(self.mcfg.stiffness_min,
                                                            self.mcfg.stiffness_max,
                                                            shape=[B], return_norm=True, device=device)
        else:
            stiffness = torch.ones(B).to(device) * self.mcfg.stiffness_override
            stiffness_input = relative_between_log(self.mcfg.stiffness_min, self.mcfg.stiffness_max,
                                                  stiffness)

        return stiffness, stiffness_input

    def get_wind(self, device, B):
        if self.mcfg.wind_override is None:
            wind_scale, _ = random_between(self.mcfg.wind_min, self.mcfg.wind_max, shape=[B],
                                            return_norm=True, device=device)
            phi = np.random.uniform(0, 2 * np.pi)
            cos_theta = np.random.uniform(-1, 1)
            theta = np.arccos(cos_theta)
            x = np.sin(theta) * np.cos(phi)
            y = np.sin(theta) * np.sin(phi)
            z = np.cos(theta)
            wind_direction = np.array([x, y, z], dtype=np.float32)
            wind_direction = torch.from_numpy(wind_direction).to(device)
            wind = wind_scale * wind_direction
        else:
            wind = np.array(self.mcfg.wind_override, dtype=np.float32)
            wind = torch.from_numpy(wind).to(device)
        wind_input = wind
        return wind, wind_input

    def get_k_stretch(self, device, B):
        if self.mcfg.k_stretch_override is None:
            k_stretch, k_stretch_input = random_between_log(self.mcfg.k_stretch_min,
                                                           self.mcfg.k_stretch_max,
                                                           shape=[B], return_norm=True, device=device)
        else:
            k_stretch = torch.ones(B).to(device) * self.mcfg.k_stretch_override
            k_stretch_input = relative_between_log(self.mcfg.k_stretch_min, self.mcfg.k_stretch_max,
                                                  k_stretch)

        return k_stretch, k_stretch_input

    def get_k_shear(self, device, B):
        if self.mcfg.k_shear_override is None:
            k_shear, k_shear_input = random_between_log(self.mcfg.k_shear_min,
                                                         self.mcfg.k_shear_max,
                                                         shape=[B], return_norm=True, device=device)
        else:
            k_shear = torch.ones(B).to(device) * self.mcfg.k_shear_override
            k_shear_input = relative_between_log(self.mcfg.k_shear_min, self.mcfg.k_shear_max,
                                                k_shear)

        return k_shear, k_shear_input

    def get_energy_type(self):
        """
        st30_sp40_ba30 means 30% chance of stvk, 40% chance of spring, 30% chance of baraff
        """
        if self.mcfg.energy_type_prob_override is None:
            energy_type_prob = self.mcfg.energy_type_prob
        else:
            energy_type_prob = self.mcfg.energy_type_prob_override

        if energy_type_prob is None:
            energy_type = 'stvk'
        else:
            energy_type = self.energy_type_from_prob(energy_type_prob)

        return energy_type

    def energy_type_from_prob(self, energy_type_prob):
        if '_' not in energy_type_prob:
            assert energy_type_prob in ['stvk', 'spring', 'baraff'], f'Invalid energy type {energy_type_prob}'
            energy_type = energy_type_prob
        else:
            energy_type_prob = energy_type_prob.split('_')
            energy_type_prob = {k[:2]: float(k[2:]) for k in energy_type_prob}
            energy_type_prob = [energy_type_prob['st'], energy_type_prob['sp'], energy_type_prob['ba']]
            energy_type_prob = np.array(energy_type_prob) / np.sum(energy_type_prob)
            energy_type = str(np.random.choice(['stvk', 'spring', 'baraff'], p=energy_type_prob))
        return energy_type
    
    
    def get_bending_coeff(self, device, B):
        if self.mcfg.bending_coeff_override is None:
            bending_coeff, bending_coeff_input = random_between_log(self.mcfg.bending_coeff_min,
                                                                    self.mcfg.bending_coeff_max,
                                                                    shape=[B], return_norm=True, device=device)
        else:
            bending_coeff = torch.ones(B).to(device) * self.mcfg.bending_coeff_override
            bending_coeff_input = relative_between_log(self.mcfg.bending_coeff_min,
                                                       self.mcfg.bending_coeff_max, bending_coeff)
            
        return bending_coeff, bending_coeff_input
    
    def add_material(self, sample, cloth_obj, material_dict=None):

        B = sample.num_graphs
        device = sample['cloth'].pos.device

        if 'lame_mu' in sample['cloth']:
            # this is for supervised training when the material is already provided
            for k in ['lame_mu', 'lame_mu_input', 'lame_lambda', 'lame_lambda_input', 'bending_coeff',
                      'bending_coeff_input', 'density', 'stiffness', 'k_stretch', 'k_shear', 'wind']:
                assert k in sample['cloth'], f'{k} is not in the sample'
            density = sample['cloth'].density
            lame_mu = sample['cloth'].lame_mu
            lame_lambda = sample['cloth'].lame_lambda
            bending_coeff = sample['cloth'].bending_coeff
            stiffness = sample['cloth'].stiffness
            wind = sample['cloth'].wind
            k_stretch = sample['cloth'].k_stretch
            k_shear = sample['cloth'].k_shear
            # energy_type = sample['cloth'].energy_type
        else:
            if material_dict is None or len(material_dict) == 0 or (len(material_dict) == 1 and 'ext_force' in material_dict):
                density = self.get_density(device, B)
                lame_mu, lame_mu_input = self.get_lame_mu(device, B)
                lame_lambda, lame_lambda_input = self.get_lame_lambda(device, B)
                bending_coeff, bending_coeff_input = self.get_bending_coeff(device, B)
                stiffness, stiffness_input = self.get_stiffness(device, B)
                wind, wind_input = self.get_wind(device, B)
                k_stretch, k_stretch_input = self.get_k_stretch(device, B)
                k_shear, k_shear_input = self.get_k_shear(device, B)
                energy_type = self.get_energy_type()
            else:
                density = material_dict['density']
                lame_mu = material_dict['lame_mu']
                lame_lambda = material_dict['lame_lambda']
                bending_coeff = material_dict['bending_coeff']
                stiffness = material_dict['stiffness']
                wind = material_dict['wind']
                wind = torch.from_numpy(np.array(wind, dtype=np.float32)).to(device)
                k_stretch = material_dict['k_stretch']
                k_shear = material_dict['k_shear']
                # energy_type = self.energy_type_from_prob(material_dict['energy_type_prob'])
                assert '_' not in material_dict['energy_type_prob'], f'Invalid energy type {material_dict["energy_type_prob"]}'
                energy_type = material_dict['energy_type_prob']

                lame_mu_input = relative_between_log(self.mcfg.lame_mu_min, self.mcfg.lame_mu_max, lame_mu)
                lame_lambda_input = relative_between(self.mcfg.lame_lambda_min, self.mcfg.lame_lambda_max, lame_lambda)
                bending_coeff_input = relative_between_log(self.mcfg.bending_coeff_min, self.mcfg.bending_coeff_max,
                                                             bending_coeff)
                stiffness_input = relative_between_log(self.mcfg.stiffness_min, self.mcfg.stiffness_max, stiffness)
                k_stretch_input = relative_between_log(self.mcfg.k_stretch_min, self.mcfg.k_stretch_max, k_stretch)
                k_shear_input = relative_between_log(self.mcfg.k_shear_min, self.mcfg.k_shear_max, k_shear)
                wind_input = wind

            add_field_to_pyg_batch(sample, 'density', density, 'cloth', reference_key=None, one_per_sample=True)
            add_field_to_pyg_batch(sample, 'lame_mu', lame_mu, 'cloth', reference_key=None, one_per_sample=True)
            add_field_to_pyg_batch(sample, 'lame_lambda', lame_lambda, 'cloth', reference_key=None,
                                   one_per_sample=True)
            add_field_to_pyg_batch(sample, 'bending_coeff', bending_coeff, 'cloth', reference_key=None,
                                   one_per_sample=True)
            add_field_to_pyg_batch(sample, 'lame_mu_input', lame_mu_input, 'cloth', reference_key=None, one_per_sample=True)
            add_field_to_pyg_batch(sample, 'lame_lambda_input', lame_lambda_input, 'cloth', reference_key=None,
                                   one_per_sample=True)
            add_field_to_pyg_batch(sample, 'bending_coeff_input', bending_coeff_input, 'cloth', reference_key=None,
                                   one_per_sample=True)
            add_field_to_pyg_batch(sample, 'stiffness', stiffness, 'cloth', reference_key=None, one_per_sample=True)
            add_field_to_pyg_batch(sample, 'stiffness_input', stiffness_input, 'cloth', reference_key=None,
                                   one_per_sample=True)
            add_field_to_pyg_batch(sample, 'k_stretch', k_stretch, 'cloth', reference_key=None, one_per_sample=True)
            add_field_to_pyg_batch(sample, 'k_stretch_input', k_stretch_input, 'cloth', reference_key=None,
                                      one_per_sample=True)
            add_field_to_pyg_batch(sample, 'k_shear', k_shear, 'cloth', reference_key=None, one_per_sample=True)
            add_field_to_pyg_batch(sample, 'k_shear_input', k_shear_input, 'cloth', reference_key=None,
                                      one_per_sample=True)
            add_field_to_pyg_batch(sample, 'wind', wind[None], 'cloth', reference_key=None, one_per_sample=True)
            sample['cloth']._mapping['energy_type'] = [energy_type]

        bending_multiplier = self.mcfg.bending_multiplier
        material = Material(density, lame_mu, lame_lambda,
                            bending_coeff, stiffness, k_stretch, k_shear, wind, bending_multiplier)
        cloth_obj.set_material(material)
        
        return sample, cloth_obj
    