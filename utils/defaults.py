import os
import socket

from munch import munchify

hostname = socket.gethostname()

# Paths can be overridden with environment variables of the same name
HOOD_PROJECT = os.environ.get("HOOD_PROJECT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HOOD_DATA = os.environ.get("HOOD_DATA", os.path.join(HOOD_PROJECT, "hood_data"))
# simulated ground truth used for validation during training
NC_DIR = os.environ.get("NC_DIR", os.path.join(HOOD_DATA, "neural_cloth"))

DEFAULTS = dict()

DEFAULTS['server'] = 'local'
DEFAULTS['data_root'] = HOOD_DATA
DEFAULTS['experiment_root'] = os.path.join(HOOD_DATA, 'experiments')
DEFAULTS['vto_root'] = os.path.join(HOOD_DATA, 'vto_dataset')
DEFAULTS['aux_data'] = os.path.join(HOOD_DATA, 'aux_data')
DEFAULTS['project_dir'] = HOOD_PROJECT

# TODO: change ts_scale to 1
DEFAULTS['hostname'] = hostname
DEFAULTS = munchify(DEFAULTS)
