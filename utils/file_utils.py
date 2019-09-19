import os
import logging
import numpy as np

# If not exists creates the specified folder
def prepare_data_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)


def setup_running_directories(opt):
	if not os.path.exists(opt.setting_dir):
		os.makedirs(opt.setting_dir)

	if not os.path.exists(opt.model_dir):
		os.makedirs(opt.model_dir)

	if not os.path.exists(opt.result_dir):
		os.makedirs(opt.result_dir)


def save_experiment_settings(opt):
	setting_npy_path = os.path.join(opt.setting_dir, opt.exp_prefix + '.npy')
	logging.info('Setting of the experiment is saved to %s', setting_npy_path)
	np.save(setting_npy_path, opt)