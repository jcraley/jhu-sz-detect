import sys

import torch

import nntestconfiguration as tc
import utils.pathmanager as pm
from nnpipeline import NnPipeline

torch.manual_seed(0)


def main():
    # Check the device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Using device {}'.format(device))
    print("Let's use", torch.cuda.device_count(), "GPUs!")

    """Set up the experiment, initialize folders, and write config"""
    # Load the experiment configuration and paths
    argv = sys.argv[1:]
    params = tc.NnTestConfiguration('nn_default.ini', argv)
    paths = pm.PathManager(params)
    paths.initialize_experiment_folders()
    pipeline = NnPipeline(params, paths, device)

    # Save experiment configuration
    pipeline.write_config_file()

    # Load the datsets
    pipeline.initialize_val_dataset()
    pipeline.initialize_train_dataset()

    """Train or load a model"""
    if pipeline.params['load model fn']:
        pipeline.load_model()
    else:
        pipeline.initialize_data_loaders()
        pipeline.initialize_model()
        pipeline.initialize_training()
        pipeline.initialize_loss()
        pipeline.train()

    """Score test and train sets"""
    if params['score train']:
        pipeline.score_train_dataset()

    if params['score val']:
        pipeline.score_val_dataset()


if __name__ == '__main__':
    main()
