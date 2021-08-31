import json
import os
import time

import numpy as np
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import torch.nn.utils.rnn as rnn

import nntestconfiguration as tc
import utils.evaluation as evaluation
import utils.output_tools as out
import utils.pathmanager as pm
import utils.visualization as viz
from nn.classifiers import *
from nn.cnn_architectures import *
from nn.encoderclassifier import EncoderClassifier
from nn.encoders import *
from nn.resencoder import ResEncoder
from nn.quadfilter import FilterNet
from nn.focalloss import FocalLoss
from nn.nn_tools import train_model
from utils.pipeline import Pipeline
from utils.dataset import EpilepsyDataset

import matplotlib.pyplot as plt


def plot_three_state(preds, label, fn=None, threshold=None):
    fig = plt.figure()
    ax = fig.gca()
    ax.plot(preds[:, 1], label='SZ')
    ax.plot(preds[:, 0], label='Pre-SZ')
    ax.plot(preds[:, 2], label='Post-SZ')
    ax.plot(label, label='Ground Truth')
    ax.legend(ncol=4, loc='upper center', bbox_to_anchor=(0.5, 1.1))

    if threshold:
        plt.axhline(threshold, 0, 1, linestyle='--',
                    color='k', label='Threshold')
        classifications = np.asarray(preds[:, 1] >= threshold, dtype=int)
        plt.fill_between(np.arange(len(classifications)),
                         preds[:, 1] * classifications, 0,
                         color='blue',       # The outline color
                         alpha=0.2)

    ax.set_xlabel('Time (s)', fontsize=16)
    ax.set_ylabel('Model Output', fontsize=16)
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)
    ax.set_ylim([-0.05, 1.05])
    ax.set_xlim([0, len(preds[:, 1])])
    if fn:
        plt.tight_layout()
        plt.savefig(fn)
        plt.close()

    else:
        plt.show()


def collate_sequences(batch):
    data = [item['buffers'] for item in batch]
    target = [item['labels'] for item in batch]
    return {'buffers': data, 'labels': target}


def collate_sequences_as_tensor(batch):
    """Collate a batch of samples as a tensor where each has the same length

    Shorter sequences are zero padded to the length of the longest sample

    Args:
        batch (list): List of samples

    Returns:
        Dictionary: Dict with fields {'buffers' (bsize, T_max, C, L),
                                      'labels' (bsize, T_max)}
    """
    nitems = len(batch)
    data = rnn.pad_sequence([item['buffers'] for item in batch],
                            batch_first=True)
    labels = rnn.pad_sequence([item['labels'] for item in batch],
                              batch_first=True)

    pt_numbers = []
    onset_zones = torch.zeros((nitems), dtype=torch.long)
    lateralizations = torch.zeros((nitems), dtype=torch.long)
    lobes = torch.zeros((nitems), dtype=torch.long)
    for ii, item in enumerate(batch):
        pt_numbers.append(item['patient number'])
        onset_zones[ii] = item['onset zone']
        lateralizations[ii] = item['lateralization']
        lobes[ii] = item['lobe']
    return {'buffers': data, 'labels': labels,
            'patient numbers': pt_numbers,
            'onset zones': onset_zones,
            'lateralizations': lateralizations,
            'lobes': lobes}


class NnPipeline(Pipeline):

    def __init__(self, params, paths, device='cpu'):
        super().__init__(params, paths, device=device)
        paths['sequence results'] = os.path.join(paths['trial'],
                                                 'sequence_results')

    def initialize_train_dataset(self, post_sz=False):
        if self.params['load to device']:
            device = self.device
        else:
            device = 'cpu'
        print('Loading to device: {}'.format(device))
        self.train_dataset = EpilepsyDataset(
            self.params['train manifest'],
            self.paths['buffers'],
            self.params['window length'],
            self.params['overlap'],
            device=device,
            features_dir=self.paths['features'],
            features=self.params['features'],
            post_sz=post_sz
        )
        if self.params['load as'] != 'iid':
            self.train_dataset.set_as_sequences(True)

    def initialize_val_dataset(self, post_sz=False):
        if self.params['load to device']:
            device = self.device
        else:
            device = 'cpu'
        print('Loading to device: {}'.format(device))
        self.val_dataset = EpilepsyDataset(
            self.params['val manifest'],
            self.paths['buffers'],
            self.params['window length'],
            self.params['overlap'],
            device=device,
            features_dir=self.paths['features'],
            features=self.params['features'],
            post_sz=post_sz
        )
        if self.params['load as'] != 'iid':
            self.val_dataset.set_as_sequences(True)

    def initialize_data_loaders(self):
        """Create the dataloaders"""
        loader_kwargs = {
            'batch_size': self.params['batch size'],
            'num_workers': 0,
            'shuffle': True,
        }
        if self.params['load as'] == 'tensor_sequences':
            loader_kwargs['collate_fn'] = collate_sequences_as_tensor
        elif self.params['load as'] == 'sequences':
            loader_kwargs['collate_fn'] = collate_sequences
        # Val dataloader should never reweight
        val_dataloader = DataLoader(self.val_dataset, **loader_kwargs)
        # Set up weighted random sampling
        if (self.params['load as'] == 'iid'
                and self.params['weighted sampling']):
            print('Using weighted sampling')
            train_labels = self.train_dataset.get_all_labels()
            counts = torch.bincount(train_labels).double()
            weight = 1. / counts
            sample_weights = torch.tensor([weight[t] for t in train_labels])
            sampler = torch.utils.data.WeightedRandomSampler(sample_weights,
                                                             len(sample_weights),
                                                             replacement=True)
            train_dataloader = DataLoader(self.train_dataset,
                                          batch_size=self.params['batch size'],
                                          sampler=sampler)
        else:
            train_dataloader = DataLoader(self.train_dataset, **loader_kwargs)
        self.dataloaders = {
            'train': train_dataloader,
            'val': val_dataloader
        }

    def initialize_loss(self):
        """Set up the loss function, optimizer, and lr scheduler"""
        class_counts = self.train_dataset.class_counts()
        class_counts[class_counts == 0] = 1
        alpha = None
        if self.params['weight by class']:
            weights = torch.sum(class_counts).float() / class_counts.float()
            alpha = weights
            alpha = alpha.to(self.device)
        self.criterion = FocalLoss(gamma=self.params['gamma'], alpha=alpha)
        self.criterion = self.criterion.to(self.device)

    def initialize_model(self):
        """Initialize the experiment's model"""

        # Initialize the encoder
        nchns, T = self.val_dataset.d_out
        encoder_kwargs = json.loads(self.params['encoder kwargs'])
        encoder_kwargs.update({'nchns': nchns, 'T': T})
        encoder_str = self.params['encoder'] + '(**encoder_kwargs)'
        encoder = eval(encoder_str)

        # If specified load a pretrained encoder
        if self.params['load encoder cfg'] != '':
            encoder_exp_params = tc.CnnTestConfiguration(
                self.params['load encoder cfg'])
            encoder_exp_paths = pm.PathManager(encoder_exp_params)
            classifier_kwargs = json.loads(
                encoder_exp_params['classifier kwargs'])
            classifier_str = (encoder_exp_params['classifier']
                              + '(encoder.d_out, **classifier_kwargs)')
            classifier = eval(classifier_str)
            load_model = EncoderClassifier(encoder, classifier)
            model_fn = os.path.join(encoder_exp_paths['models'],
                                    'model.pt')
            load_model.load_state_dict(torch.load(model_fn))
            encoder = load_model.encoder

        # Initialize the classifier
        classifier_kwargs = json.loads(self.params['classifier kwargs'])
        classifier_str = (self.params['classifier']
                          + '(encoder.d_out, **classifier_kwargs)')
        classifier = eval(classifier_str)

        # Combine encoder and classifier
        self.model = EncoderClassifier(encoder, classifier)
        self.model.float()
        self.model = self.model.to(self.device)

    def load_model(self):
        model = torch.load(self.params['load model fn'])
        if isinstance(model, dict):
            self.initialize_model()
            self.model.load_state_dict(model)
        else:
            self.model = model
        self.model = self.model.to(self.device)

    def initialize_training(self):
        """ Set up optimizer and schedulers
        """
        self.optimizer = eval(
            """optim.{}(self.model.parameters(), lr=self.params['lr'],
                        **self.params['optimizer kwargs'])""".format(
                self.params['optimizer'])
        )

        self.scheduler = eval(
            """lr_scheduler.{}(self.optimizer,
                               **self.params['scheduler kwargs'])""".format(
                self.params['scheduler'])
        )

    def train(self, save_model=True, save_history=True,
              visualize_history=True):
        """Train the model"""
        since = time.time()
        print('Training on {}'.format(self.device))
        self.model, self.history = train_model(
            self.model,
            self.dataloaders,
            self.criterion, self.optimizer,
            self.scheduler, self.device,
            num_epochs=self.params['epochs'],
            save_folder=None
        )
        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60), flush=True)

        if save_model:
            self.model = self.model.to('cpu')
            state_dict = self.model.state_dict()
            torch.save(state_dict,
                       os.path.join(self.paths['models'], 'model.pt'))
            torch.save(self.model,
                       os.path.join(self.paths['models'], 'full_model.pt'))
            self.model = self.model.to(self.device)

        if save_history:
            fn = os.path.join(self.paths['trial'], 'history.pkl')
            out.save_obj(self.history, fn)
        if visualize_history:
            fn = os.path.join(self.paths['figures'], 'history.png')
            viz.visualize_history(self.history, fn)
