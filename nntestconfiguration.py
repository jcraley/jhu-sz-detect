from utils.testconfiguration import TestConfiguration
import json


class NnTestConfiguration(TestConfiguration):

    def __init__(self, default_cfg='default.ini', argv=None):
        super().__init__(default_cfg, argv)

    def _update(self):
        self.all_params = {}
        self._update_experiment_params()
        self._update_preprocessing_params()
        self._update_model_params()
        self._update_optimization_params()

    def _update_model_params(self):
        # MODEL parameters
        model_cfg = self.config['MODEL']
        self.all_params.update({
            'encoder': model_cfg['encoder'],
            'encoder kwargs': model_cfg['encoder kwargs'],
            'classifier': model_cfg['classifier'],
            'classifier kwargs': model_cfg['classifier kwargs'],
            'load encoder cfg': model_cfg['load encoder cfg'],
        })

    def _update_optimization_params(self):
        # Optimization parameters
        opt_cfg = self.config['OPTIMIZATION']
        self.all_params.update({
            'lr': opt_cfg.getfloat('lr'),
            'batch size': opt_cfg.getint('batch size'),
            'optimizer': opt_cfg['optimizer'],
            'optimizer kwargs': json.loads(opt_cfg['optimizer kwargs']),
            'epochs': opt_cfg.getint('epochs'),
            'weight decay': opt_cfg.getfloat('weight decay'),
            'gamma': opt_cfg.getfloat('gamma'),
            'weight by class': opt_cfg.getboolean('weight by class'),
            'scheduler': opt_cfg['scheduler'],
            'scheduler kwargs': json.loads(opt_cfg['scheduler kwargs']),
            'weighted sampling': opt_cfg.getboolean('weighted sampling')
        })
