import os
import torch
from models import PatchTST, AGPT, SimpleTM, iTransformer, TimeMixer, FreTS, WPMixer, DLinear, TimesNet,\
            Duet, PatchMLP, TimeFilter, xPatch, PathFormer, TimeMixerPP, Autoformer, FEDformer, \
            Informer, LightTS, Reformer, ETSformer, Pyraformer, MICN, Crossformer, TiDE, TSMixer, \
            SegRNN, SCINet, TimeMosaic

class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            'PatchTST': PatchTST,
            'AGPT': AGPT,
            'SimpleTM': SimpleTM,
            'iTransformer': iTransformer,
            'TimeMixer': TimeMixer,
            'FreTS': FreTS,
            'WPMixer': WPMixer,
            'DLinear': DLinear,
            'TimesNet': TimesNet,
            'Duet': Duet,
            'PatchMLP': PatchMLP,
            'TimeFilter': TimeFilter,
            'xPatch': xPatch,
            'PathFormer': PathFormer, 
            'TimeMixerPP': TimeMixerPP,
            'Autoformer': Autoformer,
            'FEDformer': FEDformer,
            'Informer': Informer,
            'LightTS': LightTS,
            'Reformer': Reformer,
            'ETSformer': ETSformer,
            'Pyraformer': Pyraformer,
            'MICN': MICN,
            'Crossformer': Crossformer,
            'TiDE': TiDE,
            'TSMixer': TSMixer,
            'SegRNN': SegRNN,
            "SCINet": SCINet,
            "TimeMosaic": TimeMosaic
            }

        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        if self.args.use_gpu and self.args.gpu_type == 'cuda':
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        elif self.args.use_gpu and self.args.gpu_type == 'mps':
            device = torch.device('mps')
            print('Use GPU: mps')
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
