###############################################################################
from . import utils
from . import feats
from . import files
from . import datasets
from . import filters
from . import dsp
from . import builtin
from . import exceptions as errors
from . import augment
from .utils import check_dir, string2pathlib
from .files import loadsound, savesound
from .feats import plotsound 
from .filters import WienerFilter, BandSubtraction
from .dsp import generate_sound, generate_noise
from .builtin import envclassifier_feats, denoiser_feats, denoiser_train, \
    envclassifier_train, denoiser_run, filtersignal, cnnlstm_train, resnet50_train

from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)


__all__=['utils', 'feats', 'filters', 
         'WienerFilter', 'BandSubtraction','filtersignal','dsp','errors',
         'plotsound', 'loadsound', 'savesound', 'datasets', 'envclassifier_feats',
         'denoiser_feats', 'denoiser_train', 'envclassifier_train',
         'generate_sound', 'generate_noise', 'denoiser_run', 'builtin', 'cnnlstm_train',
         'resnet50_train', 'augment', 'check_dir', 'string2pathlib']
