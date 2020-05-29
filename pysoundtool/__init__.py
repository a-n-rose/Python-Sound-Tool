###############################################################################
from . import paths
from .paths import PathSetup
from .utils import utils
from . import feats
from .feats import FeatPrep_SoundClassifier, prepfeatures, getfeatsettings, \
    visualize_feats
from . import filters
from .filters import WienerFilter, BandSubtraction, filtersignal
from .filters import calc_audioclass_powerspecs as welch2class
from .filters import coll_beg_audioclass_samps as save_class_noise
from . import dsp
from . import exceptions as errors
from .data import loadsound
from . import data

__all__=['paths', 'PathSetup', 'utils', 'feats',\
     'FeatPrep_SoundClassifier', 'prepfeatures','getfeatsettings',\
        'filters','WienerFilter','BandSubtraction','welch2class',\
            'save_class_noise','filtersignal', \
             'dsp','errors', 'visualize_feats', 'loadsound', 'data']
