###############################################################################
from . import paths
from .paths import PathSetup
from .tools import tools
from . import feats
from .feats import FeatPrep_SoundClassifier, prepfeatures, getfeatsettings, \
    visualize_feats
from . import filters
from .filters import WienerFilter, BandSubtraction, filtersignal
from .filters import calc_audioclass_powerspecs as welch2class
from .filters import coll_beg_audioclass_samps as save_class_noise
from . import dsp
from . import exceptions as errors

__all__=['paths', 'PathSetup', 'tools', 'feats',\
     'FeatPrep_SoundClassifier', 'prepfeatures','getfeatsettings',\
        'filters','WienerFilter','BandSubtraction','welch2class',\
            'save_class_noise','filtersignal', \
             'dsp','errors', 'visualize_feats']
