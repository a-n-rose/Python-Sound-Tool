###############################################################################
from .file_architecture import paths
from .file_architecture.paths import PathSetup
from .acousticfeats_ml import featorg
from .acousticfeats_ml.featorg import audio2datasets
from .acousticfeats_ml.modelfeats import PrepFeatures
from .acousticfeats_ml.modelfeats import prepfeatures as run_featprep 
from .acousticfeats_ml.modelfeats import loadfeature_settings as getfeatsettings
from .filterfun import filters
from .filterfun.filters import WienerFilter
from .filterfun.filters import calc_audioclass_powerspecs as welch2class
from .filterfun.filters import coll_beg_audioclass_samps as save_class_noise
from .filterfun.applyfilter import filtersignal
from .mathfun import dsp, matrixfun, augmentdata
from . import exceptions as errors


__all__=['paths', 'PathSetup', \
    'featorg', 'audio2datasets', 'PrepFeatures', 'run_featprep','getfeatsettings',\
        'filters','WienerFilter','welch2class', 'save_class_noise','filtersignal', \
            'dsp', 'matrixfun', 'augmentdata', \
                'errors']
