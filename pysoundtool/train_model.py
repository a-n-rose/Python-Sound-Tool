import os, sys
import inspect
currentdir = os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())))
packagedir = os.path.dirname(currentdir)
sys.path.insert(0, packagedir)

import pysoundtool as pyst
import pysoundtool.models as pystmodels


# TODO make function names more clearly defined for their tasks
# label data
# handle speech labels..



# data prep
data_scene_dir = './audiodata/minidatasets/background_noise/'
data_speech_recognition_dir = './audiodata/minidatasets/speech_commands/'
data_denoise_dir_noisy = './audiodata/minidatasets/denoise/noisyspeech_IEEE_small/'

# datasets

# to npy files 

# classifier:

# 1) collect labels 
labels = []
data_scene_dir = pyst.utils.str2path('./audiodata/minidatasets/background_noise/')
for label in data_scene_dir.glob('*/'):
    labels.append(label.stem)
labels = set(labels)

# 2) create encoding labels dicts
dict_encode, dict_decode = pyst.data.create_dicts_labelsencoded(labels)

# 3) save dicts
classify_data_path = pyst.utils.check_dir('./audiodata/test_classifier/')
try:
    dict_encode_path = pyst.data.save_dict(dict_encode, 
                                       filename= str(classify_data_path)+'dict_encode.csv')
except FileExistsError:
    pass
try:
    dict_decode_path = pyst.data.save_dict(dict_encode, 
                                       filename= str(classify_data_path)+'dict_decode.csv')
except FileExistsError:
    pass

# 4) save audio paths to each label in dict 
paths_list = []
# allow all data types to be collected (not only .wav)
for item in data_scene_dir.glob('**/*'):
    paths_list.append(item)
# pathlib.glob collects hidden files as well - remove them if they are there:
paths_list = [x for x in paths_list if x.stem[0] != '.']
dict_encodedlabel2audio = pyst.data.create_encodedlabel2audio_dict(dict_encode,
                                                     paths_list)
try:
    dict_encdodedlabel2audio_path = pyst.data.save_dict(dict_encodedlabel2audio, 
                                            filename = str(classify_data_path)+'dict_label2audio.csv', replace=True)
except FileExistsError:
    pass

# organize datasets:
training_datasets = pyst.data.audio2datasets(dict_encodedlabel2audio,
                                             perc_train=0.8,
                                             limit=None,
                                             seed=40)

train, val, test = pyst.data.audio2datasets(filename,
                                             perc_train=0.8,
                                             limit=None,
                                             seed=40)

# 5) extract features




# label dict
speech_recognition_data_path = './audiodata/test_speechrec/'



denoise_data_path = './audiodata/test_denoise/'


# train model
cnn_classifier = pystmodels.cnn_classifier()

autoencoder_denoise = pystmodels.autoencoder_denoise(input_shape = (10,11,40))

# implement model w new data



# mix and match models
