import pysoundtool as pyst
import pysoundtool.explore_sound as exsound 
import pysoundtool.soundprep as soundprep 

import soundfile as sf

#aiff_soundfile = './audiodata/traffic.aiff'
#monowav_soundfile = './audiodata/python.wav'

## make .wav file
#wavfilename = soundprep.convert2wav(aiff_soundfile, samplerate=16000)

## ensure sound data is mono
#data, samplerate = soundprep.loadsound(wavfilename, mono=True)
#print(data.shape)
#print(samplerate)
##(240000,)
##16000

## check if 16 or 32 bit depth (compatible with scipy.io.wavfile)
#wav=sf.SoundFile(wavfilename)
#print(wav.subtype)

## if not:
#wavfile_newbit = soundprep.newbitdepth(
    #wave=wavfilename, bitdepth=16, newname='./audiodata/wav_newdepth', overwrite=False)

###speech = './audiodata/python.wav'
###noise = './audiodata/traffic.aiff'
###data_noisy, samplerate = soundprep.add_sound_to_signal(speech, noise, delay_target_sec=1, scale = 0.1)
###from scipy.io.wavfile import write
###write('./audiodata/python_traffic.wav', samplerate, data_noisy)


###sound2filter = './audiodata/python_traffic.wav'
###pyst.filtersignal(output_filename = './audiodata/python_traffic_filtered.wav',
                  ###wavfile = sound2filter)


####sound2filter = './audiodata/python_traffic.wav'
####pyst.filtersignal(output_filename = './audiodata/python_traffic_filtered_pf.wav',
                  ####wavfile = sound2filter,
                  ####apply_postfilter = True)


###from pysoundtool.templates import soundclassifer

###project_name = 'test_backgroundnoise_classifier'
###headpath = 'saved_features_and_models'
###audio_classes_dir = './audiodata/minidatasets/background_noise/'

###soundclassifer(project_name,
                ###headpath,
                ###audiodir = audio_classes_dir,
                ###feature_type = 'mfcc',
                ###target_wavfile = './audiodata/rain.wav')

#from pysoundtool.templates import noisefilter

#noize.filtersignal( output_file, 
                    #target_file)

#noisefilter( output_file, 
                    #target_file)

#project_name = 'test_smartfilter'
#headpath = 'directory_where_createdfiles_should_be_saved'
#audio_classes_dir = 'directory_where_training_data_is_located'

#filteredwavfile = noisefilter(project_name,
                                #headpath,
                                #audio_classes_dir,
                                #sounddata = 'noisysignal.wav')



#from pysoundtool.templates import soundclassifier

