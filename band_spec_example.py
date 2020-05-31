import pysoundtool as pyst


output_wave_name1 = './test_bandspec.wav'
output_wave_name2 = './test_wiener.wav'
output_wave_name3 = './test_wiener_postfilter.wav'
target_wav = '/home/airos/Projects/github/a-n-rose/Python-Sound-Tool/audiodata/python_traffic.wav'#'/home/airos/Projects/Data/denoising_sample_data/noisy_speech_varied_IEEE_small/S_01_02__Water Cooler_scale0.3.wav'#'/home/airos/Projects/Data/denoising_sample_data/noisyspeech_uwnu_limit100_test/CHM02_56-08__noise_keyboard_scale0.3.wav' #'/home/airos/Projects/Data/denoising_sample_data/noisy_speech_varied_IEEE_small/S_01_02__Water Cooler_scale0.3.wav'
noise_file = None#'/home/airos/Projects/Data/CDBook_SpeechEnhancement/Databases/Noise Recordings/Water Cooler.wav'#None# '/home/airos/Projects/Data/denoising_sample_data/noise_samples/noise_keyboard.wav'#'/home/airos/Projects/Data/sound/noize_examples/originals/air_conditioner/352807__bigmanjoe__fan.wav'#



#duration_noise_ms = 1000

#pyst.filtersignal( 
                 #audiofile = target_wav, 
                 #noise_file= noise_file,
                 #visualize=True,
                 #visualize_every_n_frames=100,
                 #filter_scale=1,
                 #duration_noise_ms=duration_noise_ms,
                 #frame_duration_ms=20,
                 #percent_overlap=0.5,
                 #filter_type='band_specsub', # 'band_specsub', 'wiener'
                 #num_bands = 6,
                 #apply_postfilter=False,
                 #phase_radians=True,
                 #real_signal=False,
                 #save2wav=True,
                 #output_filename = output_wave_name1)

#pyst.filtersignal( 
                 #audiofile = target_wav, 
                 #noise_file= noise_file,
                 #visualize=True,
                 #visualize_every_n_frames=100,
                 #filter_scale=1,
                 #duration_noise_ms=duration_noise_ms,
                 #frame_duration_ms=20,
                 #percent_overlap=0.5,
                 #filter_type='wiener', # 'band_specsub', 'wiener'
                 #apply_postfilter=False,
                 #phase_radians=True,
                 #real_signal=True,
                 #save2wav=True,
                 #output_filename = output_wave_name2)

#pyst.filtersignal( 
                 #audiofile = target_wav, 
                 #noise_file= noise_file,
                 #visualize=True,
                 #visualize_every_n_frames=100,
                 #filter_scale=1,
                 #duration_noise_ms=duration_noise_ms,
                 #frame_duration_ms=20,
                 #percent_overlap=0.5,
                 #filter_type='wiener (post filter)', # 'band_specsub', 'wiener'
                 #apply_postfilter=True,
                 #phase_radians=True,
                 #real_signal=True,
                 #save2wav=True,
                 #output_filename = output_wave_name2)

pyst.filtersignal(audiofile = target_wav, filter_type = 'bandsubptraction', save2wav=True)
                 
pyst.filtersignal(audiofile = target_wav, filter_type = 'wiener',save2wav=True)

pyst.filtersignal(audiofile = target_wav, filter_type='wiener (post filter)',save2wav=True)
