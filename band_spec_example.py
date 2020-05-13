import pysoundtool as pyst


output_wave_name1 = './test_bandspec.wav'
output_wave_name2 = './test_wiener.wav'
output_wave_name3 = './test_wiener_postfilter.wav'
target_wav = '/home/airos/Projects/Data/denoising_sample_data/noisy_speech_varied_IEEE_small/S_01_02__Water Cooler_scale0.3.wav'#'/home/airos/Projects/Data/denoising_sample_data/noisyspeech_uwnu_limit100_test/CHM02_56-08__noise_keyboard_scale0.3.wav' #'/home/airos/Projects/Data/denoising_sample_data/noisy_speech_varied_IEEE_small/S_01_02__Water Cooler_scale0.3.wav'
noise_file = '/home/airos/Projects/Data/CDBook_SpeechEnhancement/Databases/Noise Recordings/Water Cooler.wav'#None# '/home/airos/Projects/Data/denoising_sample_data/noise_samples/noise_keyboard.wav'#'/home/airos/Projects/Data/sound/noize_examples/originals/air_conditioner/352807__bigmanjoe__fan.wav'#



pyst.filtersignal(output_filename = output_wave_name1, 
                 audiofile = target_wav, 
                 noise_file= noise_file,
                 visualize=False,
                 visualize_every_n_frames=200,
                 scale=1,
                 duration_noise_ms=120,
                 filter_type='band_specsub', # 'band_specsub', 'wiener'
                 apply_postfilter=False,
                 phase_radians=True,
                 real_signal=True)

pyst.filtersignal(output_filename = output_wave_name2, 
                 audiofile = target_wav, 
                 noise_file= noise_file,
                 visualize=False,
                 visualize_every_n_frames=200,
                 scale=1,
                 duration_noise_ms=120,
                 filter_type='wiener', # 'band_specsub', 'wiener'
                 apply_postfilter=False,
                 phase_radians=True,
                 real_signal=True)

pyst.filtersignal(output_filename = output_wave_name3, 
                 audiofile = target_wav, 
                 noise_file= noise_file,
                 visualize=False,
                 visualize_every_n_frames=200,
                 scale=1,
                 duration_noise_ms=120,
                 filter_type='wiener', # 'band_specsub', 'wiener'
                 apply_postfilter=True,
                 phase_radians=True,
                 real_signal=True)
