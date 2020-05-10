import pysoundtool as pyst


output_wave_name1 = './test_bandspec1.wav'
output_wave_name2 = './test_bandspec2.wav'
target_wav = '/home/airos/Projects/Data/denoising_sample_data/noisyspeech_uwnu_limit100_test/CHM02_56-08__noise_keyboard_scale0.3.wav' #'/home/airos/Projects/Data/denoising_sample_data/noisy_speech_varied_IEEE_small/S_01_02__Water Cooler_scale0.3.wav'
noise_file = '/home/airos/Projects/Data/denoising_sample_data/noise_samples/noise_keyboard.wav'#'/home/airos/Projects/Data/sound/noize_examples/originals/air_conditioner/352807__bigmanjoe__fan.wav'#


# phase radians set to True sounds much better.. I think. Point is, both work.
pyst.apply_band_specsub(output_wave_name = output_wave_name1,
                        target_wav = target_wav,
                        noise_file = noise_file,
                        visualize=True,
                        phase_radians=True,
                        visualize_freq=100)

pyst.filtersignal(output_filename = output_wave_name2,
                        wavfile = target_wav,
                        noise_file = noise_file,
                        visualize=False,
                        visualize_freq=50)


'''
wiener filter does not reconstruct whole spectrum
'''
