import pysoundtool as pyst


output_wave_name1 = './test_bandspec1.wav'
output_wave_name2 = './test_bandspec2.wav'
target_wav = '/home/airos/Projects/Data/denoising_sample_data/noisyspeech_uwnu_limit100_test/CHM02_56-08__noise_keyboard_scale0.3.wav'
noise_file = '/home/airos/Projects/Data/denoising_sample_data/noise_samples/noise_keyboard.wav'

pyst.apply_band_specsub(output_wave_name = output_wave_name1,
                        target_wav = target_wav,
                        noise_file = noise_file,
                        visualize=True)

pyst.filtersignal(output_filename = output_wave_name2,
                        wavfile = target_wav,
                        noise_file = noise_file,
                        visualize=False)
