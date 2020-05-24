import pysoundtool as pyst
###############################################################################

# TODO add random seed to application of noise and scales 
# TODO add snr levels instead of noise_scales
def main(create_audiodata = False,
         cleandata_path = None,
         noisedata_path = None,
         trainingdata_dir = None,
         limit=None):
    
    if create_audiodata:
        noisydata_dir, cleandata_dir = pyst.featorg.create_autoencoder_data(
            cleandata_path = cleandata_path,
            noisedata_path = noisedata_path,
            trainingdata_dir = trainingdata_dir,
            perc_train = 0.8,
            perc_val = 0.2,
            limit = limit,
            noise_scales = [0.3,0.2,0.1],
            sr = 22050)

if __name__ == '__main__':
    create_audiodata = True
    cleandata_path = \
        '/home/airos/Projects/Data/sound/uwnu-v1-speech/uwnu-v1/audio/'
    noisedata_path = \
        '/home/airos/Projects/Data/denoising_sample_data/noise_samples/'
    trainingdata_dir = \
        '/home/airos/Projects/Data/denoising_sample_data/testing/'
    limit = 50
    
    main(create_audiodata = create_audiodata,
         cleandata_path = cleandata_path,
         noisedata_path = noisedata_path,
         trainingdata_dir = trainingdata_dir,
         limit = limit)
