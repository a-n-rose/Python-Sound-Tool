
## Current
- make it easier to use / build different models
- implement autoencoder model
- implement denoising with autoencoder model
- build autoencoder in keras
- build autoencoder in pytorch
- build via docker image

## Functionality

- autoencoder training
- get postfilter to work on spectral subtraction
- set power_scale default to 'power_to_db'?
- functions to use librosa or not to perform tasks (librosa doesn't work on notebooks.ai for example)
- measure level of snr
- measure quality of filtering/speech enhancement
- measure signal similarity
- source separation
- gender switch
- text to speech
- speech to text
- dataset exploration (visualize 10 random samples/ based on size?, etc.)
- simple inclusion of noise reduction into training models
- pysoundtool and pysoundtool.online version? (use librosa vs no librosa)

## Presentation

- blog post on each set of functionalities
- presentation of examples
- get documentation online
- simplify functions
- improve documentation (references, examples, testing, data shapes!!, help options)

## Testing

- expand test cases
- efficiency of code

## Organization

- reorganize based on use... how import statement should work
- make sample_rate, samprate, samplingrate, sr namespace consistent
- make features/feature_type namespace consistent
- use keyword arguments for librosa and scipy?
- simplify


## Organization ideas:

pyst.loadsound(audiofile, sr)
pyst.playsound(audiofile, sr)?
pyst.plotsound(audiofile, sr, feature_type)

pyst.data.train_val_test(input_data, output_data)
pyst.data.analyze(audo_dir)? For example for audio types, lengths?, sizes? etc. Useful for logging?
pyst.feats.plot()
pyst.feats.hear()
pyst.feats.extract()
model = pyst.models.speechrec_simple() # model will be a class instance..
history = pyst.models.train(model, train_path, val_path)
matplotplib.pyplot.plot(history) ?
pyst.models.plot(history)
pyst.models.run(model, test_path)

pyst.filters.wiener()
pyst.filters.bandsubtraction()
pyst.models.soundclassifier()
pyst.models.autoencoder_denoise()
pyst.models.speechrec()
