Filtering
---------

NOTE: only .wav files of bit depth 16 or 32 can currently be used. See
subsection 'Convert Soundfiles for use with scipy.io.wavfile'

For visuals, we will look at the sound as their FBANK features.

Noisy sound file
~~~~~~~~~~~~~~~~

Add 'python' speech segment and traffic noise to create noisy speech.
Save as .wav file.

::

    >>> speech = './audiodata/python.wav'
    >>> noise = './audiodata/traffic.aiff'
    >>> data_noisy, samplerate = pyst.dsp.add_backgroundsound(speech, noise, delay_mainsound_sec=1, scale_background = 0.3, total_len_sec=5)
    >>> pyst.plotsound(data_noisy, sr=samplerate, feature_type='fbank', power_scale='power_to_db')

.. image:: ./_static/python_traffic_fbank.png

Then filter the traffic out:

.. figure:: https://i.imgur.com/FOcjwAl.png
   :alt: Imgur

   Imgur

This is what the noise power spectrum of the full FFT looks like:

.. figure:: https://i.imgur.com/7CIiTfM.png
   :alt: Imgur

   Imgur

If you set ``real_signal`` to true, this is what the noise power
spectrum looks like:

.. figure:: https://i.imgur.com/6AWr5dV.png
   :alt: Imgur

   Imgur

In numpy, you can use the full fft signal by using numpy.fft.fft or you
can use the real fft, for audio signals for example, by using
numpy.fft.rfft. The latter may be more efficent and there isn't a
difference between the two. I've seen some Implement the full fft and
others the rfft.

Wiener filter
^^^^^^^^^^^^^

::

    >>> pyst.filtersignal(output_filename = 'python_traffic_wiener_filter.wav',
                        audiofile = noisy_speech_filename,
                        filter_type = 'wiener',
                        filter_scale = 1) # how strong the filter should be

What the filtered signal looks like in raw samples, power spectrum
(basically stft), and fbank features:

.. figure:: https://i.imgur.com/42liCr1.png
   :alt: Imgur

   Imgur

.. figure:: https://i.imgur.com/tx87UEL.png
   :alt: Imgur

   Imgur

.. figure:: https://i.imgur.com/TrwKJ4j.png
   :alt: Imgur

   Imgur

Wiener filter with postfilter
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If there is some distortion in the signal, try a post filter:

::

    >>> pyst.filtersignal(output_filename = 'python_traffic_wiener_postfilter.wav',
                        audiofile = noisy_speech_filename,
                        filter_type = 'wiener_postfilter',
                        filter_scale = 1, # how strong the filter should be
                        apply_postfilter = True) 

What the filtered signal looks like in raw samples, power spectrum
(basically stft), and fbank features:

.. figure:: https://i.imgur.com/zTR4kX3.png
   :alt: Imgur

   Imgur

.. figure:: https://i.imgur.com/lKe4dRQ.png
   :alt: Imgur

   Imgur

.. figure:: https://i.imgur.com/AwontYt.png
   :alt: Imgur

   Imgur

Band spectral subtraction filter
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For comparison, try a band spectral subtraction filter:

::

    >>> pyst.filtersignal(output_filename = 'python_traffic_bandspecsub.wav',
                        audiofile = noisy_speech_filename,
                        filter_type = 'band_spectral_subtracion',
                        filter_scale = 1, # how strong the filter should be
                        num_bands = 6) 

What the filtered signal looks like in raw samples, power spectrum
(basically stft), and fbank features:

.. figure:: https://i.imgur.com/Kg9cR2S.png
   :alt: Imgur

   Imgur

.. figure:: https://i.imgur.com/jSX4ijV.png
   :alt: Imgur

   Imgur

.. figure:: https://i.imgur.com/cFdaGLl.png
   :alt: Imgur

   Imgur
