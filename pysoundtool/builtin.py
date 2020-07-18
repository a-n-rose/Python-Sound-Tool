'''The builtin module includes more complex functions that pull from several
other functions to complete fairly complex tasks, such as dataset formatting, extracting features for neural networks, training and implementing models on new data.

Basically, if you want PySoundTool to automate some complex tasks for you, here is where to look.
''' 
import time
import pathlib
import random
import numpy as np
import soundfile as sf
from scipy.io.wavfile import write
import keras
from keras.models import load_model
from keras.optimizers import Adam

# in order to import pysoundtool
import os, sys
import inspect
currentdir = os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())))
packagedir = os.path.dirname(currentdir)
sys.path.insert(0, packagedir)
import pysoundtool as pyst
import pysoundtool.models as pystmodels


def filtersignal(audiofile, 
                 sr = None,
                 noise_file=None,
                 filter_type='wiener', # 'band_specsub'
                 filter_scale=1,
                 apply_postfilter=False,
                 duration_noise_ms=120,
                 real_signal=False,
                 phase_radians=True,
                 num_bands=None,
                 visualize=False,
                 visualize_every_n_windows=50,
                 max_vol = 0.4,
                 min_vol = 0.15,
                 save2wav=False,
                 output_filename=None,
                 overwrite=False,
                 use_scipy=False,
                 **kwargs):
    """Apply Wiener or band spectral subtraction filter to signal using noise. 
    
    The noise can be provided as a separate file / samples, or it can be taken
    from the beginning of the provided audio. How much noise is measured can be
    set in the parameter `duration_noise_ms`.

    Parameters 
    ----------
    audiofile : str, np.ndarray [size=(num_samples,) or (num_samples, num_channels)] 
        Filename or the audio data of the signal to be filtered.
    sr : int
        The sample rate of the audio. If `audiofile` is type np.ndarray, sr is 
        required. (default None)
    noise_file : str, tuple, optional
        Path to either noise audiofile or .npy file containing average power 
        spectrum values. If tuple, must include samples and sr.
        If None, the beginning of the `audiofile` will be used for noise data. 
        (default None)
    filter_type : str
        Type of filter to apply. Options 'wiener' or 'band_specsub'.
    filter_scale : int or float
        The scale at which the filter should be applied. This value will be multiplied 
        to the noise levels thereby increasing or decreasing the filter strength. 
        (default 1) 
    apply_postfilter : bool
        Whether or not the post filter should be applied. The post filter 
        reduces musical noise (i.e. distortion) in the signal as a byproduct
        of filtering.
    duration_noise_ms : int or float
        The amount of time in milliseconds to use from noise to apply the 
        Welch's method to. In other words, how much of the noise to use 
        when approximating the average noise power spectrum.
    real_signal : bool 
        If True, only half of the (mirrored) fast Fourier transform will be used 
        during filtering. For audio, there is no difference. This is visible in the 
        plots, however, if you are interested. (default False)
    phase_radians : bool 
        Relevant for band spectral subtraction: whether phase should be calculated in 
        radians or complex values/ power spectrum. (default True)
    num_bands : int
        Relevant for band spectral subtraction: the number of bands to section frequencies
        into. By grouping sections of frequencies during spectral subtraction filtering, 
        musical noise or distortion should be reduced. (defaults to 6)
    visualize : bool 
        If True, plots of the windows and filtered signal will be made. (default False)
    visualize_every_n_windows : int 
        If `visualize` is set to True, this controls how often plots are made: every 50
        windows, for example. (default 50)
    max_vol : int or float 
        The maximum volume level of the filtered signal. This is useful if you know you
        do not want the signal to be louder than a certain value. Ears are important
        (default 0.4) TODO improve on matching volume to original signal? At least use 
        objective measures.
    min_vol : int or float 
        The minimum volume level of the filtered signal. (default 0.15) 
        TODO improve on matching volume to original signal.
    save2wav : bool 
        If True, will save the filtered signal as a .wav file
    output_filename : str, pathlib.PosixPath, optional
        path and name the filtered signal is to be saved. (default None) If no filename
        provided, will save under date.
    overwrite : bool 
        If True and an audiofile by the same name exists, that file will be overwritten.
    use_scipy : bool 
        If False, audiofiles will be loaded using librosa. Otherwise, scipy.io.wavfile.
        (default False)
    **kwargs : additional keyword arguments
        Keyword arguments for `pysoundtool.filters.WienerFilter` or 
        'pysoundtool.filters.BandSubtraction` (depending on `filter_type`).
    
    Returns
    -------
    enhanced_signal : np.ndarray [size = (num_samples, )]
        The enhanced signal in raw sample form. Stereo audio has not yet been tested.
    sr : int 
        The sample rate of the enhanced/ filtered signal.
        
    References
    ----------
    Kamath, S. and Loizou, P. (2002). A multi-band spectral subtraction 
    method for enhancing speech corrupted by colored noise. Proc. IEEE Int.
    Conf. Acoust.,Speech, Signal Processing
    
    Kamath, S. and Loizou, P. (2006). mband.m MATLAB code from the book:
    
    C Loizou, P. (2013). Speech Enhancement: Theory and Practice. 
    """
    if 'wiener' in filter_type:
        if sr == 22050:
            import warnings
            warnings.warn('\n\nWARNING: sample rate 22050 may have some '+\
                'missing frames within the filtered signal. \nIf possible, '+\
                    'perhaps use 8000, 16000, 41000, or 48000 sample rate instead.\n')
        fil = pyst.WienerFilter(sr = sr, **kwargs)
    elif 'band' in filter_type:
        # at this point, band spectral subtraction only works with 
        if sr != 48000:
            import warnings
            warnings.warn('\n\nWARNING: Band spectral subtraciton requires a sample rate'+\
                ' of 48 kHz. Sample rate adjusted from {} to 48000.\n'.format(sr))
            sr = 48000
        fil = pyst.BandSubtraction(sr=48000,
                                   **kwargs)
    if visualize:
        frame_subtitle = 'frame size {}ms, window shift {}ms'.format(fil.frame_dur, int(fil.percent_overlap*fil.frame_dur))

    # load signal (to be filtered)
    if not isinstance(audiofile, np.ndarray):
        samples_orig, sr = pyst.loadsound(audiofile, fil.sr, dur_sec=None, use_scipy=use_scipy)
    else:
        samples_orig, sr = audiofile, sr
    if sr != fil.sr:
        samples_orig, sr = pyst.dsp.resample_audio(samples_orig, 
                                                   sr_original = sr, 
                                                   sr_desired = fil.sr)
    assert fil.sr == sr
    # TODO improve on volume control, improve SNR 
    # set volume max and min, or based on original sample data
    fil.set_volume(samples_orig, max_vol = max_vol, min_vol = min_vol)
    # set how many subframes are needed to process entire target signal
    fil.set_num_subframes(len(samples_orig), is_noise=False, zeropad=fil.zeropad)
    # prepare noise
    # set up how noise will be considered: either as audiofile, averaged
    # power values, or the first section of the target audiofile (i.e. None)
    samples_noise = None
    if noise_file:
        if isinstance(noise_file, tuple):
            # tuple must contain samples and sampling rate
            samples_noise, sr_noise = noise_file
            if sr_noise != fil.sr:
                samples_noise, sr_noise = pyst.dsp.resample_audio(samples_noise,
                                                                  sr_noise,
                                                                  fil.sr)
                assert sr_noise == fil.sr
        # ensure string objects converted to pathlib.PosixPath objects:
        elif not isinstance(noise_file, pathlib.PosixPath) and isinstance(noise_file, str):
            noise_file = pathlib.Path(noise_file)
        # find out path information
        if isinstance(noise_file, pathlib.PosixPath):
            extension = noise_file.suffix
            if '.npy' in extension:
                # if noise power spectrum already calculated or not
                if 'powspec' in noise_file.stem or 'powerspectrum' in noise_file.stem:
                    noise_power = fil.load_power_vals(noise_file)
                    samples_noise = None
                elif 'stft' in noise_file.stem:
                    noise_power = np.load(noise_file)
                    noise_power = np.abs(noise_power)**2
            else:
                # assume audio pathway
                if duration_noise_ms is not None:
                    dur_sec = duration_noise_ms/1000
                else:
                    dur_sec = None
                samples_noise, sr_noise = pyst.loadsound(noise_file, 
                                                         fil.sr, 
                                                         dur_sec=dur_sec,
                                                         use_scipy=use_scipy)
                assert sr_noise == fil.sr
        if samples_noise is None and noise_power is None:
            raise TypeError('Expected one of the following: '+\
                '\ntype tuple containing (samples, samplerate) of noise data'+\
                    '\naudiofile pathway to noise file'+\
                        '\n.npy file with powerspectrum values for noise'+\
                            '\n\nDid not expect {} as input.'.format(noise_file))

    else:
        starting_noise_len = pyst.dsp.calc_frame_length(fil.sr, 
                                                         duration_noise_ms)
        samples_noise = samples_orig[:starting_noise_len]
    # if noise samples have been collected...
    # TODO improve snr / volume measurements
    if samples_noise is not None:
        # set how many subframes are needed to process entire noise signal
        fil.set_num_subframes(len(samples_noise), is_noise=True, zeropad=fil.zeropad)
    if visualize:
        pyst.feats.plot(samples_orig, 'signal', title='Signal to filter'.upper(),sr = fil.sr)
        pyst.feats.plot(samples_noise, 'signal', title= 'Noise samples to filter out'.upper(), sr=fil.sr)
    # prepare noise power matrix (if it's not loaded already)
    if fil.noise_subframes:
        if real_signal:
            #only the first half of fft (+1)
            total_rows = fil.num_fft_bins//2+1
        else:
            total_rows = fil.num_fft_bins
        noise_power = pyst.dsp.create_empty_matrix((total_rows,))
        section = 0
        for frame in range(fil.noise_subframes):
            noise_section = samples_noise[section:section+fil.frame_length]
            noise_w_win = pyst.dsp.apply_window(noise_section, fil.get_window(),
                                                zeropad=fil.zeropad)
            noise_fft = pyst.dsp.calc_fft(noise_w_win, real_signal=real_signal)
            noise_power_frame = pyst.dsp.calc_power(noise_fft)
            noise_power += noise_power_frame
            section += fil.overlap_length
        # welch's method: take average of power that has been collected
        # in windows
        noise_power = pyst.dsp.calc_average_power(noise_power, 
                                                   fil.noise_subframes)
        assert section == fil.noise_subframes * fil.overlap_length
    if visualize:
        pyst.feats.plot(noise_power, 'stft',title='Average noise power spectrum'.upper()+'\n{}'.format(frame_subtitle), energy_scale='power_to_db',
                             sr = fil.sr)
    
    # prepare target power matrix
    increment_length = int(fil.frame_length * fil.percent_overlap)
    total_rows =  increment_length + increment_length * fil.target_subframes
    filtered_sig = pyst.dsp.create_empty_matrix(
        (total_rows,), complex_vals=True)
    section = 0
    row = 0
    target_power_baseline = 0
    # increase/decrease noise values to increase strength of filter
    if filter_scale is None:
        filter_scale = 1
    noise_power *= filter_scale
    try:
        for frame in range(fil.target_subframes):
            target_section = samples_orig[section:section+fil.frame_length]
            target_w_window = pyst.dsp.apply_window(target_section,
                                                    fil.get_window(), 
                                                    zeropad=fil.zeropad)
            if visualize and frame % visualize_every_n_windows == 0:
                pyst.feats.plot(target_section,'signal', title='Signal'.upper()+' \nframe {}: {}'.format( frame+1,frame_subtitle),sr = fil.sr)
                pyst.feats.plot(target_w_window,'signal', title='Signal with {} window'.format(fil.window_type).upper()+'\nframe {}: {}'.format( frame+1,frame_subtitle),sr = fil.sr)
            target_fft = pyst.dsp.calc_fft(target_w_window, real_signal=real_signal)
            target_power = pyst.dsp.calc_power(target_fft)
            # now start filtering!!
            # initialize SNR matrix
            if visualize and frame % visualize_every_n_windows == 0:
                pyst.feats.plot(target_power,'stft', title='Signal power spectrum'.upper()+'\nframe {}: {}'.format( frame+1,frame_subtitle), energy_scale='power_to_db', sr = fil.sr)
            if 'wiener' in filter_type:
                enhanced_fft = fil.apply_wienerfilter(frame, 
                                                       target_fft, 
                                                       target_power, 
                                                       noise_power)
                if apply_postfilter:
                    enhanced_fft = fil.apply_postfilter(enhanced_fft,
                                                        target_fft,
                                                        target_power)
            elif 'band' in filter_type:
                target_phase = pyst.dsp.calc_phase(target_fft, 
                                                   radians=phase_radians)
                enhanced_fft = fil.apply_bandspecsub(target_power, 
                                                      target_phase, 
                                                      noise_power)
                if apply_postfilter:
                    enhanced_fft = fil.apply_postfilter(enhanced_fft,
                                                        target_fft,
                                                        target_power,
                                                        noise_power)
            
            enhanced_ifft = pyst.dsp.calc_ifft(enhanced_fft, 
                                               real_signal=real_signal)
            try:
                filtered_sig[row:row+fil.frame_length] += enhanced_ifft
            except ValueError:
                # with sample rate 22050, had some problems... zeropad missing frames
                if len(filtered_sig[row:row+fil.frame_length]) < fil.frame_length:
                    diff = fil.frame_length - len(filtered_sig[row:row+fil.frame_length])
                    filtered_sig[row:row+fil.frame_length] += \
                        enhanced_ifft[:fil.frame_length-diff]
                elif len(enhanced_ifft) < fil.frame_length:
                    diff = fil.frame_length - len(enhanced_ifft)
                    filtered_sig[row:row+fil.frame_length-diff] += enhanced_ifft
            if visualize and frame % visualize_every_n_windows == 0:
                pyst.feats.plot(filtered_sig,'signal', title='Filtered signal'.upper()+'\nup to frame {}: {}'.format(frame+1,frame_subtitle), sr = fil.sr)
            row += fil.overlap_length
            section += fil.overlap_length
    except ValueError as e:
        raise e
    assert row == fil.target_subframes * fil.overlap_length
    assert section == fil.target_subframes * fil.overlap_length
    # make enhanced_ifft values real
    enhanced_signal = filtered_sig.real
    if visualize:
        rows = len(filtered_sig)//increment_length
        cols = increment_length
        pyst.feats.plot(np.abs(filtered_sig.reshape((
                rows,cols,)))**2,
            'stft', title='Final filtered signal power spectrum'.upper()+'\n{}: {}'.format(filter_type,frame_subtitle), energy_scale='power_to_db')
        pyst.feats.plot(enhanced_signal,'signal', title='Final filtered signal'.upper()+'\n{}'.format(filter_type), sr = fil.sr)
    enhanced_signal = fil.check_volume(enhanced_signal)
    if len(enhanced_signal) > len(samples_orig):
        enhanced_signal = enhanced_signal[:len(samples_orig)]
    # for backwards compatibility
    if output_filename is not None or save2wav:
        if output_filename is None:
            output_filename = pyst.utils.get_date()+'.wav'
        saved_filename = pyst.savesound(str(output_filename), 
                                        enhanced_signal,
                                        sr=fil.sr,
                                        overwrite=overwrite)
    return enhanced_signal, fil.sr

def dataset_logger(audiofile_dir = None, recursive=True):
    '''Logs name, format, bitdepth, sr, duration of audiofiles, num_channels
    
    Parameters
    ----------
    audiofile_dir : str or pathlib.PosixPath
        The directory where audiofiles of interest are. If no directory 
        provided, the current working directory will be used.
        
    recursive : bool 
        If True, all audiofiles will be analyzed, also in nested directories.
        Otherwise, only the audio files in the immediate directory will be
        analyzed. (default True)
    
    
    Returns
    -------
    audiofile_dict : dict 
        Dictionary within a dictionary, holding the formats of the audiofiles in the 
        directory/ies.
        
    
    Examples
    --------
    >>> audio_info = dataset_logger()
    >>> # look at three audio files:
    >>> count = 0
    >>> for key, value in audio_info.items(): 
    ...:     for k, v in value.items(): 
    ...:         print(k, ' : ', v) 
    ...:     count += 1 
    ...:     print() 
    ...:     if count > 2: 
    ...:         break 
    audio  :  audiodata/dogbark_2channels.wav
    sr  :  48000
    num_channels  :  2
    dur_sec  :  0.389
    format_type  :  WAV
    bitdepth  :  PCM_16
    <BLANKLINE>
    audio  :  audiodata/python_traffic_pf.wav
    sr  :  48000
    num_channels  :  1
    dur_sec  :  1.86
    format_type  :  WAV
    bitdepth  :  DOUBLE
    <BLANKLINE>
    audio  :  audiodata/259672__nooc__this-is-not-right.wav
    sr  :  44100
    num_channels  :  1
    dur_sec  :  2.48453514739229
    format_type  :  WAV
    bitdepth  :  PCM_16
    
    
    See Also
    --------
    soundfile.available_subtypes
        The subtypes available with the package SoundFile
        
    soundfile.available_formats
        The formats available with the package SoundFile
    '''
    # ensure audio directory exists:
    if audiofile_dir is None:
        audiofile_dir = './'
    audiofile_dir = pyst.utils.check_dir(audiofile_dir)
    
    audiofiles = pyst.files.collect_audiofiles(audiofile_dir,
                                               recursive = recursive)
    
    audiofile_dict = dict()
    
    for i, audio in enumerate(audiofiles):
        # set sr to None to get audio file's sr
        # set mono to False to see if mono or stereo sound 
        y, sr = pyst.loadsound(audio, sr=None, mono=False)
        # see number of channels
        if len(y.shape) > 1:
            num_channels = y.shape[1]
        else:
            num_channels = 1
        
        dur_sec = len(y)/sr
        
        try:
            so = sf.SoundFile(audio)
            bitdepth = so.subtype
            format_type = so.format
        except RuntimeError:
            if isinstance(audio, str):
                audio = pathlib.Path(audio)
            format_type = audio.suffix.upper()[1:] # remove starting dot
            bitdepth = 'unknown'
        # ensure audio is string: if pathlib.PosixPath, it saves
        # the PurePath in the string and makes it difficult to deal 
        # with later.
        audio = str(audio)
        curr_audio_dict = dict(audio = audio,
                          sr = sr,
                          num_channels = num_channels,
                          dur_sec = dur_sec, 
                          format_type = format_type, 
                          bitdepth = bitdepth)
        
        
        audiofile_dict[audio] = curr_audio_dict
        pyst.utils.print_progress(i, len(audiofiles), task='logging audio file details')
        
    return audiofile_dict

def dataset_formatter(audiodirectory=None, recursive=False, new_dir=None, sr=None, dur_sec=None,
                      zeropad=False, format='WAV', bitdepth=None, overwrite=False, 
                      mono=False):
    '''Formats all audio files in a directory to set parameters.
    
    The audiofiles formatted can be limited to the specific directory or be 
    extended to the subfolders of that directory. 
    
    Parameters
    ----------
    audiodirectory : str or pathlib.PosixPath
        The directory where audio files live. If no directory provided, the current 
        working directory will be used.
        
    recursive : bool 
        If False, only audiofiles limited to the specific directory will be 
        formatted. If True, audio files in nested directories will also be
        formatted. (default False)
    
    new_dir : str or pathlib.PosixPath
        The audiofiles will be saved with the same structure in this directory. 
        If None, a default directory name with time stamp will be generated.
    
    sr : int 
        The desired sample rate to assign to the audio files. If None, the orignal
        sample rate will be maintained.
        
    dur_sec : int 
        The desired length in seconds the audio files should be limited to. If
        `zeropad` is set to True, the samples will be zeropadded to match this length
        if they are too short. If None, no limitation will be applied.
        
    zeropad : bool 
        If True, samples will be zeropadded to match `dur_sec`. (default False)
        
    format : str 
        The format to save the audio data in. (default 'WAV')
        
    bitdepth : int, str 
        The desired bitdepth. If int, 16 or 32 are possible. Defaults to 'PCM_16'.
        
    overwrite : bool 
        If True and `new_dir` is None, the audio data will be reformatted in the original
        directory and saved over any existing filenames. (default False)
        
    mono : bool 
        If True, the audio will be limited to a single channel. Note: not much has been 
        tested for stereo sound and PySoundTool. (default False)
        
    Returns
    -------
    directory : pathlib.PosixPath
        The directory where the formatted audio files are located.
    
    See Also
    --------
    pysoundtool.files.collect_audiofiles
        Collects audiofiles from a given directory.
        
    pysoundtool.files.conversion_formats
        The available formats for converting audio data.
        
    soundfile.available_subtypes
        The subtypes or bitdepth possible for soundfile
    '''
    if new_dir is None and not overwrite:
        new_dir = 'audiofile_reformat_'+pyst.utils.get_date()
        import warnings
        message = '\n\nATTENTION: Due to the risk of corrupting existing datasets, '+\
            'reformated audio will be saved in the following directory: '+\
                '\n{}\n'.format(new_dir)
        warnings.warn(message)
        
    # ensure new dir exists, and if not make it
    if new_dir is not None:
        new_dir = pyst.utils.check_dir(new_dir, make=True)
        
    if audiodirectory is None:
        audiodirectory = './'
    # ensure audiodirectory exists
    audiodirectory = pyst.utils.check_dir(audiodirectory, make=False)
    audiofiles = pyst.files.collect_audiofiles(audiodirectory,
                                               recursive=recursive)
    audiodir_parent = audiodirectory.stem
    # add this base directory to 'new_dir'
    new_dir = new_dir.joinpath(audiodir_parent)
    new_dir = pyst.utils.check_dir(new_dir,make=True)

    # set bitdepth for soundfile
    if bitdepth is None:
        # get default bitdepth from soundfile
        bd = sf.default_subtype(format)
    elif bitdepth == 16:
        bd = 'PCM_16'
    elif bitdepth == 32:
        bd = 'PCM_32'
    else:
        bd = bitdepth 
        
    # ensure format and bitdepth are valid for soundfile
    valid = sf.check_format(format, bd)
    if not valid:
        if not format in sf.available_formats():
            raise ValueError('Format {} is not available. Here is a list '+\
                'of available formats: \n{}'.format(format, 
                                                    sf.available_formats()))
        raise ValueError('Format {} cannot be assigned '.format(format)+\
            ' bitdepth {}.\nAvailable bitdepths include:'.format(bitdepth)+\
            '\n{}'.format(sf.available_subtypes(format)))
    
    for i, audio in enumerate(audiofiles):
        y, sr2 = pyst.loadsound(audio,
                               sr=sr, 
                               dur_sec = dur_sec,
                               mono = mono)
        # ensure the sr matches what was set
        if sr is not None:
            assert sr2 == sr
        
        if zeropad and dur_sec:
            goal_num_samples = int(dur_sec*sr2)
            y = pyst.dsp.set_signal_length(y, goal_num_samples)
            
        if overwrite is not True:
            # limit audiopath to parent dir 
            fparts = list(audio.parts)
            dir_idx = [i for i, j in enumerate(fparts) if j == audiodir_parent]
            dir_id = dir_idx[-1]
            fparts = fparts[dir_id+1:] # dir name included in new_dir path
            audio = pathlib.Path('/'.join(fparts))
            # maintains structure of old directory in new directory            
            new_filename = new_dir.joinpath(audio)
        else:
            new_filename = audio
            
        # change the audio file name to match desired file format:
        if format:
            new_filename = pyst.files.replace_ext(new_filename, format.lower())
            
        try:
            new_filename = pyst.savesound(new_filename, y, sr2, 
                                      overwrite=overwrite,
                                      format=format,subtype=bd)
        except FileExistsError:
            print('File {} already exists.'.format(new_filename))
            
        pyst.utils.print_progress(i, len(audiofiles), 
                                  task = 'reformatting dataset')
        
    if new_dir:
        return new_dir
    else:
        return audiodirectory
 

# TODO speed this up, e.g. preload noise data?
# TODO randomize sections of noise applied
def create_denoise_data(cleandata_dir, noisedata_dir, trainingdata_dir, limit=None,
                            snr_levels = None, delay_mainsound_sec = None, seed = None,
                            overwrite = False, **kwargs):
    '''Applies noise to clean audio; saves clean and noisy audio to `traingingdata_dir`.

    Parameters
    ----------
    cleandata_dir : str, pathlib.PosixPath
        Name of folder containing clean audio data for autoencoder. E.g. 'clean_speech'
    noisedata_dir : str, pathlib.PosixPath
        Name of folder containing noise to add to clean data. E.g. 'noise'
    trainingdata_dir : str, pathlib.PosixPath
        Directory to save newly created train, validation, and test data
    limit : int, optional
        Limit in number of audiofiles used for training data
    snr_levels : list of ints, optional
        List of varying signal-to-noise ratios to apply to noise levels.
        (default None)
    delay_mainsound_sec : int, float, optional
        Amount in seconds the main sound should be delayed. In other words, in seconds how
        long the background sound should play before the clean / main / target audio starts.
        (default None)
    seed : int 
        A value to allow random order of audiofiles to be predictable. 
        (default None). If None, the order of audiofiles will not be predictable.
    overwrite : bool 
        If True, a new dataset will be created regardless of whether or not a matching 
        directory already exists. (default False)
    **kwargs : additional keyword arguments
        The keyword arguments for pysoundtool.files.loadsound
        

    Returns
    -------
    saveinput_path : pathlib.PosixPath
        Path to where noisy audio files are located
    saveoutput_path : pathlib.PosixPath   
        Path to where clean audio files are located
        
    See Also
    --------
    pysoundtool.files.loadsound
        Loads audiofiles.
    
    pysoundtool.dsp.add_backgroundsound
        Add background sound / noise to signal at a determined signal-to-noise ratio.
    '''
    import math
    import time
    
    start = time.time()

    # check to ensure clean and noisy data are there
    # and turn them into pathlib.PosixPath objects:
    cleandata_dir = pyst.utils.check_dir(cleandata_dir, make=False)
    noisedata_dir = pyst.utils.check_dir(noisedata_dir, make=False)
    trainingdata_dir = pyst.utils.string2pathlib(trainingdata_dir)
    
    cleandata_folder = 'clean'
    noisedata_folder = 'noisy'
    if limit is not None:
        cleandata_folder += '_limit'+str(limit)
        noisedata_folder += '_limit'+str(limit)
    
    newdata_clean_dir = trainingdata_dir.joinpath(cleandata_folder)
    newdata_noisy_dir = trainingdata_dir.joinpath(noisedata_folder)
    
    # See if databases already exist:
    if not overwrite:
        try:
            newdata_clean_dir = pyst.utils.check_dir(newdata_clean_dir, make=False,
                                                     append = False)
            newdata_noisy_dir = pyst.utils.check_dir(newdata_noisy_dir, make=False,
                                                     append = False)
        except FileExistsError:
            raise FileExistsError('Datasets already exist at this location. Set '+\
                '`append` to True or designate a new directory.')
        except FileNotFoundError:
            pass
    
    # create directory to save new data (if not exist)
    newdata_clean_dir = pyst.utils.check_dir(newdata_clean_dir, make = True)
    newdata_noisy_dir = pyst.utils.check_dir(newdata_noisy_dir, make = True)
   
    # collect audiofiles (not limited to .wav files)
    cleanaudio = sorted(pyst.files.collect_audiofiles(cleandata_dir,
                                                      hidden_files = False,
                                                      wav_only = False,
                                                      recursive = False))
    noiseaudio = sorted(pyst.files.collect_audiofiles(noisedata_dir,
                                                      hidden_files = False,
                                                      wav_only = False,
                                                      recursive = False))
    
    if seed is not None:
        random.seed(seed)
    random.shuffle(cleanaudio)
    
    if limit is not None:
        cleanaudio = cleanaudio[:limit]
    
    # ensure snr_levels is array-like 
    if snr_levels is not None:
        if not isinstance(snr_levels, list) and not isinstance(snr_levels, np.ndarray):
            snr_levels = list(snr_levels)
    
    for i, wavefile in enumerate(cleanaudio):
        pyst.utils.print_progress(iteration=i, 
                    total_iterations=len(cleanaudio),
                    task='clean and noisy audio data generation')
        # no random seed applied here:
        # each choice would be the same for each iteration
        noise = random.choice(noiseaudio)
        if snr_levels is not None:
            snr = random.choice(snr_levels)
        else:
            snr = None
        clean_stem = wavefile.stem
        noise_stem = noise.stem
        # load clean data to get duration
        clean_data, sr = pyst.loadsound(wavefile, **kwargs)
        clean_seconds = len(clean_data)/sr
        noisy_data, sr, snr_appx = pyst.dsp.add_backgroundsound(audio_main = wavefile, 
                                                      audio_background = noise, 
                                                      snr = snr, 
                                                      delay_mainsound_sec=delay_mainsound_sec, 
                                                      total_len_sec = clean_seconds,
                                                      wrap = True,
                                                      **kwargs)
        # ensure both noisy and clean files have same beginning to filename (i.e. clean filename)
        noisydata_filename = newdata_noisy_dir.joinpath(clean_stem+'_'+noise_stem\
            +'_snr'+str(snr)+'.wav')
        cleandata_filename = newdata_clean_dir.joinpath(clean_stem+'.wav')     
        write(noisydata_filename, sr, noisy_data)
        write(cleandata_filename, sr, clean_data)

    end = time.time()
    total_time, units = pyst.utils.adjust_time_units(end-start)
    print('Data creation took a total of {} {}.'.format(
        round(total_time,2), 
        units))

    return newdata_noisy_dir, newdata_clean_dir

def envclassifier_feats(
    data_dir,
    data_features_dir = None,
    feature_type = None,
    dur_sec = 1,
    perc_train = 0.8,
    **kwargs):
    '''Environment Classifier: feature extraction of scene audio into train, val, & test datasets.
    
    Saves extracted feature datasets (train, val, test datasets) as well as 
    feature extraction settings in the directory `data_features_dir`.
    
    Parameters
    ----------
    data_dir : str or pathlib.PosixPath
        The directory with scene subfolders (e.g. 'air_conditioner', 'traffic') that 
        contain audio files belonging to that scene (e.g. 'air_conditioner/ac1.wav',
        'air_conditioner/ac2.wav', 'traffic/t1.wav'). 
    
    data_features_dir : str or pathlib.PosixPath, optional
        The directory where feature extraction related to the dataset will be stored. 
        Within this directory, a unique subfolder will be created each time features are
        extracted. This allows several versions of extracted features on the same dataset
        without overwriting files.
    
    feature_type : str 
        The type of features to be extracted. Options: 'stft', 'powspec', 'mfcc', 'fbank'. (default 'fbank')
        
    dur_sec : int, float
        The duration of each audio sample to be extracted. (default 1)
    
    kwargs : additional keyword arguments
        Keyword arguments for `pysoundtool.feats.save_features_datasets` and 
        `pysoundtool.feats.get_feats`.
        
    Returns
    -------
    feat_extraction_dir : pathlib.PosixPath
        The pathway to where all feature extraction files can be found, including datasets.
        
    See Also
    --------
    pysoundtool.feats.get_feats
        Extract features from audio file or audio data.
        
    pysoundtool.feats.save_features_datasets
        Preparation of acoustic features in train, validation and test datasets.
    '''
    if data_features_dir is None:
        data_features_dir = './audiodata/example_feats_models/envclassifier/'
    if feature_type is None:
        feature_type = 'fbank'
    if 'signal' in feature_type:
        raise ValueError('Feature type "signal" is not yet supported for this model.')

    feat_extraction_dir = 'features_'+ feature_type + '_' + pyst.utils.get_date()

    # collect labels 
    labels = []
    data_dir = pyst.utils.string2pathlib(data_dir)
    for label in data_dir.glob('*/'):
        labels.append(label.stem)
    labels = set(labels)

    # create paths for what we need to save:
    data_features_dir = pyst.utils.check_dir(data_features_dir)
    feat_extraction_dir = data_features_dir.joinpath(feat_extraction_dir)
    feat_extraction_dir = pyst.utils.check_dir(feat_extraction_dir, make=True)

    # dictionaries containing encoding and decoding labels:
    dict_encode_path = feat_extraction_dir.joinpath('dict_encode.csv')
    dict_decode_path = feat_extraction_dir.joinpath('dict_decode.csv')
    # dictionary for which audio paths are assigned to which labels:
    dict_encdodedlabel2audio_path = feat_extraction_dir.joinpath('dict_encdodedlabel2audio.csv')

    # designate where to save train, val, and test data
    data_train_path = feat_extraction_dir.joinpath('{}_data_{}.npy'.format('train',
                                                                        feature_type))
    data_val_path = feat_extraction_dir.joinpath('{}_data_{}.npy'.format('val',
                                                                        feature_type))
    data_test_path = feat_extraction_dir.joinpath('{}_data_{}.npy'.format('test',
                                                                        feature_type))

    # create and save encoding/decoding labels dicts
    dict_encode, dict_decode = pyst.datasets.create_dicts_labelsencoded(labels)
    dict_encode_path = pyst.utils.save_dict(
        filename = dict_encode_path,
        dict2save = dict_encode, 
        overwrite=False)
    dict_decode_path = pyst.utils.save_dict(
        filename = dict_decode_path,
        dict2save = dict_encode, 
        overwrite=False)

    # save audio paths to each label in dict 
    paths_list = pyst.files.collect_audiofiles(data_dir, recursive=True)
    paths_list = sorted(paths_list)

    dict_encodedlabel2audio = pyst.datasets.create_encodedlabel2audio_dict(dict_encode,
                                                        paths_list)
    dict_encdodedlabel2audio_path = pyst.utils.save_dict(
        dict2save = dict_encodedlabel2audio, 
        filename = dict_encdodedlabel2audio_path, 
        overwrite=False)
    # assign audiofiles into train, validation, and test datasets
    train, val, test = pyst.datasets.audio2datasets(dict_encdodedlabel2audio_path,
                                                perc_train=perc_train,
                                                limit=None,
                                                seed=40)

    # save audiofiles for each dataset to dict and save
    dataset_dict = dict([('train',train),('val', val),('test',test)])
    dataset_dict_path = feat_extraction_dir.joinpath('dataset_audiofiles.csv')
    dataset_dict_path = pyst.utils.save_dict(
        dict2save = dataset_dict, 
        filename = dataset_dict_path, 
        overwrite=True)
    # save paths to where extracted features of each dataset will be saved to dict w same keys
    datasets_path2save_dict = dict([('train',data_train_path),
                                    ('val', data_val_path),
                                    ('test',data_test_path)])

    # extract features

    start = time.time()

    dataset_dict, datasets_path2save_dict = pyst.feats.save_features_datasets(
        datasets_dict = dataset_dict,
        datasets_path2save_dict = datasets_path2save_dict,
        labeled_data = True,
        feature_type = feature_type,
        dur_sec = dur_sec,
        **kwargs)

    end = time.time()
    
    total_dur_sec = end-start
    total_dur, units = pyst.utils.adjust_time_units(total_dur_sec)
    print('\nFinished! Total duration: {} {}.'.format(round(total_dur,2), units))

    # save which audiofiles were extracted for each dataset
    # save where extracted data were saved
    # save how long feature extraction took
    dataprep_settings = dict(dataset_dict=dataset_dict,
                            datasets_path2save_dict=datasets_path2save_dict,
                            total_dur_sec = total_dur_sec)
    dataprep_settings_path = pyst.utils.save_dict(
        dict2save = dataprep_settings,
        filename = feat_extraction_dir.joinpath('dataset_audio_assignments.csv'))
    
    return feat_extraction_dir

def denoiser_feats(
    data_clean_dir,
    data_noisy_dir,
    data_features_dir = None,
    feature_type = None,
    dur_sec = 3,
    frames_per_sample = 11,
    limit = None,
    perc_train = 0.8,
    **kwargs):
    '''Autoencoder Denoiser: feature extraction of clean & noisy audio into train, val, & test datasets.
    
    Saves extracted feature datasets (train, val, test datasets) as well as 
    feature extraction settings in the directory `data_features_dir`.
    
    Parameters
    ----------
    data_clean_dir : str or pathlib.PosixPath
        The directory with clean audio files. 
    
    data_noisy_dir : str or pathlib.PosixPath
        The directory with noisy audio files. These should be the same as the clean audio,
        except noise has been added. 
    
    data_features_dir : str or pathlib.PosixPath, optional
        The directory where feature extraction related to the dataset will be stored. 
        Within this directory, a unique subfolder will be created each time features are
        extracted. This allows several versions of extracted features on the same dataset
        without overwriting files.
    
    feature_type : str, optional
        The type of features to be extracted. Options: 'stft', 'powspec', 'mfcc', 'fbank' or
        'signal'. (default 'fbank')
        
    dur_sec : int, float
        The duration of each audio sample to be extracted. (default 1)
        
    frames_per_sample : int 
        If you want to section each audio file feature data into smaller frames. This might be 
        useful for speech related contexts. (Can avoid this by simply reshaping data later)
        (default 11)
        
    limit : int, optional
        The limit of audio files for feature extraction. (default None)
    
    kwargs : additional keyword arguments
        Keyword arguments for `pysoundtool.feats.save_features_datasets` and 
        `pysoundtool.feats.get_feats`.
        
    Returns
    -------
    feat_extraction_dir : pathlib.PosixPath
        The pathway to where all feature extraction files can be found, including datasets.
        
    See Also
    --------
    pysoundtool.datasets.create_denoise_data
        Applies noise at specified SNR levels to clean audio files.
    
    pysoundtool.feats.get_feats
        Extract features from audio file or audio data.
        
    pysoundtool.feats.save_features_datasets
        Preparation of acoustic features in train, validation and test datasets.
    '''
    # ensure these are not None, and if so, fill with sample data
    if data_features_dir is None:
        data_features_dir = './audiodata/example_feats_models/denoiser/'
    if feature_type is None:
        feature_type = 'fbank'
    
    # create unique directory for feature extraction session:
    feat_extraction_dir = 'features_'+feature_type + '_' + pyst.utils.get_date()

    # 1) Ensure clean and noisy data directories exist
    audio_clean_path = pyst.utils.check_dir(data_clean_dir, make=False)
    audio_noisy_path = pyst.utils.check_dir(data_noisy_dir, make=False)

    # 2) create paths for what we need to save:
    denoise_data_path = pyst.utils.check_dir(data_features_dir, make=True)
    feat_extraction_dir = denoise_data_path.joinpath(feat_extraction_dir)
    feat_extraction_dir = pyst.utils.check_dir(feat_extraction_dir, make=True)
    # Noisy and clean train, val, and test data paths:
    data_train_noisy_path = feat_extraction_dir.joinpath('{}_data_{}_{}.npy'.format('train',
                                                                            'noisy',
                                                                        feature_type))
    data_val_noisy_path = feat_extraction_dir.joinpath('{}_data_{}_{}.npy'.format('val',
                                                                        'noisy',
                                                                        feature_type))
    data_test_noisy_path = feat_extraction_dir.joinpath('{}_data_{}_{}.npy'.format('test',
                                                                        'noisy',
                                                                        feature_type))
    data_train_clean_path = feat_extraction_dir.joinpath('{}_data_{}_{}.npy'.format('train',
                                                                            'clean',
                                                                        feature_type))
    data_val_clean_path = feat_extraction_dir.joinpath('{}_data_{}_{}.npy'.format('val',
                                                                        'clean',
                                                                        feature_type))
    data_test_clean_path = feat_extraction_dir.joinpath('{}_data_{}_{}.npy'.format('test',
                                                                        'clean',
                                                                        feature_type))

    # 3) collect audiofiles and divide them into train, val, and test datasets
    # noisy data
    noisyaudio = pyst.files.collect_audiofiles(audio_noisy_path, 
                                                    hidden_files = False,
                                                    wav_only = False,
                                                    recursive = False)
    # sort audio (can compare if noisy and clean datasets are compatible)
    noisyaudio = sorted(noisyaudio)
    if limit is not None:
        noisyaudio =  noisyaudio[:limit]

    # clean data
    cleanaudio = pyst.files.collect_audiofiles(audio_clean_path, 
                                                    hidden_files = False,
                                                    wav_only = False,
                                                    recursive = False)
    cleanaudio = sorted(cleanaudio)
    if limit is not None:
        cleanaudio =  cleanaudio[:limit]


    # check if they match up: (expects clean file name to be in noisy file name)
    for i, audiofile in enumerate(noisyaudio):
        if not pyst.utils.check_noisy_clean_match(audiofile, cleanaudio[i]):
            raise ValueError('The noisy and clean audio datasets do not appear to match.')

    # save collected audiofiles for noisy and clean datasets to dictionary
    noisy_audio_dict = dict([('noisy', noisyaudio)])
    clean_audio_dict = dict([('clean', cleanaudio)])
    
    noisy_audio_dict_path = feat_extraction_dir.joinpath('noisy_audio.csv')
    noisy_audio_dict_path = pyst.utils.save_dict(
        dict2save = noisy_audio_dict, 
        filename = noisy_audio_dict_path,
        overwrite=False)
    clean_audio_dict_path = feat_extraction_dir.joinpath('clean_audio.csv')
    clean_audio_dict_path = pyst.utils.save_dict(
        dict2save = clean_audio_dict, 
        filename = clean_audio_dict_path,
        overwrite=False)
    # separate into datasets
    train_noisy, val_noisy, test_noisy = pyst.datasets.audio2datasets(
        noisy_audio_dict_path, perc_train = perc_train, seed=40)
    train_clean, val_clean, test_clean = pyst.datasets.audio2datasets(
        clean_audio_dict_path,
        perc_train = perc_train,
        seed=40)

    # save train, val, test dataset assignments to dict
    dataset_dict_noisy = dict([('train', train_noisy),('val', val_noisy),('test', test_noisy)])
    dataset_dict_clean = dict([('train', train_clean),('val', val_clean),('test', test_clean)])

    # keep track of paths to save data
    dataset_paths_noisy_dict = dict([('train',data_train_noisy_path),
                                    ('val', data_val_noisy_path),
                                    ('test',data_test_noisy_path)])
    dataset_paths_clean_dict = dict([('train',data_train_clean_path),
                                    ('val', data_val_clean_path),
                                    ('test',data_test_clean_path)])

    path2_noisy_datasets = feat_extraction_dir.joinpath('audiofiles_datasets_noisy.csv')
    path2_clean_datasets = feat_extraction_dir.joinpath('audiofiles_datasets_clean.csv')

    # save dicts to .csv files
    path2_noisy_datasets = pyst.utils.save_dict(
        dict2save = dataset_dict_noisy,
        filename = path2_noisy_datasets,
        overwrite=False)
    path2_clean_datasets = pyst.utils.save_dict(
        dict2save = dataset_dict_clean,
        filename = path2_clean_datasets,
        overwrite=False)

    # 5) extract features
            
    # ensure the noisy and clean values match up:
    for key, value in dataset_dict_noisy.items():
        for j, audiofile in enumerate(value):
            if not pyst.utils.check_noisy_clean_match(audiofile,
                                                    dataset_dict_clean[key][j]):
                raise ValueError('There is a mismatch between noisy and clean audio. '+\
                    '\nThe noisy file:\n{}'.format(dataset_dict_noisy[key][i])+\
                        '\ndoes not seem to match the clean file:\n{}'.format(audiofile))

    start = time.time()

    # first clean data
    dataset_dict_clean, dataset_paths_clean_dict = pyst.feats.save_features_datasets(
        datasets_dict = dataset_dict_clean,
        datasets_path2save_dict = dataset_paths_clean_dict,
        feature_type = feature_type + ' clean',
        dur_sec = dur_sec,
        frames_per_sample = frames_per_sample,
        **kwargs)
        
    # then noisy data
    dataset_dict_noisy, dataset_paths_noisy_dict = pyst.feats.save_features_datasets(
        datasets_dict = dataset_dict_noisy,
        datasets_path2save_dict = dataset_paths_noisy_dict,
        feature_type = feature_type + ' noisy',
        dur_sec = dur_sec,
        frames_per_sample = frames_per_sample,
        **kwargs)

    end = time.time()

    total_dur_sec = round(end-start,2)
    total_dur, units = pyst.utils.adjust_time_units(total_dur_sec)
    print('\nFinished! Total duration: {} {}.'.format(total_dur, units))
    # save which audiofiles were extracted for each dataset
    # save where extracted data were saved
    # save total duration of feature extraction
    dataprep_settings = dict(dataset_dict_noisy = dataset_dict_noisy,
                            dataset_paths_noisy_dict = dataset_paths_noisy_dict,
                            dataset_dict_clean = dataset_dict_clean,
                            dataset_paths_clean_dict = dataset_paths_clean_dict,
                            total_dur_sec = total_dur_sec)
    dataprep_settings_path = pyst.utils.save_dict(
        dict2save = dataprep_settings,
        filename = feat_extraction_dir.joinpath('dataset_audio_assignments.csv'))
    return feat_extraction_dir



# TODO include example extraction data in feature_extraction_dir?
def denoiser_train(feature_extraction_dir,
                   model_name = 'model_autoencoder_denoise',
                   feature_type = None,
                   use_generator = True,
                   normalized = False,
                   patience = 10, 
                   **kwargs):
    '''Collects training features and train autoencoder denoiser.
    
    Parameters
    ----------
    feature_extraction_dir : str or pathlib.PosixPath
        Directory where extracted feature files are located (format .npy).
        
    model_name : str
        The name for the model. This can be quite generic as the date up to 
        the millisecond will be added to ensure a unique name for each trained model.
        (default 'model_autoencoder_denoise')
        
    feature_type : str, optional
        The type of features that will be used to train the model. This is 
        only for the purposes of naming the model. If set to None, it will 
        not be included in the model name.
        
    use_generator : bool 
        If True, a generator will be used to feed training data to the model. Otherwise
        the entire training data will be used to train the model all at once.
        (default True)
        
    normalized : bool 
        If False, the data will be normalized before feeding to the model.
        (default False)
        
    patience : int 
        Number of epochs to train without improvement before early stopping.
        
    **kwargs : additional keyword arguments
        The keyword arguments for keras.fit() and keras.fit_generator(). Note, 
        the keyword arguments differ for validation data so be sure to use the 
        correct keyword arguments, depending on if you use the generator or not.
        TODO: add link to keras.fit() and keras.fit_generator()
        
    Returns
    -------
    model_dir : pathlib.PosixPath
        The directory where the model and associated files can be found.
        
    See Also
    --------
    pysoundtool.datasets.separate_train_val_test_files
        Generates paths lists for train, validation, and test files. Useful
        for noisy vs clean datasets and also for multiple training files.
    
    pysoundtool.models.generator
        The generator function that feeds data to the model.
        
    pysoundtool.models.modelsetup.setup_callbacks
        The function that sets up callbacks (e.g. logging, save best model, early
        stopping, etc.)
        
    pysoundtool.models.template_models.autoencoder_denoise
        Template model architecture for basic autoencoder denoiser.
    '''
    dataset_path = pyst.utils.check_dir(feature_extraction_dir, make=False)
    
    # designate where to save model and related files
    if feature_type:
        model_name += '_'+feature_type + '_' + pyst.utils.get_date() 
    else:
        model_name += '_' + pyst.utils.get_date() 
    model_dir = dataset_path.joinpath(model_name)
    model_dir = pyst.utils.check_dir(model_dir, make=True)
    model_name += '.h5'
    model_path = model_dir.joinpath(model_name)
    
    # prepare features files to load for training
    features_files = list(dataset_path.glob('*.npy'))
    # NamedTuple: 'datasets.train.noisy', 'datasets.train.clean', etc.
    datasets = pyst.datasets.separate_train_val_test_files(
        features_files)
    
    # TODO test this:
    if not datasets.train:
        # perhaps data files located in subdirectories 
        features_files = list(dataset_path.glob('**/*.npy'))
        datasets = pyst.datasets.separate_train_val_test_files(
            features_files)
        if not datasets.train:
            raise FileNotFoundError('Could not locate train, validation, or test '+\
                '.npy files in the provided directory: \n{}'.format(dataset_path) +\
                    '\nThis program expects "train", "val", or "test" to be '+\
                        'included in each filename (not parent directory/ies) names.')
    
    # only need train and val feature data for autoencoder 
    train_paths_noisy, train_paths_clean = datasets.train.noisy, datasets.train.clean
    val_paths_noisy, val_paths_clean = datasets.val.noisy, datasets.val.clean
    
    # make sure both dataset pathways match in length and order:
    try:
        assert len(train_paths_noisy) == len(train_paths_clean)
        assert len(val_paths_noisy) == len(val_paths_clean)
    except AssertionError:
        raise ValueError('Noisy and Clean datasets do not match in length. '+\
            'They must be the same length.')
    train_paths_noisy = sorted(train_paths_noisy)
    train_paths_clean = sorted(train_paths_clean)
    val_paths_noisy = sorted(val_paths_noisy)
    val_paths_clean = sorted(val_paths_clean)
    
    # load smaller dataset to determine input size:
    data_val_noisy = np.load(val_paths_noisy[0])
    # expect shape (num_audiofiles, batch_size, num_frames, num_features)
    input_shape = data_val_noisy.shape[2:] + (1,)
    del data_val_noisy

    
    # setup model 
    denoiser, settings_dict = pystmodels.autoencoder_denoise(
        input_shape = input_shape)
    
    # create callbacks variable if not in kwargs
    # allow users to use different callbacks if desired
    if 'callbacks' not in kwargs:
        callbacks = pystmodels.setup_callbacks(patience = patience,
                                                best_modelname = model_path, 
                                                log_filename = model_dir.joinpath('log.csv'))
    adm = keras.optimizers.Adam(learning_rate=0.0001)
    denoiser.compile(optimizer=adm, loss='binary_crossentropy')

    # TODO remove?
    # save variables that are not too large:
    local_variables = locals()
    global_variables = globals()
    pyst.utils.save_dict(
        dict2save = local_variables, 
        filename = model_dir.joinpath('local_variables_{}.csv'.format(
                            model_name)),
                        overwrite=True)
    pyst.utils.save_dict(
        dict2save = global_variables,
        filename = model_dir.joinpath('global_variables_{}.csv'.format(
                            model_name)),
        overwrite = True)
        
    # start training
    start = time.time()

    for i, train_path in enumerate(train_paths_noisy):
        if i == 0:
            if 'epochs' in kwargs:
                epochs = kwargs['epochs']
            else:
                epochs = 10 # default in Keras
            total_epochs = epochs * len(train_paths_noisy)
            print('\n\nThe model will be trained {} epochs per '.format(epochs)+\
                'training session. \nTotal possible epochs: {}\n\n'.format(total_epochs))
        start_session = time.time()
        data_train_noisy_path = train_path
        data_train_clean_path = train_paths_clean[i]
        # just use first validation data file
        data_val_noisy_path = val_paths_noisy[0]
        data_val_clean_path = val_paths_clean[0]

        print('\nTRAINING SESSION ',i+1)
        print("Training on: ")
        print(data_train_noisy_path)
        print(data_train_clean_path)
        print()
        
        data_train_noisy = np.load(data_train_noisy_path)
        data_train_clean = np.load(data_train_clean_path)
        data_val_noisy = np.load(data_val_noisy_path)
        data_val_clean = np.load(data_val_clean_path)

        # reinitiate 'callbacks' for additional iterations
        # TODO test for when callbacks already in **kwargs
        if i > 0: 
            if 'callbacks' not in kwargs:
                callbacks = pystmodels.setup_callbacks(patience = patience,
                                                        best_modelname = model_path, 
                                                        log_filename = model_dir.joinpath('log.csv'))
            #else:
                ## apply callbacks set in **kwargs
                #callbacks = kwargs['callbacks']

        if use_generator:
            train_generator = pystmodels.Generator(data_matrix1 = data_train_noisy, 
                                                    data_matrix2 = data_train_clean,
                                                    normalized = normalized,
                                                    add_tensor_last = True)
            val_generator = pystmodels.Generator(data_matrix1 = data_val_noisy,
                                                data_matrix2 = data_val_clean,
                                                normalized = normalized,
                                                add_tensor_last = True)

            train_generator.generator()
            val_generator.generator()
            try:
                history = denoiser.fit_generator(train_generator.generator(),
                                                    steps_per_epoch = data_train_noisy.shape[0],
                                                    callbacks = callbacks,
                                                    validation_data = val_generator.generator(),
                                                    validation_steps = data_val_noisy.shape[0], 
                                                    **kwargs)
            except ValueError as e:
                print('\nValueError: ', e)
                raise ValueError('Try setting changing the parameter '+\
                    '`add_tensor_last` (in function '+\
                        '`pysoundtool.models.dataprep.Generator`)'+\
                        ' to either True, False, or None.')

        else:
            #reshape to mix samples and batchsizes:
            train_shape = (data_train_noisy.shape[0]*data_train_noisy.shape[1],)+ data_train_noisy.shape[2:] + (1,)
            val_shape = (data_val_noisy.shape[0]*data_val_noisy.shape[1],)+ data_val_noisy.shape[2:] + (1,)
            
            if not normalized:
                data_train_noisy = pyst.feats.normalize(data_train_noisy)
                data_train_clean = pyst.feats.normalize(data_train_clean)
                data_val_noisy = pyst.feats.normalize(data_val_noisy)
                data_val_clean = pyst.feats.normalize(data_val_clean)
                
            X_train = data_train_noisy.reshape(train_shape)
            y_train = data_train_clean.reshape(train_shape)
            X_val = data_val_noisy.reshape(val_shape)
            y_val = data_val_clean.reshape(val_shape)
            
            history = denoiser.fit(X_train, y_train,
                            batch_size = data_train_noisy.shape[1],
                            callbacks = callbacks, 
                            validation_data = (X_val, y_val),
                            **kwargs)
        end_session = time.time()
        total_dur_sec_session = round(end_session-start_session,2)
        model_features_dict = dict(model_path = model_path,
                                data_train_noisy_path = data_train_noisy_path,
                                data_val_noisy_path = data_val_noisy_path, 
                                data_train_clean_path = data_train_clean_path, 
                                data_val_clean_path = data_val_clean_path,
                                total_dur_sec_session = total_dur_sec_session,
                                use_generator = use_generator,
                                kwargs = kwargs)
        model_features_dict.update(settings_dict)
        if i == len(train_paths_noisy)-1:
            end = time.time()
            total_duration_seconds = round(end-start,2)
            time_dict = dict(total_duration_seconds=total_duration_seconds)
            model_features_dict.update(time_dict)

        model_features_dict_path = model_dir.joinpath('info_{}_{}.csv'.format(
            model_name, i))
        model_features_dict_path = pyst.utils.save_dict(
            dict2save = model_features_dict,
            filename = model_features_dict_path)
    print('\nFinished training the model. The model and associated files can be '+\
        'found here: \n{}'.format(model_dir))
    
    return model_dir, history


    
###############################################################################

# TODO include example extraction data in feature_extraction_dir?
def envclassifier_train(feature_extraction_dir,
                        model_name = 'model_cnn_classifier',
                        feature_type = None,
                        use_generator = True,
                        normalized = False,
                        patience = 15,
                        add_tensor_last = True,
                        **kwargs):
    '''Collects training features and trains cnn environment classifier.
    
    This model may be applied to any speech and label scenario, for example, 
    male vs female speech, clinical vs healthy speech, simple speech / word
    recognition, as well as noise / scene / environment classification.
    
    Parameters
    ----------
    feature_extraction_dir : str or pathlib.PosixPath
        Directory where extracted feature files are located (format .npy).
    
    model_name : str
        The name for the model. This can be quite generic as the date up to 
        the millisecond will be added to ensure a unique name for each trained model.
        (default 'model_cnn_classifier')
        
    feature_type : str, optional
        The type of features that will be used to train the model. This is 
        only for the purposes of naming the model. If set to None, it will 
        not be included in the model name.
        
    use_generator : bool 
        If True, a generator will be used to feed training data to the model. Otherwise
        the entire training data will be used to train the model all at once.
        (default True)
        
    normalized : bool 
        If False, the data will be normalized before feeding to the model.
        (default False)
        
    patience : int 
        Number of epochs to train without improvement before early stopping.
        
    **kwargs : additional keyword arguments
        The keyword arguments for keras.fit() and keras.fit_generator(). Note, 
        the keyword arguments differ for validation data so be sure to use the 
        correct keyword arguments, depending on if you use the generator or not.
        TODO: add link to keras.fit() and keras.fit_generator()
        
    Returns
    -------
    model_dir : pathlib.PosixPath
        The directory where the model and associated files can be found.
        
    See Also
    --------
    pysoundtool.datasets.separate_train_val_test_files
        Generates paths lists for train, validation, and test files. Useful
        for noisy vs clean datasets and also for multiple training files.
    
    pysoundtool.models.generator
        The generator function that feeds data to the model.
        
    pysoundtool.models.modelsetup.setup_callbacks
        The function that sets up callbacks (e.g. logging, save best model, early
        stopping, etc.)
        
    pysoundtool.models.template_models.cnn_classifier
        Template model architecture for a low-computational CNN sound classifier.
    '''
    # ensure feature_extraction_folder exists:
    if feature_extraction_dir is None:
        feature_extraction_dir = './audiodata/example_feats_models/envclassifier/'+\
            'features_fbank_6m20d0h18m11s123ms/'
    dataset_path = pyst.utils.check_dir(feature_extraction_dir, make=False)
    
    # designate where to save model and related files
    if feature_type:
        model_name += '_'+feature_type + '_' + pyst.utils.get_date() 
    else:
        model_name += '_' + pyst.utils.get_date() 
    model_dir = dataset_path.joinpath(model_name)
    model_dir = pyst.utils.check_dir(model_dir, make=True)
    model_name += '.h5'
    model_path = model_dir.joinpath(model_name)
    
    # prepare features files to load for training
    features_files = list(dataset_path.glob('*.npy'))
    # NamedTuple: 'datasets.train', 'datasets.val', 'datasets.test'
    datasets = pyst.datasets.separate_train_val_test_files(
        features_files)
    
    # TODO test
    if not datasets.train:
        # perhaps data files located in subdirectories 
        features_files = list(dataset_path.glob('**/*.npy'))
        datasets = pyst.datasets.separate_train_val_test_files(
            features_files)
        if not datasets.train:
            raise FileNotFoundError('Could not locate train, validation, or test '+\
                '.npy files in the provided directory: \n{}'.format(dataset_path) +\
                    '\nThis program expects "train", "val", or "test" to be '+\
                        'included in each filename (not parent directory/ies) names.')
    
    train_paths = datasets.train
    val_paths = datasets.val 
    test_paths = datasets.test
    
    # need dictionary for decoding labels:
    dict_decode_path = dataset_path.joinpath('dict_decode.csv')
    if not os.path.exists(dict_decode_path):
        raise FileNotFoundError('Could not find {}.'.format(dict_decode_path))
    dict_decode = pyst.utils.load_dict(dict_decode_path)
    num_labels = len(dict_decode)
    
    # load smaller dataset to determine input size:
    data_val = np.load(val_paths[0])
    # expect shape (num_audiofiles, num_frames, num_features + label_column)
    # subtract the label column and add dimension for 'color scale' 
    input_shape = (data_val.shape[1], data_val.shape[2] - 1, 1) 
    # remove unneeded variable
    del data_val
    
    # setup model 
    envclassifier, settings_dict = pystmodels.cnn_classifier(
        input_shape = input_shape,
        num_labels = num_labels)
    
    # create callbacks variable if not in kwargs
    # allow users to use different callbacks if desired
    if 'callbacks' not in kwargs:
        callbacks = pystmodels.setup_callbacks(patience = patience,
                                                best_modelname = model_path, 
                                                log_filename = model_dir.joinpath('log.csv'))
    optimizer = 'adam'
    loss = 'sparse_categorical_crossentropy'
    metrics = ['accuracy']
    envclassifier.compile(optimizer = optimizer,
                          loss = loss,
                          metrics = metrics)

    # TODO remove?
    # save variables that are not too large:
    local_variables = locals()
    global_variables = globals()
    pyst.utils.save_dict(
        dict2save = local_variables, 
        filename = model_dir.joinpath('local_variables_{}.csv'.format(
                            model_name)),
        overwrite=True)
    pyst.utils.save_dict(
        dict2save = global_variables,
        filename = model_dir.joinpath('global_variables_{}.csv'.format(
                            model_name)),
        overwrite = True)
        
    # start training
    start = time.time()

    for i, train_path in enumerate(train_paths):
        if i == 0:
            if 'epochs' in kwargs:
                epochs = kwargs['epochs']
            else:
                epochs = 10 # default in Keras
            total_epochs = epochs * len(train_paths)
            print('\n\nThe model will be trained {} epochs per '.format(epochs)+\
                'training session. \nTotal possible epochs: {}\n\n'.format(total_epochs))
        start_session = time.time()
        data_train_path = train_path
        # just use first validation data file
        data_val_path = val_paths[0]
        # just use first test data file
        data_test_path = test_paths[0]
        
        print('\nTRAINING SESSION ',i+1)
        print("Training on: ")
        print(data_train_path)
        print()
        
        data_train = np.load(data_train_path)
        data_val = np.load(data_val_path)
        data_test = np.load(data_test_path)
        
        # reinitiate 'callbacks' for additional iterations
        if i > 0: 
            if 'callbacks' not in kwargs:
                callbacks = pystmodels.setup_callbacks(patience = patience,
                                                        best_modelname = model_path, 
                                                        log_filename = model_dir.joinpath('log.csv'))
            else:
                # apply callbacks set in **kwargs
                callbacks = kwargs['callbacks']

        if use_generator:
            train_generator = pystmodels.Generator(data_matrix1 = data_train, 
                                                    data_matrix2 = None,
                                                    normalized = normalized,
                                                    add_tensor_last = add_tensor_last)
            val_generator = pystmodels.Generator(data_matrix1 = data_val,
                                                data_matrix2 = None,
                                                normalized = normalized,
                                                    add_tensor_last = add_tensor_last)
            test_generator = pystmodels.Generator(data_matrix1 = data_test,
                                                  data_matrix2 = None,
                                                  normalized = normalized,
                                                    add_tensor_last = add_tensor_last)

            train_generator.generator()
            val_generator.generator()
            test_generator.generator()
            history = envclassifier.fit_generator(
                train_generator.generator(),
                steps_per_epoch = data_train.shape[0],
                callbacks = callbacks,
                validation_data = val_generator.generator(),
                validation_steps = data_val.shape[0],
                **kwargs)
            
            # TODO test how well prediction works. use simple predict instead?
            # need to define `y_test`
            X_test, y_test = pyst.feats.separate_dependent_var(data_test)
            y_predicted = envclassifier.predict_generator(
                test_generator.generator(),
                steps = data_test.shape[0])

        else:
            # TODO make scaling data optional?
            # data is separated and shaped for this classifier in scale_X_y..
            X_train, y_train, scalars = pyst.feats.scale_X_y(data_train,
                                                                is_train=True)
            X_val, y_val, __ = pyst.feats.scale_X_y(data_val,
                                                    is_train=False, 
                                                    scalars=scalars)
            X_test, y_test, __ = pyst.feats.scale_X_y(data_test,
                                                        is_train=False, 
                                                        scalars=scalars)
            
            history = envclassifier.fit(X_train, y_train, 
                                        callbacks = callbacks, 
                                        validation_data = (X_val, y_val),
                                        **kwargs)
            
            envclassifier.evaluate(X_test, y_test)
            y_predicted = envclassifier.predict(X_test)
            # which category did the model predict?
            
    
        y_pred = np.argmax(y_predicted, axis=1)
        if len(y_pred.shape) > len(y_test.shape):
            y_test = np.expand_dims(y_test, axis=1)
        elif len(y_pred.shape) < len(y_test.shape):
            y_pred = np.expand_dims(y_pred, axis=1)
        try:
            assert y_pred.shape == y_test.shape
        except AssertionError:
            raise ValueError('The shape of prediction data {}'.format(y_pred.shape) +\
                ' does not match the `y_test` dataset {}'.format(y_test.shape) +\
                    '\nThe shapes much match in order to measure accuracy.')
                
        match = sum(y_test == y_pred)
        if len(match.shape) == 1:
            match = match[0]
        test_accuracy = round(match/len(y_test),4)
        print('\nModel reached accuracy of {}%'.format(test_accuracy*100))
        
        end_session = time.time()
        total_dur_sec_session = round(end_session-start_session,2)
        model_features_dict = dict(model_path = model_path,
                                data_train_path = data_train_path,
                                data_val_path = data_val_path, 
                                data_test_path = data_test_path, 
                                total_dur_sec_session = total_dur_sec_session,
                                use_generator = use_generator,
                                kwargs = kwargs)
        model_features_dict.update(settings_dict)
        if i == len(train_paths)-1:
            end = time.time()
            total_duration_seconds = round(end-start,2)
            time_dict = dict(total_duration_seconds=total_duration_seconds)
            model_features_dict.update(time_dict)

        model_features_dict_path = model_dir.joinpath('info_{}_{}.csv'.format(
            model_name, i))
        model_features_dict_path = pyst.utils.save_dict(
            filename = model_features_dict_path,
            dict2save = model_features_dict)
    print('\nFinished training the model. The model and associated files can be '+\
        'found here: \n{}'.format(model_dir))
    
    return model_dir, history


def denoiser_run(model, new_audio, feat_settings_dict):
    '''Implements a pre-trained denoiser
    
    Parameters
    ----------
    model : str or pathlib.PosixPath
        The path to the denoising model.
    
    new_audio : str, pathlib.PosixPath, or np.ndarray
        The path to the noisy audiofile.
        
    feat_settings_dict : dict 
        Dictionary containing necessary settings for how the features were
        extracted for training the model. Expected keys: 'feature_type', 
        'win_size_ms', 'percent_overlap', 'sr', 'window', 'frames_per_sample',
        'input_shape', 'desired_shape', 'dur_sec', 'num_feats'.
        
    Returns
    -------
    cleaned_audio : np.ndarray [shape = (num_samples, )]
        The cleaned audio samples ready for playing or saving as audio file.
    sr : int 
        The sample rate of `cleaned_audio`.
        
    See Also
    --------
    pysoundtool.feats.get_feats
        How features are extracted.
        
    pysoundtool.feats.feats2audio
        How features are transformed back tino audio samples.
    '''
    
    feature_type = feat_settings_dict['feature_type']
    win_size_ms = feat_settings_dict['win_size_ms']
    sr = feat_settings_dict['sr']
    percent_overlap = feat_settings_dict['percent_overlap']
    try:
        window = feat_settings_dict['window']
    except KeyError:
        window = None
    frames_per_sample = feat_settings_dict['frames_per_sample']
    input_shape = feat_settings_dict['input_shape']
    dur_sec = feat_settings_dict['dur_sec']
    num_feats = feat_settings_dict['num_feats']
    desired_shape = feat_settings_dict['desired_shape']
    
    feats = pyst.feats.get_feats(new_audio, sr=sr, 
                                feature_type = feature_type,
                                win_size_ms = win_size_ms,
                                percent_overlap = percent_overlap,
                                window = window, 
                                dur_sec = dur_sec,
                                num_filters = num_feats)
    # are phase data still present? (only in stft features)
    if feats.dtype == np.complex and np.min(feats) < 0:
        original_phase = pyst.dsp.calc_phase(feats,
                                               radians=False)
    elif 'stft' in feature_type or 'powspec' in feature_type:
        feats_stft = pyst.feats.get_feats(new_audio, 
                                          sr=sr, 
                                          feature_type = 'stft',
                                          win_size_ms = win_size_ms,
                                          percent_overlap = percent_overlap,
                                          window = window, 
                                          dur_sec = dur_sec,
                                          num_filters = num_feats)
        original_phase = pyst.dsp.calc_phase(feats_stft,
                                               radians = False)
    else:
        original_phase = None
    
    if 'signal' in feature_type:
        feats_zeropadded = np.zeros(desired_shape)
        feats_zeropadded = feats_zeropadded.flatten()
        if len(feats.shape) > 1:
            feats_zeropadded = feats_zeropadded.reshape(feats_zeropadded.shape[0],
                                                        feats.shape[1])
        if len(feats) > len(feats_zeropadded):
            feats = feats[:len(feats_zeropadded)]
        feats_zeropadded[:len(feats)] += feats
        # reshape here to avoid memory issues if total # samples is large
        feats = feats_zeropadded.reshape(desired_shape)
    
    feats = pyst.feats.prep_new_audiofeats(feats,
                                           desired_shape,
                                           input_shape)
    # ensure same shape as feats
    if original_phase is not None:
        original_phase = pyst.feats.prep_new_audiofeats(original_phase,
                                                        desired_shape,
                                                        input_shape)
    
    
    feats_normed = pyst.feats.normalize(feats)
    denoiser = load_model(model)
    cleaned_feats = denoiser.predict(feats_normed, batch_size = frames_per_sample)
    
    # need to change shape back to 2D
    # current shape is (batch_size, num_frames, num_features, 1)
    # need (num_frames, num_features)

    # remove last tensor dimension
    if feats_normed.shape[-1] == 1:
        feats_normed = feats_normed.reshape(feats_normed.shape[:-1])
    feats_flattened = feats_normed.reshape(-1, 
                                            feats_normed.shape[-1])
    audio_shape = (feats_flattened.shape)
    
    cleaned_feats = cleaned_feats.reshape(audio_shape)
    if original_phase is not None:
        original_phase = original_phase.reshape(audio_shape)
    
    # now combine them to create audio samples:
    cleaned_audio = pyst.feats.feats2audio(cleaned_feats, 
                                           feature_type = feature_type,
                                           sr = sr, 
                                           win_size_ms = win_size_ms,
                                           percent_overlap = percent_overlap,
                                           phase = original_phase)
    if not isinstance(new_audio, np.ndarray):
        noisy_audio, __ = pyst.loadsound(new_audio, sr=sr)
    else:
        noisy_audio = new_audio
    if len(cleaned_audio) > len(noisy_audio):
        cleaned_audio = cleaned_audio[:len(noisy_audio)]
    
    max_energy_original = np.max(noisy_audio)
    # match the scale of the original audio:
    cleaned_audio = pyst.dsp.scalesound(cleaned_audio, max_val = max_energy_original)
    return cleaned_audio, sr

def collect_classifier_settings(feature_extraction_dir):
    # ensure feature_extraction_folder exists:
    dataset_path = pyst.utils.check_dir(feature_extraction_dir, make=False)
    
    # prepare features files to load for training
    features_files = list(dataset_path.glob('*.npy'))
    # NamedTuple: 'datasets.train', 'datasets.val', 'datasets.test'
    datasets = pyst.datasets.separate_train_val_test_files(
        features_files)
    # TODO test
    if not datasets.train:
        # perhaps data files located in subdirectories 
        features_files = list(dataset_path.glob('**/*.npy'))
        datasets = pyst.datasets.separate_train_val_test_files(
            features_files)
        if not datasets.train:
            raise FileNotFoundError('Could not locate train, validation, or test '+\
                '.npy files in the provided directory: \n{}'.format(dataset_path) +\
                    '\nThis program expects "train", "val", or "test" to be '+\
                        'included in each filename (not parent directory/ies) names.')
        
    train_paths = datasets.train
    val_paths = datasets.val 
    test_paths = datasets.test
    
    # need dictionary for decoding labels:
    dict_decode_path = dataset_path.joinpath('dict_decode.csv')
    if not os.path.exists(dict_decode_path):
        raise FileNotFoundError('Could not find {}.'.format(dict_decode_path))
    dict_decode = pyst.utils.load_dict(dict_decode_path)
    num_labels = len(dict_decode)
    
    settings_dict = pyst.utils.load_dict(
        dataset_path.joinpath('log_extraction_settings.csv'))
    num_feats = pyst.utils.restore_dictvalue(settings_dict['num_feats'])
    # should the shape include the label column or not?
    # currently not
    feat_shape = pyst.utils.restore_dictvalue(settings_dict['desired_shape'])
    feature_type = settings_dict['feat_type']
    return datasets, num_labels, feat_shape, num_feats, feature_type

def cnnlstm_train(feature_extraction_dir,
                  model_name = 'model_cnnlstm_classifier',
                  use_generator = True,
                  normalized = False,
                  patience = 15,
                  timesteps = 10,
                  context_window = 5,
                  colorscale = 1,
                  total_training_sessions = None,
                  add_tensor_last = False,
                  **kwargs):
    '''Many settings followed by the paper below.
    
    References
    ----------
    Kim, Myungjong & Cao, Beiming & An, Kwanghoon & Wang, Jun. (2018). Dysarthric Speech Recognition Using Convolutional LSTM Neural Network. 10.21437/interspeech.2018-2250.
    '''
    
    datasets, num_labels, feat_shape, num_feats, feature_type =\
        collect_classifier_settings(feature_extraction_dir)
    
    train_paths = datasets.train
    val_paths = datasets.val
    test_paths = datasets.test
    
    # Save model directory inside feature directory
    dataset_path = train_paths[0].parent
    if feature_type:
        model_name += '_'+feature_type + '_' + pyst.utils.get_date() 
    else:
        model_name += '_' + pyst.utils.get_date() 
    model_dir = dataset_path.joinpath(model_name)
    model_dir = pyst.utils.check_dir(model_dir, make=True)
    model_name += '.h5'
    model_path = model_dir.joinpath(model_name)
    
    frame_width = context_window * 2 + 1 # context window w central frame
    input_shape = (timesteps, frame_width, num_feats, colorscale)
    model, settings = pystmodels.cnnlstm_classifier(num_labels = num_labels, 
                                                    input_shape = input_shape, 
                                                    lstm_cells = num_feats)
    

    # create callbacks variable if not in kwargs
    # allow users to use different callbacks if desired
    if 'callbacks' not in kwargs:
        callbacks = pystmodels.setup_callbacks(patience = patience,
                                                best_modelname = model_path, 
                                                log_filename = model_dir.joinpath('log.csv'))
    optimizer = 'adam'
    loss = 'sparse_categorical_crossentropy'
    metrics = ['accuracy']
    model.compile(optimizer = optimizer,
                          loss = loss,
                          metrics = metrics)
    
    # update settings with optimizer etc.
    additional_settings = dict(optimizer = optimizer,
                               loss = loss,
                               metrics = metrics,
                               kwargs = kwargs)
    settings.update(additional_settings)
    
    
    # start training
    start = time.time()

    for i, train_path in enumerate(train_paths):
        if i == 0:
            if 'epochs' in kwargs:
                epochs = kwargs['epochs']
            else:
                epochs = 10 # default in Keras
            total_epochs = epochs * len(train_paths)
            print('\n\nThe model will be trained {} epochs per '.format(epochs)+\
                'training session. \nTotal possible epochs: {}\n\n'.format(total_epochs))
        start_session = time.time()
        data_train_path = train_path
        # just use first validation data file
        data_val_path = val_paths[0]
        # just use first test data file
        data_test_path = test_paths[0]
        
        print('\nTRAINING SESSION ',i+1)
        print("Training on: ")
        print(data_train_path)
        print()
        
        data_train = np.load(data_train_path)
        data_val = np.load(data_val_path)
        data_test = np.load(data_test_path)
        
        # shuffle data_train, just to ensure random
        np.random.shuffle(data_train) 
        
        # reinitiate 'callbacks' for additional iterations
        if i > 0: 
            if 'callbacks' not in kwargs:
                callbacks = pystmodels.setup_callbacks(patience = patience,
                                                        best_modelname = model_path, 
                                                        log_filename = model_dir.joinpath('log.csv'))
            else:
                # apply callbacks set in **kwargs
                callbacks = kwargs['callbacks']

        if use_generator:
            train_generator = pystmodels.Generator(data_matrix1 = data_train, 
                                                    data_matrix2 = None,
                                                    normalized = normalized,
                                                    adjust_shape = input_shape,
                                                    add_tensor_last = add_tensor_last)
            val_generator = pystmodels.Generator(data_matrix1 = data_val,
                                                data_matrix2 = None,
                                                normalized = normalized,
                                                adjust_shape = input_shape,
                                                    add_tensor_last = add_tensor_last)
            test_generator = pystmodels.Generator(data_matrix1 = data_test,
                                                  data_matrix2 = None,
                                                  normalized = normalized,
                                                  adjust_shape = input_shape,
                                                  add_tensor_last = add_tensor_last)

            train_generator.generator()
            val_generator.generator()
            test_generator.generator()
            history = model.fit_generator(
                train_generator.generator(),
                steps_per_epoch = data_train.shape[0],
                callbacks = callbacks,
                validation_data = val_generator.generator(),
                validation_steps = data_val.shape[0],
                **kwargs)
            
            # TODO test how well prediction works. use simple predict instead?
            # need to define `y_test`
            X_test, y_test = pyst.feats.separate_dependent_var(data_test)
            y_predicted = model.predict_generator(
                test_generator.generator(),
                steps = data_test.shape[0])

        else:
            # TODO make scaling data optional?
            # data is separated and shaped for this classifier in scale_X_y..
            X_train, y_train, scalars = pyst.feats.scale_X_y(data_train,
                                                                is_train=True)
            X_val, y_val, __ = pyst.feats.scale_X_y(data_val,
                                                    is_train=False, 
                                                    scalars=scalars)
            X_test, y_test, __ = pyst.feats.scale_X_y(data_test,
                                                        is_train=False, 
                                                        scalars=scalars)
            
            X_train = pyst.feats.adjust_shape(X_train, 
                                              (X_train.shape[0],)+input_shape,
                                              change_dims = True)
            
            X_val = pyst.feats.adjust_shape(X_val, 
                                            (X_val.shape[0],)+input_shape,
                                              change_dims = True)
            X_test = pyst.feats.adjust_shape(X_test, 
                                             (X_test.shape[0],)+input_shape,
                                              change_dims = True)
            
            # randomize train data
            rand_idx = np.random.choice(range(len(X_train)),
                                        len(X_train),
                                        replace=False)
            X_train = X_train[rand_idx]
            
            history = model.fit(X_train, y_train, 
                                        callbacks = callbacks, 
                                        validation_data = (X_val, y_val),
                                        **kwargs)
            
            model.evaluate(X_test, y_test)
            y_predicted = model.predict(X_test)
            # which category did the model predict?
            
    
        y_pred = np.argmax(y_predicted, axis=1)
        if len(y_pred.shape) > len(y_test.shape):
            y_test = np.expand_dims(y_test, axis=1)
        elif len(y_pred.shape) < len(y_test.shape):
            y_pred = np.expand_dims(y_pred, axis=1)
        try:
            assert y_pred.shape == y_test.shape
        except AssertionError:
            raise ValueError('The shape of prediction data {}'.format(y_pred.shape) +\
                ' does not match the `y_test` dataset {}'.format(y_test.shape) +\
                    '\nThe shapes much match in order to measure accuracy.')
                
        match = sum(y_test == y_pred)
        if len(match.shape) == 1:
            match = match[0]
        test_accuracy = round(match/len(y_test),4)
        print('\nModel reached accuracy of {}%'.format(test_accuracy*100))
        
        end_session = time.time()
        total_dur_sec_session = round(end_session-start_session,2)
        model_features_dict = dict(model_path = model_path,
                                data_train_path = data_train_path,
                                data_val_path = data_val_path, 
                                data_test_path = data_test_path, 
                                total_dur_sec_session = total_dur_sec_session,
                                use_generator = use_generator,
                                kwargs = kwargs)
        model_features_dict.update(settings)
        model_features_dict_path = model_dir.joinpath('info_{}_{}.csv'.format(
            model_name, i))
        model_features_dict_path = pyst.utils.save_dict(
            filename = model_features_dict_path,
            dict2save = model_features_dict)
        if total_training_sessions is None:
            total_training_sessions = len(train_paths)
        if i == total_training_sessions-1:
            end = time.time()
            total_duration_seconds = round(end-start,2)
            time_dict = dict(total_duration_seconds=total_duration_seconds)
            model_features_dict.update(time_dict)

            model_features_dict_path = model_dir.joinpath('info_{}_{}.csv'.format(
                model_name, i))
            model_features_dict_path = pyst.utils.save_dict(
                filename = model_features_dict_path,
                dict2save = model_features_dict,
                overwrite = True)
            print('\nFinished training the model. The model and associated files can be '+\
            'found here: \n{}'.format(model_dir))
            model.save(model_dir.joinpath('final_not_best_model.h5'))
            return model_dir, history

def resnet50_train(feature_extraction_dir,
                   model_name = 'model_resnet50_classifier',
                   use_generator = False,
                   normalized = False,
                   patience = 15,
                   colorscale = 3,
                   total_training_sessions = None,
                   add_tensor_last = None,
                   **kwargs):
    datasets, num_labels, feat_shape, num_feats, feature_type =\
        collect_classifier_settings(feature_extraction_dir)
    
    train_paths = datasets.train
    val_paths = datasets.val
    test_paths = datasets.test
    
    # Save model directory inside feature directory
    dataset_path = train_paths[0].parent
    if feature_type:
        model_name += '_'+feature_type + '_' + pyst.utils.get_date() 
    else:
        model_name += '_' + pyst.utils.get_date() 
    model_dir = dataset_path.joinpath(model_name)
    model_dir = pyst.utils.check_dir(model_dir, make=True)
    model_name += '.h5'
    model_path = model_dir.joinpath(model_name)
    
    input_shape = (feat_shape[0], num_feats, colorscale)
    model, settings = pystmodels.resnet50_classifier(num_labels = num_labels, 
                                                    input_shape = input_shape)

    # create callbacks variable if not in kwargs
    # allow users to use different callbacks if desired
    if 'callbacks' not in kwargs:
        callbacks = pystmodels.setup_callbacks(patience = patience,
                                                best_modelname = model_path, 
                                                log_filename = model_dir.joinpath('log.csv'))
    optimizer = Adam(lr=0.0001)
    loss='sparse_categorical_crossentropy'
    metrics = ['accuracy']
    model.compile(optimizer=optimizer, loss = loss, 
                metrics = metrics)
    
    # update settings with optimizer etc.
    additional_settings = dict(optimizer = optimizer,
                               loss = loss,
                               metrics = metrics,
                               kwargs = kwargs)
    settings.update(additional_settings)
    
    
    # start training
    start = time.time()

    for i, train_path in enumerate(train_paths):
        if i == 0:
            if 'epochs' in kwargs:
                epochs = kwargs['epochs']
            else:
                epochs = 10 # default in Keras
            total_epochs = epochs * len(train_paths)
            print('\n\nThe model will be trained {} epochs per '.format(epochs)+\
                'training session. \nTotal possible epochs: {}\n\n'.format(total_epochs))
        start_session = time.time()
        data_train_path = train_path
        # just use first validation data file
        data_val_path = val_paths[0]
        # just use first test data file
        data_test_path = test_paths[0]
        
        print('\nTRAINING SESSION ',i+1)
        print("Training on: ")
        print(data_train_path)
        print()
        
        data_train = np.load(data_train_path)
        data_val = np.load(data_val_path)
        data_test = np.load(data_test_path)
        
        # shuffle data_train, just to ensure random
        np.random.shuffle(data_train) 
        
        # reinitiate 'callbacks' for additional iterations
        if i > 0: 
            if 'callbacks' not in kwargs:
                callbacks = pystmodels.setup_callbacks(patience = patience,
                                                        best_modelname = model_path, 
                                                        log_filename = model_dir.joinpath('log.csv'))
            else:
                # apply callbacks set in **kwargs
                callbacks = kwargs['callbacks']

        if use_generator:
            train_generator = pystmodels.Generator(data_matrix1 = data_train, 
                                                    data_matrix2 = None,
                                                    normalized = normalized,
                                                    adjust_shape = input_shape,
                                                    add_tensor_last = add_tensor_last,
                                                    gray2color = True)
            val_generator = pystmodels.Generator(data_matrix1 = data_val,
                                                data_matrix2 = None,
                                                normalized = normalized,
                                                adjust_shape = input_shape,
                                                    add_tensor_last = add_tensor_last,
                                                    gray2color = True)
            test_generator = pystmodels.Generator(data_matrix1 = data_test,
                                                  data_matrix2 = None,
                                                  normalized = normalized,
                                                  adjust_shape = input_shape,
                                                  add_tensor_last = add_tensor_last,
                                                    gray2color = True)

            train_generator.generator()
            val_generator.generator()
            test_generator.generator()
            history = model.fit_generator(
                train_generator.generator(),
                steps_per_epoch = data_train.shape[0],
                callbacks = callbacks,
                validation_data = val_generator.generator(),
                validation_steps = data_val.shape[0],
                **kwargs)
            
            # TODO test how well prediction works. use simple predict instead?
            # need to define `y_test`
            X_test, y_test = pyst.feats.separate_dependent_var(data_test)
            y_predicted = model.predict_generator(
                test_generator.generator(),
                steps = data_test.shape[0])

        else:
            # TODO make scaling data optional?
            # data is separated and shaped for this classifier in scale_X_y..
            X_train, y_train, scalars = pyst.feats.scale_X_y(data_train,
                                                                is_train=True)
            X_val, y_val, __ = pyst.feats.scale_X_y(data_val,
                                                    is_train=False, 
                                                    scalars=scalars)
            X_test, y_test, __ = pyst.feats.scale_X_y(data_test,
                                                        is_train=False, 
                                                        scalars=scalars)
            
            print(X_train.shape)
            X_train = pyst.feats.adjust_shape(X_train, 
                                              (X_train.shape[0],)+input_shape,
                                              change_dims = True)
            print(X_train.shape)
            X_val = pyst.feats.adjust_shape(X_val, 
                                            (X_val.shape[0],)+input_shape,
                                              change_dims = True)
            X_test = pyst.feats.adjust_shape(X_test, 
                                             (X_test.shape[0],)+input_shape,
                                              change_dims = True)
            
            # randomize train data
            rand_idx = np.random.choice(range(len(X_train)),
                                        len(X_train),
                                        replace=False)
            X_train = X_train[rand_idx]
            
            # make grayscale to colorscale
            X_train = pyst.feats.grayscale2color(X_train, colorscale = 3)
            X_val = pyst.feats.grayscale2color(X_val, colorscale = 3)
            X_test = pyst.feats.grayscale2color(X_test, colorscale = 3)
            
            print(X_train.shape)
            
            history = model.fit(X_train, y_train, 
                                        callbacks = callbacks, 
                                        validation_data = (X_val, y_val),
                                        **kwargs)
            
            model.evaluate(X_test, y_test)
            y_predicted = model.predict(X_test)
            # which category did the model predict?
            
    
        y_pred = np.argmax(y_predicted, axis=1)
        if len(y_pred.shape) > len(y_test.shape):
            y_test = np.expand_dims(y_test, axis=1)
        elif len(y_pred.shape) < len(y_test.shape):
            y_pred = np.expand_dims(y_pred, axis=1)
        try:
            assert y_pred.shape == y_test.shape
        except AssertionError:
            raise ValueError('The shape of prediction data {}'.format(y_pred.shape) +\
                ' does not match the `y_test` dataset {}'.format(y_test.shape) +\
                    '\nThe shapes much match in order to measure accuracy.')
                
        match = sum(y_test == y_pred)
        if len(match.shape) == 1:
            match = match[0]
        test_accuracy = round(match/len(y_test),4)
        print('\nModel reached accuracy of {}%'.format(test_accuracy*100))
        
        end_session = time.time()
        total_dur_sec_session = round(end_session-start_session,2)
        model_features_dict = dict(model_path = model_path,
                                data_train_path = data_train_path,
                                data_val_path = data_val_path, 
                                data_test_path = data_test_path, 
                                total_dur_sec_session = total_dur_sec_session,
                                use_generator = use_generator,
                                kwargs = kwargs)
        model_features_dict.update(settings)
        model_features_dict_path = model_dir.joinpath('info_{}_{}.csv'.format(
            model_name, i))
        model_features_dict_path = pyst.utils.save_dict(
            filename = model_features_dict_path,
            dict2save = model_features_dict)
        if total_training_sessions is None:
            total_training_sessions = len(train_paths)
        if i == total_training_sessions-1:
            end = time.time()
            total_duration_seconds = round(end-start,2)
            time_dict = dict(total_duration_seconds=total_duration_seconds)
            model_features_dict.update(time_dict)

            model_features_dict_path = model_dir.joinpath('info_{}_{}.csv'.format(
                model_name, i))
            model_features_dict_path = pyst.utils.save_dict(
                filename = model_features_dict_path,
                dict2save = model_features_dict,
                overwrite = True)
            print('\nFinished training the model. The model and associated files can be '+\
            'found here: \n{}'.format(model_dir))
            model.save(model_dir.joinpath('final_not_best_model.h5'))
            return model_dir, history


def extract_voxforge():
    '''This is a function specifically suited for the VoxForge dataset
    '''
    pass
