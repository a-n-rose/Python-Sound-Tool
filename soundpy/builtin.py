'''The soundpy.builtin module includes more complex functions that pull from several
other functions to complete fairly complex tasks, such as dataset formatting, 
filtering signals, and extracting features for neural networks.
''' 
import time
import pathlib
import random
import numpy as np
import soundfile as sf
from scipy.io.wavfile import write

# in order to import soundpy
import os, sys
import inspect
currentdir = os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())))
packagedir = os.path.dirname(currentdir)
sys.path.insert(0, packagedir)
import soundpy as sp


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
                 remove_dc=True,
                 control_vol = False,
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
    remove_dc : bool
        It True, the DC bias ('direct current' bias) will be removed. In other words, the 
        mean amplitude will be made to equal 0.
    **kwargs : additional keyword arguments
        Keyword arguments for `soundpy.filters.WienerFilter` or 
        'soundpy.filters.BandSubtraction` (depending on `filter_type`).
    
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
    if sr is None:
        sr = 48000
    if 'wiener' in filter_type:
        if sr == 22050:
            import warnings
            warnings.warn('\n\nWARNING: sample rate 22050 may have some '+\
                'missing frames within the filtered signal. \nIf possible, '+\
                    'perhaps use 8000, 16000, 41000, or 48000 sample rate instead.\n')
        fil = sp.WienerFilter(sr = sr, **kwargs)
    elif 'band' in filter_type:
        # at this point, band spectral subtraction only works with 
        if sr != 48000:
            import warnings
            warnings.warn('\n\nWARNING: Band spectral subtraciton requires a sample rate'+\
                ' of 48 kHz. Sample rate adjusted from {} to 48000.\n'.format(sr))
            sr = 48000
        fil = sp.BandSubtraction(sr=48000,
                                   **kwargs)
    if visualize:
        frame_subtitle = 'frame size {}ms, window shift {}ms'.format(fil.frame_dur, int(fil.percent_overlap*fil.frame_dur))

    # load signal (to be filtered)
    if not isinstance(audiofile, np.ndarray):
        samples_orig, sr = sp.loadsound(audiofile, fil.sr, dur_sec=None,
                                          use_scipy=use_scipy, remove_dc=remove_dc)
    else:
        samples_orig, sr = audiofile, sr
        if remove_dc:
            samples_orig = sp.dsp.remove_dc_bias(samples_orig)
    if sr != fil.sr:
        samples_orig, sr = sp.dsp.resample_audio(samples_orig, 
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
            if remove_dc:
                samples_noise = sp.dsp.remove_dc_bias(samples_noise)
            if sr_noise != fil.sr:
                samples_noise, sr_noise = sp.dsp.resample_audio(samples_noise,
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
                samples_noise, sr_noise = sp.loadsound(noise_file, 
                                                         fil.sr, 
                                                         dur_sec=dur_sec,
                                                         use_scipy=use_scipy,
                                                         remove_dc = remove_dc)
                assert sr_noise == fil.sr
        if samples_noise is None and noise_power is None:
            raise TypeError('Expected one of the following: '+\
                '\ntype tuple containing (samples, samplerate) of noise data'+\
                    '\naudiofile pathway to noise file'+\
                        '\n.npy file with powerspectrum values for noise'+\
                            '\n\nDid not expect {} as input.'.format(noise_file))

    else:
        starting_noise_len = sp.dsp.calc_frame_length(fil.sr, 
                                                         duration_noise_ms)
        samples_noise = samples_orig[:starting_noise_len]
    # if noise samples have been collected...
    # TODO improve snr / volume measurements
    if samples_noise is not None:
        # set how many subframes are needed to process entire noise signal
        fil.set_num_subframes(len(samples_noise), is_noise=True, zeropad=fil.zeropad)
    if visualize:
        sp.feats.plot(samples_orig, 'signal', title='Signal to filter'.upper(),sr = fil.sr)
        sp.feats.plot(samples_noise, 'signal', title= 'Noise samples to filter out'.upper(), sr=fil.sr)
    # prepare noise power matrix (if it's not loaded already)
    if fil.noise_subframes:
        if real_signal:
            #only the first half of fft (+1)
            total_rows = fil.num_fft_bins//2+1
        else:
            total_rows = fil.num_fft_bins
        noise_power = sp.dsp.create_empty_matrix((total_rows,))
        section = 0
        for frame in range(fil.noise_subframes):
            noise_section = samples_noise[section:section+fil.frame_length]
            noise_w_win = sp.dsp.apply_window(noise_section, fil.get_window(),
                                                zeropad=fil.zeropad)
            noise_fft = sp.dsp.calc_fft(noise_w_win, real_signal=real_signal)
            noise_power_frame = sp.dsp.calc_power(noise_fft)
            noise_power += noise_power_frame
            section += fil.overlap_length
        # welch's method: take average of power that has been collected
        # in windows
        noise_power = sp.dsp.calc_average_power(noise_power, 
                                                   fil.noise_subframes)
        assert section == fil.noise_subframes * fil.overlap_length
    if visualize:
        sp.feats.plot(noise_power, 'stft',title='Average noise power spectrum'.upper()+'\n{}'.format(frame_subtitle), energy_scale='power_to_db',
                             sr = fil.sr)
    
    # prepare target power matrix
    increment_length = int(fil.frame_length * fil.percent_overlap)
    total_rows =  increment_length + increment_length * fil.target_subframes
    filtered_sig = sp.dsp.create_empty_matrix(
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
            target_w_window = sp.dsp.apply_window(target_section,
                                                    fil.get_window(), 
                                                    zeropad=fil.zeropad)
            if visualize and frame % visualize_every_n_windows == 0:
                sp.feats.plot(target_section,'signal', title='Signal'.upper()+' \nframe {}: {}'.format( frame+1,frame_subtitle),sr = fil.sr)
                sp.feats.plot(target_w_window,'signal', title='Signal with {} window'.format(fil.window_type).upper()+'\nframe {}: {}'.format( frame+1,frame_subtitle),sr = fil.sr)
            target_fft = sp.dsp.calc_fft(target_w_window, real_signal=real_signal)
            target_power = sp.dsp.calc_power(target_fft)
            # now start filtering!!
            # initialize SNR matrix
            if visualize and frame % visualize_every_n_windows == 0:
                sp.feats.plot(target_power,'stft', title='Signal power spectrum'.upper()+'\nframe {}: {}'.format( frame+1,frame_subtitle), energy_scale='power_to_db', sr = fil.sr)
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
                target_phase = sp.dsp.calc_phase(target_fft, 
                                                   radians=phase_radians)
                enhanced_fft = fil.apply_bandspecsub(target_power, 
                                                      target_phase, 
                                                      noise_power)
                if apply_postfilter:
                    enhanced_fft = fil.apply_postfilter(enhanced_fft,
                                                        target_fft,
                                                        target_power,
                                                        noise_power)
            
            enhanced_ifft = sp.dsp.calc_ifft(enhanced_fft, 
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
                sp.feats.plot(filtered_sig,'signal', title='Filtered signal'.upper()+'\nup to frame {}: {}'.format(frame+1,frame_subtitle), sr = fil.sr)
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
        sp.feats.plot(np.abs(filtered_sig.reshape((
                rows,cols,)))**2,
            'stft', title='Final filtered signal power spectrum'.upper()+'\n{}: {}'.format(filter_type,frame_subtitle), energy_scale='power_to_db')
        sp.feats.plot(enhanced_signal,'signal', title='Final filtered signal'.upper()+'\n{}'.format(filter_type), sr = fil.sr)
    if control_vol:
        enhanced_signal = fil.check_volume(enhanced_signal)
    if len(enhanced_signal) > len(samples_orig):
        enhanced_signal = enhanced_signal[:len(samples_orig)]
    # for backwards compatibility
    if output_filename is not None or save2wav:
        if output_filename is None:
            output_filename = sp.utils.get_date()+'.wav'
        saved_filename = sp.savesound(str(output_filename), 
                                        enhanced_signal,
                                        sr=fil.sr,
                                        overwrite=overwrite,
                                        remove_dc=remove_dc)
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
    >>> import soundpy as sp
    >>> audio_info = sp.builtin.dataset_logger()
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
    audiofile_dir = sp.utils.check_dir(audiofile_dir)
    
    audiofiles = sp.files.collect_audiofiles(audiofile_dir,
                                               recursive = recursive)
    
    audiofile_dict = dict()
    
    for i, audio in enumerate(audiofiles):
        # set sr to None to get audio file's sr
        # set mono to False to see if mono or stereo sound 
        y, sr = sp.loadsound(audio, sr=None, mono=False)
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
        sp.utils.print_progress(i, len(audiofiles), task='logging audio file details')
        
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
        tested for stereo sound and soundpy. (default False)
        
    Returns
    -------
    directory : pathlib.PosixPath
        The directory where the formatted audio files are located.
    
    See Also
    --------
    soundpy.files.collect_audiofiles
        Collects audiofiles from a given directory.
        
    soundpy.files.conversion_formats
        The available formats for converting audio data.
        
    soundfile.available_subtypes
        The subtypes or bitdepth possible for soundfile
    '''
    if new_dir is None and not overwrite:
        new_dir = 'audiofile_reformat_'+sp.utils.get_date()
        import warnings
        message = '\n\nATTENTION: Due to the risk of corrupting existing datasets, '+\
            'reformated audio will be saved in the following directory: '+\
                '\n{}\n'.format(new_dir)
        warnings.warn(message)
        
    # ensure new dir exists, and if not make it
    if new_dir is not None:
        new_dir = sp.utils.check_dir(new_dir, make=True)
        
    if audiodirectory is None:
        audiodirectory = './'
    # ensure audiodirectory exists
    audiodirectory = sp.utils.check_dir(audiodirectory, make=False)
    audiofiles = sp.files.collect_audiofiles(audiodirectory,
                                               recursive=recursive)
    audiodir_parent = audiodirectory.stem
    # add this base directory to 'new_dir'
    new_dir = new_dir.joinpath(audiodir_parent)
    new_dir = sp.utils.check_dir(new_dir,make=True)

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
        y, sr2 = sp.loadsound(audio,
                               sr=sr, 
                               dur_sec = dur_sec,
                               mono = mono)
        # ensure the sr matches what was set
        if sr is not None:
            assert sr2 == sr
        
        if zeropad and dur_sec:
            goal_num_samples = int(dur_sec*sr2)
            y = sp.dsp.set_signal_length(y, goal_num_samples)
            
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
            new_filename = sp.files.replace_ext(new_filename, format.lower())
            
        try:
            new_filename = sp.savesound(new_filename, y, sr2, 
                                      overwrite=overwrite,
                                      format=format,subtype=bd)
        except FileExistsError:
            print('File {} already exists.'.format(new_filename))
            
        sp.utils.print_progress(i, len(audiofiles), 
                                  task = 'reformatting dataset')
        
    if new_dir:
        return new_dir
    else:
        return audiodirectory
 

# TODO speed this up, e.g. preload noise data?
def create_denoise_data(cleandata_dir, noisedata_dir, trainingdata_dir, limit=None,
                            snr_levels = None, pad_mainsound_sec = None, 
                            random_seed = None, overwrite = False, **kwargs):
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
    pad_mainsound_sec : int, float, optional
        Amount in seconds the main sound should be padded. In other words, in seconds how
        long the background sound should play before the clean / main / target audio starts. 
        The same amount of noise will be appended at the end.
        (default None)
    random_seed : int 
        A value to allow random order of audiofiles to be predictable. 
        (default None). If None, the order of audiofiles will not be predictable.
    overwrite : bool 
        If True, a new dataset will be created regardless of whether or not a matching 
        directory already exists. (default False)
    **kwargs : additional keyword arguments
        The keyword arguments for soundpy.files.loadsound
        

    Returns
    -------
    saveinput_path : pathlib.PosixPath
        Path to where noisy audio files are located
    saveoutput_path : pathlib.PosixPath   
        Path to where clean audio files are located
        
    See Also
    --------
    soundpy.files.loadsound
        Loads audiofiles.
    
    soundpy.dsp.add_backgroundsound
        Add background sound / noise to signal at a determined signal-to-noise ratio.
    '''
    import math
    import time
    
    start = time.time()

    # check to ensure clean and noisy data are there
    # and turn them into pathlib.PosixPath objects:
    cleandata_dir = sp.utils.check_dir(cleandata_dir, make=False)
    noisedata_dir = sp.utils.check_dir(noisedata_dir, make=False)
    trainingdata_dir = sp.utils.string2pathlib(trainingdata_dir)
    
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
            newdata_clean_dir = sp.utils.check_dir(newdata_clean_dir, make=False,
                                                     append = False)
            newdata_noisy_dir = sp.utils.check_dir(newdata_noisy_dir, make=False,
                                                     append = False)
        except FileExistsError:
            raise FileExistsError('Datasets already exist at this location. Set '+\
                '`overwrite` to True or designate a new directory.')
        except FileNotFoundError:
            pass
    
    # create directory to save new data (if not exist)
    newdata_clean_dir = sp.utils.check_dir(newdata_clean_dir, make = True)
    newdata_noisy_dir = sp.utils.check_dir(newdata_noisy_dir, make = True)
   
    # collect audiofiles (not limited to .wav files)
    cleanaudio = sorted(sp.files.collect_audiofiles(cleandata_dir,
                                                      hidden_files = False,
                                                      wav_only = False,
                                                      recursive = False))
    noiseaudio = sorted(sp.files.collect_audiofiles(noisedata_dir,
                                                      hidden_files = False,
                                                      wav_only = False,
                                                      recursive = False))
    
    if random_seed is not None:
        random.seed(random_seed)
    random.shuffle(cleanaudio)
    
    if limit is not None:
        cleanaudio = cleanaudio[:limit]
    
    # ensure snr_levels is array-like 
    if snr_levels is not None:
        if not isinstance(snr_levels, list) and not isinstance(snr_levels, np.ndarray):
            snr_levels = list(snr_levels)
    
    for i, wavefile in enumerate(cleanaudio):
        sp.utils.print_progress(iteration=i, 
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
        if 'sr' not in kwargs:
            # set at high sr for measuring noise in signals, 
            # necessary in applying noise at specific SNR level
            kwargs['sr'] = 44100
        else: 
            if int(kwargs['sr']) < 44100:
                import warnings
                msg = 'The measuring of signal to noise ratio is '+\
                    'improved if the sample rate is at or above 44100 Hz.'+\
                        '\nConsider changing sr = {} to sr = 44100.'.format(
                            kwargs['sr'])
                warnings.warn(msg)
        clean_data, sr = sp.loadsound(wavefile, **kwargs)
        noise_data, sr = sp.loadsound(noise, **kwargs)
        
        # makes adding of sounds smoother:
        clean_data = sp.dsp.remove_dc_bias(clean_data)
        noise_data = sp.dsp.remove_dc_bias(noise_data)
        
        # incase any weird clicks at beginning / end of signals:
        clean_data = sp.dsp.clip_at_zero(clean_data, samp_win = 10)
        noise_data = sp.dsp.clip_at_zero(noise_data, samp_win = 10)
        
        noisy_data, snr_appx = sp.dsp.add_backgroundsound(
            audio_main = clean_data, 
            audio_background = noise_data, 
            snr = snr, 
            pad_mainsound_sec = pad_mainsound_sec, 
            wrap = False,
            **kwargs)
        if pad_mainsound_sec:
            # pad clean the same way as noisy so they are the same length
            num_pad_samps = sp.dsp.calc_frame_length(pad_mainsound_sec * 1000, sr)
            padding = np.zeros(num_pad_samps)
            clean_data = np.concatenate((padding, clean_data, padding))
        
        # ensure length of clean and noisy data match
        assert len(clean_data) == len(noisy_data)
        
        # ensure both noisy and clean files have same beginning to filename (i.e. clean filename)
        noisydata_filename = newdata_noisy_dir.joinpath(clean_stem+'_'+noise_stem\
            +'_snr'+str(snr)+'.wav')
        cleandata_filename = newdata_clean_dir.joinpath(clean_stem+'.wav')  
        
        write(noisydata_filename, sr_down, noisy_data)
        write(cleandata_filename, sr_down, clean_data)

    end = time.time()
    total_time, units = sp.utils.adjust_time_units(end-start)
    print('Data creation took a total of {} {}.'.format(
        round(total_time, 2), 
        units))

    return newdata_noisy_dir, newdata_clean_dir

def envclassifier_feats(
    data_dir,
    data_features_dir = None,
    perc_train = 0.8,
    ignore_label_marker = None,
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
        
    perc_train : float 
        The amount of data to be set aside for train data. The rest will be divided into
        validation and test datasets.
        
    ignore_label_marker : str 
        A string to look for in the labels if the "label" should not be included.
        For example, '__' to ignore a subdirectory titled "__noise" or "not__label".
    
    kwargs : additional keyword arguments
        Keyword arguments for `soundpy.feats.save_features_datasets` and 
        `soundpy.feats.get_feats`.
        
    Returns
    -------
    feat_extraction_dir : pathlib.PosixPath
        The pathway to where all feature extraction files can be found, including datasets.
        
    See Also
    --------
    soundpy.feats.get_feats
        Extract features from audio file or audio data.
        
    soundpy.feats.save_features_datasets
        Preparation of acoustic features in train, validation and test datasets.
    '''
    if data_features_dir is None:
        data_features_dir = './audiodata/example_feats_models/envclassifier/'

    feat_extraction_dir = 'features_' + sp.utils.get_date()

    # collect labels 
    labels = []
    data_dir = sp.utils.string2pathlib(data_dir)
    for label in data_dir.glob('*/'):
        if label.suffix:
            # avoid adding unwanted files in the directory
            # want only directory names
            continue
        if ignore_label_marker is not None:
            if ignore_label_marker in label.stem:
                continue
        # ignores hidden directories
        if label.stem[0] == '.':
            continue
        labels.append(label.stem)
    labels = set(labels)
    if len(labels) == 0:
        raise ValueError('No subdirectories found to offer as labels. Ensure this path'+\
            ' contains the data:\n{}'.format(data_dir))

    # create paths for what we need to save:
    data_features_dir = sp.utils.check_dir(data_features_dir)
    feat_extraction_dir = data_features_dir.joinpath(feat_extraction_dir)
    feat_extraction_dir = sp.utils.check_dir(feat_extraction_dir, make=True)

    # dictionaries containing encoding and decoding labels:
    dict_encode_path = feat_extraction_dir.joinpath('dict_encode.csv')
    dict_decode_path = feat_extraction_dir.joinpath('dict_decode.csv')
    # dictionary for which audio paths are assigned to which labels:
    dict_encdodedlabel2audio_path = feat_extraction_dir.joinpath('dict_encdodedlabel2audio.csv')
    
    # designate where to save train, val, and test data
    data_train_path = feat_extraction_dir.joinpath('{}_data.npy'.format('train'))
    data_val_path = feat_extraction_dir.joinpath('{}_data.npy'.format('val'))
    data_test_path = feat_extraction_dir.joinpath('{}_data.npy'.format('test'))

    # create and save encoding/decoding labels dicts
    dict_encode, dict_decode = sp.datasets.create_dicts_labelsencoded(labels)
    dict_encode_path = sp.utils.save_dict(
        filename = dict_encode_path,
        dict2save = dict_encode, 
        overwrite=False)
    dict_decode_path = sp.utils.save_dict(
        filename = dict_decode_path,
        dict2save = dict_encode, 
        overwrite=False)

    # save audio paths to each label in dict 
    paths_list = sp.files.collect_audiofiles(data_dir, recursive=True)
    paths_list = sorted(paths_list)

    dict_encodedlabel2audio = sp.datasets.create_encodedlabel2audio_dict(dict_encode,
                                                        paths_list)
    dict_encdodedlabel2audio_path = sp.utils.save_dict(
        dict2save = dict_encodedlabel2audio, 
        filename = dict_encdodedlabel2audio_path, 
        overwrite = False)
    # assign audiofiles into train, validation, and test datasets
    train, val, test = sp.datasets.audio2datasets(
        dict_encdodedlabel2audio_path,
        perc_train = perc_train,
        limit = None,
        seed = 40)

    # save audiofiles for each dataset to dict and save
    dataset_dict = dict([('train',train),('val', val),('test',test)])
    dataset_dict_path = feat_extraction_dir.joinpath('dataset_audiofiles.csv')
    dataset_dict_path = sp.utils.save_dict(
        dict2save = dataset_dict, 
        filename = dataset_dict_path, 
        overwrite=True)
    # save paths to where extracted features of each dataset will be saved to dict w same keys
    datasets_path2save_dict = dict([('train',data_train_path),
                                    ('val', data_val_path),
                                    ('test',data_test_path)])

    # extract features

    start = time.time()

    dataset_dict, datasets_path2save_dict = sp.feats.save_features_datasets(
        datasets_dict = dataset_dict,
        datasets_path2save_dict = datasets_path2save_dict,
        labeled_data = True,
        decode_dict = dict_decode,
        **kwargs)

    end = time.time()
    
    total_dur_sec = end-start
    total_dur, units = sp.utils.adjust_time_units(total_dur_sec)
    print('\nFinished! Total duration: {} {}.'.format(round(total_dur,2), units))

    # save which audiofiles were extracted for each dataset
    # save where extracted data were saved
    # save how long feature extraction took
    dataprep_settings = dict(dataset_dict = dataset_dict,
                            datasets_path2save_dict = datasets_path2save_dict,
                            total_dur_sec = total_dur_sec)
    dataprep_settings_path = sp.utils.save_dict(
        dict2save = dataprep_settings,
        filename = feat_extraction_dir.joinpath('dataset_audio_assignments.csv'))
    
    return feat_extraction_dir

def denoiser_feats(
    data_clean_dir,
    data_noisy_dir,
    data_features_dir = None,
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
        
    limit : int, optional
        The limit of audio files for feature extraction. (default None)
    
    kwargs : additional keyword arguments
        Keyword arguments for `soundpy.feats.save_features_datasets` and 
        `soundpy.feats.get_feats`.
        
    Returns
    -------
    feat_extraction_dir : pathlib.PosixPath
        The pathway to where all feature extraction files can be found, including datasets.
        
    See Also
    --------
    soundpy.datasets.create_denoise_data
        Applies noise at specified SNR levels to clean audio files.
    
    soundpy.feats.get_feats
        Extract features from audio file or audio data.
        
    soundpy.feats.save_features_datasets
        Preparation of acoustic features in train, validation and test datasets.
    '''
    if data_features_dir is None:
        data_features_dir = './audiodata/example_feats_models/denoiser/'
    
    # create unique directory for feature extraction session:
    feat_extraction_dir = 'features_' + sp.utils.get_date()

    # 1) Ensure clean and noisy data directories exist
    audio_clean_path = sp.utils.check_dir(data_clean_dir, make=False)
    audio_noisy_path = sp.utils.check_dir(data_noisy_dir, make=False)

    # 2) create paths for what we need to save:
    denoise_data_path = sp.utils.check_dir(data_features_dir, make=True)
    feat_extraction_dir = denoise_data_path.joinpath(feat_extraction_dir)
    feat_extraction_dir = sp.utils.check_dir(feat_extraction_dir, make=True)
    # Noisy and clean train, val, and test data paths:
    data_train_noisy_path = feat_extraction_dir.joinpath('{}_data_{}.npy'.format('train',
                                                                            'noisy'))
    data_val_noisy_path = feat_extraction_dir.joinpath('{}_data_{}.npy'.format('val',
                                                                        'noisy'))
    data_test_noisy_path = feat_extraction_dir.joinpath('{}_data_{}.npy'.format('test',
                                                                        'noisy'))
    data_train_clean_path = feat_extraction_dir.joinpath('{}_data_{}.npy'.format('train',
                                                                            'clean'))
    data_val_clean_path = feat_extraction_dir.joinpath('{}_data_{}.npy'.format('val',
                                                                        'clean'))
    data_test_clean_path = feat_extraction_dir.joinpath('{}_data_{}.npy'.format('test',
                                                                        'clean'))

    # 3) collect audiofiles and divide them into train, val, and test datasets
    # noisy data
    noisyaudio = sp.files.collect_audiofiles(audio_noisy_path, 
                                                    hidden_files = False,
                                                    wav_only = False,
                                                    recursive = False)
    # sort audio (can compare if noisy and clean datasets are compatible)
    noisyaudio = sorted(noisyaudio)
    if limit is not None:
        noisyaudio =  noisyaudio[:limit]

    # clean data
    cleanaudio = sp.files.collect_audiofiles(audio_clean_path, 
                                                    hidden_files = False,
                                                    wav_only = False,
                                                    recursive = False)
    cleanaudio = sorted(cleanaudio)
    if limit is not None:
        cleanaudio =  cleanaudio[:limit]


    # check if they match up: (expects clean file name to be in noisy file name)
    for i, audiofile in enumerate(noisyaudio):
        if not sp.utils.check_noisy_clean_match(audiofile, cleanaudio[i]):
            raise ValueError('The noisy and clean audio datasets do not appear to match.')

    # save collected audiofiles for noisy and clean datasets to dictionary
    noisy_audio_dict = dict([('noisy', noisyaudio)])
    clean_audio_dict = dict([('clean', cleanaudio)])
    
    noisy_audio_dict_path = feat_extraction_dir.joinpath('noisy_audio.csv')
    noisy_audio_dict_path = sp.utils.save_dict(
        dict2save = noisy_audio_dict, 
        filename = noisy_audio_dict_path,
        overwrite=False)
    clean_audio_dict_path = feat_extraction_dir.joinpath('clean_audio.csv')
    clean_audio_dict_path = sp.utils.save_dict(
        dict2save = clean_audio_dict, 
        filename = clean_audio_dict_path,
        overwrite=False)
    # separate into datasets
    train_noisy, val_noisy, test_noisy = sp.datasets.audio2datasets(
        noisy_audio_dict_path, perc_train = perc_train, seed=40)
    train_clean, val_clean, test_clean = sp.datasets.audio2datasets(
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
    path2_noisy_datasets = sp.utils.save_dict(
        dict2save = dataset_dict_noisy,
        filename = path2_noisy_datasets,
        overwrite=False)
    path2_clean_datasets = sp.utils.save_dict(
        dict2save = dataset_dict_clean,
        filename = path2_clean_datasets,
        overwrite=False)

    # 5) extract features
            
    # ensure the noisy and clean values match up:
    for key, value in dataset_dict_noisy.items():
        for j, audiofile in enumerate(value):
            if not sp.utils.check_noisy_clean_match(audiofile,
                                                    dataset_dict_clean[key][j]):
                raise ValueError('There is a mismatch between noisy and clean audio. '+\
                    '\nThe noisy file:\n{}'.format(dataset_dict_noisy[key][i])+\
                        '\ndoes not seem to match the clean file:\n{}'.format(audiofile))

    start = time.time()

    # first clean data
    dataset_dict_clean, dataset_paths_clean_dict = sp.feats.save_features_datasets(
        datasets_dict = dataset_dict_clean,
        datasets_path2save_dict = dataset_paths_clean_dict,
        **kwargs)
        
    # then noisy data
    dataset_dict_noisy, dataset_paths_noisy_dict = sp.feats.save_features_datasets(
        datasets_dict = dataset_dict_noisy,
        datasets_path2save_dict = dataset_paths_noisy_dict,
        **kwargs)

    end = time.time()

    total_dur_sec = round(end-start,2)
    total_dur, units = sp.utils.adjust_time_units(total_dur_sec)
    print('\nFinished! Total duration: {} {}.'.format(total_dur, units))
    # save which audiofiles were extracted for each dataset
    # save where extracted data were saved
    # save total duration of feature extraction
    dataprep_settings = dict(dataset_dict_noisy = dataset_dict_noisy,
                            dataset_paths_noisy_dict = dataset_paths_noisy_dict,
                            dataset_dict_clean = dataset_dict_clean,
                            dataset_paths_clean_dict = dataset_paths_clean_dict,
                            total_dur_sec = total_dur_sec,
                            limit = limit, 
                            perc_train = perc_train)
    dataprep_settings_path = sp.utils.save_dict(
        dict2save = dataprep_settings,
        filename = feat_extraction_dir.joinpath('dataset_audio_assignments.csv'))
    return feat_extraction_dir
