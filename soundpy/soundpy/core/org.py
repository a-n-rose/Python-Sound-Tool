import warnings
import pathlib
from typing import List, Any, Tuple, Callable, Optional
from pydantic import BaseModel, field_validator
import random



class FFTsettings(BaseModel):
    win_size_ms: Optional[int]=50
    percent_overlap: Optional[float]=0
    real_signal: Optional[bool]=False
    fft_bins: int
    window: Optional[str]="hann" # TODO add validator

class VADsettings(BaseModel):
    """Store voice detection settings
    """
    energy_thresh: Optional[int]=40
    freq_thresh: Optional[int]=185
    sfm_thresh: Optional[int]=5
    # min_energy: Optional[float]=None
    # min_freq: Optional[float]=None
    # min_sfm: Optional[float]=None
    use_beg_ms: Optional[int]=120

class Settings(BaseModel):
    """Handles voice detection variables
    """
    fft: Optional[dict | FFTsettings]
    vad: Optional[dict | VADsettings]

    @field_validator('fft', 'vad')
    @classmethod
    def convert_to_dict(cls, v: dict | FFTsettings | VADsettings,) -> dict:
        if not isinstance(v, dict):
            return v.__dict__
        return v

class SoundObj(BaseModel):
    id: int=random.randint(100000000000,199999999999)
    data: Any 
    sr: int