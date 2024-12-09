# -*- coding: utf-8 -*-
"""
Created on Sat Nov  5 21:05:28 2022

@author: jordan
"""

# READING WAV DATA

# METHOD 1
import soundfile as sf
data, Fs = sf.read('onscreen.wav')
data = data.astype('float')

# METHOD 2
import numpy as np
import scipy
Fs2, data2 = scipy.io.wavfile.read('../../../../../Downloads/onscreen.wav')
scale = np.iinfo(data2.dtype).max + 1
data2 = data2.astype('float64') / scale


# PLAYING DATA
import sounddevice as sd
sd.play(data,Fs)
sd.wait()    


