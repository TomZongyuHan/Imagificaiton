import numpy as np
from scipy.fft import fft
from Transformer import LogScaler
import os
import sys


def b_fft(x,it):
    b = fft(x)
    b_real = b.real
    b_real = np.where(b_real<0,b_real,0)
    
    b_imag = b.imag
    b_imag = np.where(b_imag<0,b_imag,0)
    
    
    scale_fft = LogScaler()
    b_real_norm = scale_fft.fit_transform(np.abs(b_real+1))
    b_imag_norm = scale_fft.fit_transform(np.abs(b_imag+1))

    x_fft_real = it.transform(b_real_norm)
    x_fft_imag = it.transform(b_imag_norm)
    
    return x_fft_real,x_fft_imag


class Logger(object):
    def __init__(self, file_name="Terminal.log", stream=sys.stdout):
        self.terminal = stream
        self.log = open(file_name, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

