import numpy as np
from scipy.fft import fft
from pyscripts.Transformer import LogScaler

def b_fft(x, it):  # fast fourier transform
    b = fft(x)  # If X is a matrix, then fft(X) treats the columns of X as vectors and returns the Fourier transform of each column.
    b_real = b.real
    b_real = np.where(b_real < 0, b_real, 0)

    b_imag = b.imag
    b_imag = np.where(b_imag < 0, b_imag, 0)

    scale_fft = LogScaler()  # Log normalize and scale data
    b_real_norm = scale_fft.fit_transform(np.abs(b_real + 1))
    b_imag_norm = scale_fft.fit_transform(np.abs(b_imag + 1))

    x_fft_real = it.transform(b_real_norm)
    x_fft_imag = it.transform(b_imag_norm)

    return x_fft_real, x_fft_imag
