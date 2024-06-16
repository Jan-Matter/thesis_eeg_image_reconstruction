from abc import ABC, abstractmethod
import numpy as np
import pywt
import copy

class Transform(ABC):

    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def transform(self, data):
        pass

class FourierTransform(Transform):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def transform(self, data):
        time_series_data = data["time_series"]
        fourier_transform = {}
        for freq_band, time_series in time_series_data.items():
            fourier_transform[freq_band] = self.__fourier_transform(time_series)
        data["fourier_transform"] = fourier_transform
        return data

    def __fourier_transform(self, data):
        return np.fft.fft(data, axis=-1)
    

class MorletWaveletTransform(Transform):

    def __init__(self, wavelet_scales, **kwargs):
        super().__init__(**kwargs)
        self.__wavelet_scales = wavelet_scales

    def transform(self, data):
        time_series_data = data["time_series"]
        morelet_wavelet_transform = {}
        for freq_band, time_series in time_series_data.items():
            morelet_wavelet_transform[freq_band] = self.__morelet_wavelet_transform(time_series)
        data["morlet_wavelet_transform"] = morelet_wavelet_transform
        return data

    def __morelet_wavelet_transform(self, data):
        return pywt.cwt(data, self.__wavelet_scales, 'morl')[0][0]



