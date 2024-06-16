import os
import numpy as np
import sys
from pathlib import Path
from dotenv import load_dotenv, find_dotenv

sys.path.append(str(Path(__file__).parent.parent.parent))
load_dotenv(find_dotenv())

class EEGPreprocessor:

    def __init__(self):
        self.__channel_selection = None
        self.__epoching = None
        self.__denoising = None
        self.__frequency_filtering = None
        self.__fourier_transform = None
        self.__wavelet_transform = None

    @property
    def channel_selection(self):
        return self.__channel_selection

    @channel_selection.setter
    def channel_selection(self, channel_selection):
        self.__channel_selection = channel_selection

    @property
    def epoching(self):
        return self.__epoching
    
    @epoching.setter
    def epoching(self, epoching):
        self.__epoching = epoching
        
    @property
    def denoising(self):
        return self.__denoising
    
    @denoising.setter
    def denoising(self, denoising):
        self.__denoising = denoising
        
    @property
    def frequency_filtering(self):
        return self.__frequency_filtering
    
    @frequency_filtering.setter
    def frequency_filtering(self, frequency_filtering):
        self.__frequency_filtering = frequency_filtering
    
    @property
    def fourier_transform(self):
        return self.__fourier_transform
    
    @fourier_transform.setter
    def fourier_transform(self, fourier_transform):
        self.__fourier_transform = fourier_transform

    @property
    def wavelet_transform(self):
        return self.__wavelet_transform
    
    @wavelet_transform.setter
    def wavelet_transform(self, wavelet_transform):
        self.__wavelet_transform = wavelet_transform


    def preprocess(self, data, subj, split, max_img_rep=20, seed=20200220):
        # Channel selection
        preprocessed_data = []
        for session in range(4):
            session_data = data[session]
            if self.__channel_selection is not None:
                session_data = self.__channel_selection.transform(session_data)
            
            # Epoching
            if self.__epoching is not None:
                session_data = self.__epoching.transform(session_data, max_img_rep=max_img_rep, seed=seed)
            else:
                raise Exception("Epoching is a required preprocessing step")
            
            # Denoising
            if self.__denoising is not None:
                session_data = self.__denoising.transform(session_data)
            
            if self.__frequency_filtering is not None:
                session_data = self.__frequency_filtering.transform(session_data)

                # Transforms
                if self.__fourier_transform is not None:
                    session_data = self.__fourier_transform.transform(session_data)
                
                if self.__wavelet_transform is not None:
                    session_data = self.__wavelet_transform.transform(session_data)
            preprocessed_data.append(session_data)


        # Save preprocessed data
        self.__save_preprocessed_data_per_subj_split_nice(preprocessed_data, subj, split)
        

    def __save_preprocessed_data_per_img_cond(self, data, subj, session, split):
        save_dir = os.getenv("SCIENCE_DIRECT_PREPROCESSED_DATA_FOLDER")
        os.makedirs(save_dir, exist_ok=True)
        # Image conditions × EEG repetitions × EEG channels × EEG time points
        epoched_eeg_data = data["epoched_eeg_data"]
        sfreq = int(data["sfreq"])
        data_dict = {}
        for img_cond_idx in range(epoched_eeg_data.shape[0]):
            img_cond = data["img_cond"][img_cond_idx]
            for rep_idx in range(epoched_eeg_data.shape[1]):
                rep = rep_idx
                for ch_idx in range(epoched_eeg_data.shape[2] - 1):
                    for freq_band in data["time_series"].keys():
                        ch = data["ch_names"][ch_idx]
                        if 'time_series' in data.keys():
                            time_series = data["time_series"][freq_band][img_cond_idx, rep_idx, ch_idx]
                            data_path = f"{save_dir}/time_series_subj-{subj}_session-{session}_split-{split}_rep-{rep}_ch_name-{ch}_sfreq-{sfreq}_freq_band-{freq_band}.npy"
                            if data_path not in data_dict.keys():
                                data_dict[data_path] = []
                            data_dict[data_path].append(time_series)
                        if 'fourier_transform' in data.keys():
                            data_path = f"{save_dir}/fourier_transform_subj-{subj}_session-{session}_split-{split}_rep-{rep}_ch_name-{ch}_sfreq-{sfreq}_freq_band-{freq_band}.npy"
                            fourier_transform = data["fourier_transform"][freq_band][img_cond_idx, rep_idx, ch_idx]
                            if data_path not in data_dict.keys():
                                data_dict[data_path] = []
                            data_dict[data_path].append(fourier_transform)
                        if 'morlet_wavelet_transform' in data.keys():
                            morlet_wavelet_transform = data["morlet_wavelet_transform"][freq_band][img_cond_idx, rep_idx, ch_idx]
                            data_path = f"{save_dir}/morlet_wavelet_transform_subj-{subj}_session-{session}_split-{split}_rep-{rep}_ch_name-{ch}_sfreq-{sfreq}_freq_band-{freq_band}.npy"
                            if data_path not in data_dict.keys():
                                data_dict[data_path] = []
                            data_dict[data_path].append(morlet_wavelet_transform)
        for data_path, output_data in data_dict.items():
            np.save(data_path, output_data)
        
        
    def __save_preprocessed_data_per_session_time(self, data, subj, session, split):
        save_dir = os.getenv("SCIENCE_DIRECT_PREPROCESSED_DATA_FOLDER_TRAINING_DATASET")
        os.makedirs(save_dir, exist_ok=True)
        # Image conditions × EEG repetitions × EEG channels × EEG time points
        epoched_eeg_data = data["epoched_eeg_data"]
        sfreq = int(data["sfreq"])
        data_path = f"{save_dir}/time_subj-{subj}_session-{session}_split-{split}.npy"
        output_data = []
        for img_cond_idx in range(epoched_eeg_data.shape[0]):
            img_cond = data["img_cond"][img_cond_idx]
            img_data = []
            for rep_idx in range(epoched_eeg_data.shape[1]):
                rep = rep_idx
                ch_data = []
                for ch_idx in range(epoched_eeg_data.shape[2] - 1):
                    freq_band_data = []
                    for freq_band in data["time_series"].keys():
                        if freq_band == 'bandpass':
                            continue
                        ch = data["ch_names"][ch_idx]
                        if 'time_series' in data.keys():
                            time_series = data["time_series"][freq_band][img_cond_idx, rep_idx, ch_idx]
                            freq_band_data.append(time_series)
                    freq_band_array = np.array(freq_band_data)[:, None, :]
                    ch_data.append(freq_band_array)
                ch_data_array = np.concatenate(ch_data, axis=1)
                img_data.append(ch_data_array)
            img_data_array = np.array(img_data)
            output_data.append(img_data_array)
        output_data = np.array(output_data)
        np.save(data_path, output_data)

    
    def __save_preprocessed_data_per_subj_split_nice(self, data, subj, split):
        save_dir = os.getenv("SCIENCE_DIRECT_PREPROCESSED_DATA_FOLDER_TRAINING_DATASET_NICE_ADAPTED")
        os.makedirs(save_dir, exist_ok=True)
        # Image conditions × EEG repetitions × EEG channels × EEG time points
        img_conditions = np.concatenate([session_data["img_cond"] for session_data in data])
        epoched_eeg_data_unmerged = np.concatenate([session_data["epoched_eeg_data"] for session_data in data])

        epoched_eeg_data = []
        for img_condition_idx in range(len(np.unique(img_conditions))):
            img_cond_idxs = np.where(img_conditions == img_condition_idx + 1)
            epoched_eeg_data_selected = epoched_eeg_data_unmerged[img_cond_idxs]
            epoched_eeg_data_selected = epoched_eeg_data_selected.reshape(-1, epoched_eeg_data_selected.shape[2], epoched_eeg_data_selected.shape[3])
            epoched_eeg_data_mean = np.mean(epoched_eeg_data_selected, axis=0, keepdims=True)
            epoched_eeg_data.append(epoched_eeg_data_mean)
        output_data = np.stack(epoched_eeg_data)
        data_path = f"{save_dir}/time_subj-{subj}_split-{split}.npy"
        np.save(data_path, output_data)
    



class EEGPreprocessorBuilder:

    def __init__(self):
        self.__eeg_preprocessor = EEGPreprocessor()
        

    def set_channel_selection(self, channel_selection):
        self.__eeg_preprocessor.channel_selection = channel_selection
        return self
    
    def set_epoching(self, epoching):
        self.__eeg_preprocessor.epoching = epoching
        return self
    
    def set_denoising(self, denoising):
        self.__eeg_preprocessor.denoising = denoising
        return self
    
    def set_frequency_filtering(self, frequency_filtering):
        self.__eeg_preprocessor.frequency_filtering = frequency_filtering
        return self
    
    def set_fourier_transform(self, fourier_transform):
        self.__eeg_preprocessor.fourier_transform = fourier_transform
        return self
    
    def set_wavelet_transform(self, fourier_transform):
        self.__eeg_preprocessor.wavelet_transform = fourier_transform
        return self
    
    def build(self):
        return self.__eeg_preprocessor

if __name__ == '__main__':
    from pathlib import Path
    import sys

    sys.path.append(str(Path(__file__).parent.parent.parent))

    from src.preprocessors.eeg_preprocessing_utils.channel_selection import ChannelSelectionByName
    from src.preprocessors.eeg_preprocessing_utils.resampling import ResamplingByFreq
    from src.preprocessors.eeg_preprocessing_utils.epoching import EpochingByMNE
    from src.preprocessors.eeg_preprocessing_utils.denoising import MVNN
    from src.preprocessors.eeg_preprocessing_utils.frequency_filtering import FrequencyBandFiltering
    from src.preprocessors.eeg_preprocessing_utils.transforms import FourierTransform, MorletWaveletTransform
    from src.datasets.EEGSciencedirectRoughIterator import EEGSciencedirectRoughIterator

    chan_order = ['Fp1', 'Fp2', 'AF7', 'AF3', 'AFz', 'AF4', 'AF8', 'F7', 'F5', 'F3',
				  'F1', 'F2', 'F4', 'F6', 'F8', 'FT9', 'FT7', 'FC5', 'FC3', 'FC1', 
				  'FCz', 'FC2', 'FC4', 'FC6', 'FT8', 'FT10', 'T7', 'C5', 'C3', 'C1',
				  'Cz', 'C2', 'C4', 'C6', 'T8', 'TP9', 'TP7', 'CP5', 'CP3', 'CP1', 
				  'CPz', 'CP2', 'CP4', 'CP6', 'TP8', 'TP10', 'P7', 'P5', 'P3', 'P1',
				  'Pz', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO3', 'POz', 'PO4', 'PO8',
				  'O1', 'Oz', 'O2']

    eeg_preprocessor = EEGPreprocessorBuilder()\
        .set_channel_selection(ChannelSelectionByName(names=chan_order))\
        .set_epoching(EpochingByMNE(tmin=-0.2, tmax=1.0, sfreq=250, baseline=(None, 0)))\
        .set_denoising(MVNN(mvnn_dim="epochs"))\
        .build()
    

    data_iter = iter(EEGSciencedirectRoughIterator())
    preprocess_configs = {
                "max_img_rep": 20,
                "seed": 0
            }
    
    to_be_preprocessed = []
    for split in ["training", "test"]:
    
        for i, data in enumerate(data_iter):
            if data["split"] != split:
                continue

            if data["split"] == "test":
                preprocess_configs = {
                    "max_img_rep": 20,
                    "seed": 20200220
                }
            else:
                preprocess_configs = {
                    "max_img_rep": 2,
                    "seed": 20200220
                }
            
            to_be_preprocessed.append(data["eeg_data"]["data"])

            if data['session'] == 4:
                eeg_preprocessor.preprocess(to_be_preprocessed,
                                                                data["subj"],
                                                                data["split"],
                                                                **preprocess_configs)
                to_be_preprocessed = []
            del data
        
        
