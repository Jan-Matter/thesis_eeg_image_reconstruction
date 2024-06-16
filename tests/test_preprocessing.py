import sys
import os
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.preprocessors.eeg_preprocessing_utils.channel_selection import ChannelSelectionByName
from src.preprocessors.eeg_preprocessing_utils.resampling import ResamplingByFreq
from src.preprocessors.eeg_preprocessing_utils.denoising import MVNN

#Channel Selection Tests
def test_channel_selection_by_name():
    data = {
        "raw_eeg_data": np.random.rand(10, 100),
        "ch_names": ["Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "stim"],
        "ch_types": ["eeg", "eeg", "eeg", "eeg", "eeg", "eeg", "eeg", "eeg", "eeg", "stim"]
    }
    channel_selection = ChannelSelectionByName(names=["Fp1", "Fp2", "C3", "C4"])
    data = channel_selection.transform(data)
    assert data["raw_eeg_data"].shape == (5, 100)
    assert data["ch_names"] == ["Fp1", "Fp2", "C3", "C4", "stim"]
    assert data["ch_types"] == ["eeg", "eeg", "eeg", "eeg", "stim"]

def test_channel_selection_by_regex():
    data = {
        "raw_eeg_data": np.random.rand(10, 100),
        "ch_names": ["Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "stim"],
        "ch_types": ["eeg", "eeg", "eeg", "eeg", "eeg", "eeg", "eeg", "eeg", "eeg", "stim"]
    }
    channel_selection = ChannelSelectionByName(regex="^F")
    data = channel_selection.transform(data)
    assert data["raw_eeg_data"].shape == (5, 100)
    assert data["ch_names"] == ["Fp1", "Fp2", "F3", "F4", "stim"]
    assert data["ch_types"] == ["eeg", "eeg", "eeg", "eeg", "stim"]



# Epoching Tests
from src.preprocessors.eeg_preprocessing_utils.epoching import EpochingByMNE

def test_epoching_by_mne():
    # Create dummy data
    ch_names = ['Fz', 'Cz', 'Pz', 'stim']
    sfreq = 100
    ch_types = ['eeg', 'eeg', 'eeg', 'stim']
    n_samples = 1000
    raw_eeg_data = np.random.randn(len(ch_names), n_samples)
    raw_eeg_data[3] = np.zeros(n_samples, dtype=np.float64)
    raw_eeg_data[3][::10] = 1.0
    raw_eeg_data[3][0] = 0.0
    data = {
        'ch_names': ch_names,
        'sfreq': sfreq,
        'ch_types': ch_types,
        'raw_eeg_data': raw_eeg_data
    }

    # Initialize EpochingByMNE object
    tmin = -0.2
    tmax = 0.8
    epoching = EpochingByMNE(tmin=tmin, tmax=tmax, baseline=(None, 0), max_img_rep=20)

    # Test transform method
    sorted_epochs = epoching.transform(data, max_img_rep=20, seed=0)

    # Check output
    assert sorted_epochs["epoched_eeg_data"].shape == (1, 20, len(ch_names), sfreq * (tmax - tmin) + 1)



# Denoising Tests
def test_mvnn_denoising():
    # Create dummy data
    data = {
        "epoched_eeg_data": np.random.rand(10, 64, 12, 101)
    }
    # Initialize MVNN denoising object
    mvnn = MVNN()
    # Apply denoising transform
    denoised_data = mvnn.transform(data)
    # Check if the shape of the output is correct
    assert denoised_data["epoched_eeg_data"].shape == data["epoched_eeg_data"].shape

if __name__ == "__main__":
    test_channel_selection_by_name()