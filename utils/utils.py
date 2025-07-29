import librosa
import torchaudio
import torch

def load_file(filepath):
    signal, sr = librosa.load(filepath)
    # convert to mono if necessary
    if signal.shape[0] > 1:
        signal = librosa.to_mono(signal)
    return torch.from_numpy(signal), sr

def extract_spectrogram(signal, sr):
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(sample_rate=sr,
                                                           n_fft=1024,
                                                           hop_length=256,
                                                           n_mels=128,
                                                           window_fn=torch.hann_window,
                                                           pad_mode='constant')
    spect = mel_spectrogram(signal.float()) # -> (num_channels, num_mels, num_frames)
    spect = torchaudio.transforms.AmplitudeToDB()(spect)
    spect = spect.squeeze(0) # -> (num_mels, num_frames)

    mean = spect.mean(dim=0, keepdim=True)
    std = spect.std(dim=0, keepdim=True)
    spect = (spect - mean) / (std + 1e-8)
    return spect
    