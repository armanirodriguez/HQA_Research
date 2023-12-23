import os
from tqdm import tqdm
from glob import glob
import torch
import torchaudio
from torch.utils.data import Dataset
from torchaudio.transforms import Resample

class AudioMNIST(Dataset):
    def __init__(self,
                 root_dir,
                 classes=range(10),
                 transform=None,
                 target_sample_rate=16000,
                 n_samples=16000,
                 preprocess_dataset=False):
        self.classes = classes
        self.transform = transform
        self.target_sample_rate = target_sample_rate
        self.n_samples = n_samples
        self.preprocess_dataset = preprocess_dataset
        if preprocess_dataset:
            self.spectrograms = self._load_spectrograms(root_dir)
        else:
            self.wav_files = self._load_wavs(root_dir)
        

    def __getitem__(self, index):
        if self.preprocess_dataset:
            return self.spectrograms[index]
        file_name, label = self.wav_files[index]
        spectogram = self._load_waveform_from_file(file_name)
        return spectogram, label

    def _resize(self, signal):
        n_samples = signal.shape[1]
        if n_samples > self.n_samples:
            signal = signal[:, :self.n_samples]
        elif n_samples < self.n_samples:
            n_missing_samples = self.n_samples - n_samples
            signal = torch.nn.functional.pad(signal, (0, n_missing_samples))
        return signal

    def __len__(self):
        if self.preprocess_dataset:
            return len(self.spectrograms)
        return len(self.wav_files)

    def _load_wavs(self, root_dir):
        wav_files = list()
        cwd = os.getcwd()
        os.chdir(root_dir)
        for file_name in glob('**/*.wav', recursive=True):
            label = int(os.path.basename(file_name)[0])
            if label in self.classes:
                file_name = os.path.join(root_dir, file_name)
                wav_files.append((file_name, label))
        os.chdir(cwd)
        return wav_files

    def _load_spectrograms(self, root_dir):
        spectograms = list()
        cwd = os.getcwd()
        os.chdir(root_dir)
        print('Preprocessing dataset...')
        for file_name in tqdm(glob('**/*.wav', recursive=True)):
            label = int(os.path.basename(file_name)[0])
            if label in self.classes:
                waveform = self._load_waveform_from_file(file_name)
                if self.transform is not None:
                    spectograms.append((self.transform(waveform),label))
        os.chdir(cwd)
        return spectograms

    def _load_waveform_from_file(self,file_name):
        """Loads and preprocesses a waveform from a file

        Args:
            file_name (str): the name of the file to load from
        """
        waveform, sr = torchaudio.load(file_name, normalize=True)
        waveform = self._resample(waveform, sr)
        waveform = self._resize(waveform)
        if self.transform is not None:
            return self.transform(waveform)
        return waveform
        
    
    def _resample(self, signal, sample_rate):
        resampler = Resample(sample_rate, self.target_sample_rate)
        return resampler(signal)