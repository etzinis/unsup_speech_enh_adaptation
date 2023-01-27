"""!
@brief Pytorch dataloader for Libri1to3chime dataset.

The noise source comes from chime whereas the speaker mixture is drawn from
libri3mix with the appropriate zero masking of the non-active speakers.

@author Efthymios Tzinis {etzinis2@illinois.edu}
@copyright University of illinois at Urbana Champaign
"""

import torch
import os
import numpy as np
import glob2

import sys
sys.path.append('../..')

import baseline.dataset_loaders.abstract_dataset as abstract_dataset
from __config__ import LIBRICHiME_ROOT_PATH

import torchaudio


class Dataset(torch.utils.data.Dataset, abstract_dataset.Dataset):
    """ Dataset class for Librimix 1 to 3 for one and multi-speaker
    speech enhancement problems mixed with noise from CHiME.
    """
    def __init__(self, **kwargs):
        super(Dataset, self).__init__()
        self.kwargs = kwargs

        self.zero_pad = self.get_arg_and_check_validness(
            'zero_pad', known_type=bool)

        self.normalize_audio = self.get_arg_and_check_validness(
            'normalize_audio', known_type=bool)

        self.split = self.get_arg_and_check_validness(
            'split', known_type=str, choices=['dev', 'eval'])

        self.n_samples = self.get_arg_and_check_validness(
            'n_samples', known_type=int, extra_lambda_checks=[lambda x: x >= -1])

        self.timelength = self.get_arg_and_check_validness(
            'timelength', known_type=float)

        self.fixed_n_sources = self.get_arg_and_check_validness(
            'fixed_n_sources', known_type=int, choices=[-1, 1, 2, 3])

        self.augment = self.get_arg_and_check_validness(
            'augment', known_type=bool)

        self.sample_rate = self.get_arg_and_check_validness(
            'sample_rate', known_type=int, choices=[8000, 16000])

        self.speech_dataset_dirpath = self.get_speech_path()            
            
        if self.fixed_n_sources < 0:
            self.available_speech_filenames = [
                os.path.join(os.path.split(os.path.split(f)[0])[1], os.path.basename(f))
                for f in glob2.glob(self.speech_dataset_dirpath + '/**/*_speech.wav')]
        else:    
            self.available_speech_filenames = [
                os.path.join(os.path.split(os.path.split(f)[0])[1], os.path.basename(f))
                for f in glob2.glob(self.speech_dataset_dirpath + '/' + str(self.fixed_n_sources) + '/*_speech.wav')]
            
            
        speaker_speech_samples = len(self.available_speech_filenames)

        # Check that all files are available.
        for fname in self.available_speech_filenames:
            this_path = os.path.join(self.speech_dataset_dirpath, fname)
            if not os.path.lexists(this_path):
                raise IOError(f"File not found in: {this_path}")

        if self.n_samples <= 0:
            self.n_samples = speaker_speech_samples
        else:
            self.available_speech_filenames = self.available_speech_filenames[:self.n_samples]
            self.available_noise_filenames = self.available_noise_filenames[:self.n_samples]

        self.available_filenames_dic = {}
        cnt = 0
        for idx, fname in enumerate(sorted(self.available_speech_filenames)):

                self.available_filenames_dic[cnt] = {
                    'speech_path': os.path.join(self.speech_dataset_dirpath,fname),
                    'noise_path': os.path.join(self.speech_dataset_dirpath,fname[:-10]+'noise.wav'),
                    'mix_path': os.path.join(self.speech_dataset_dirpath,fname[:-10]+'mix.wav')
                }
                cnt += 1

        self.n_samples = len(self.available_filenames_dic)
        self.time_samples = int(self.sample_rate * self.timelength)

    def get_speech_path(self):
        path = os.path.join(LIBRICHiME_ROOT_PATH, self.split)
        if os.path.lexists(path):
            return path
        else:
            raise IOError('Dataset path: {} not found!'.format(path))

    def wavread(self, path):
        waveform, _ = torchaudio.load(path)
        # # Resample in case of a given sample rate
        # if self.sample_rate < fs:
        #     waveform = torchaudio.functional.resample(
        #         waveform, fs, self.sample_rate, resampling_method="kaiser_window")
        # elif self.sample_rate > fs:
        #     raise ValueError("Cannot upsample.")

        # Convert to single-channel
        if len(waveform.shape) > 1:
            waveform = waveform.sum(0)

        return waveform
        # return waveform - waveform.mean()
        # return (1. * waveform - waveform.mean()) / (waveform.std() + 1e-8)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        file_info = self.available_filenames_dic[idx]

        noise_w = self.wavread(file_info['noise_path'])
        speech_w = self.wavread(file_info['speech_path'])
        mix_w = self.wavread(file_info['mix_path'])
        
        noise_len = noise_w.shape[-1]

        start_index = 0
        if self.augment and noise_len > self.time_samples > 0:
            start_index = np.random.randint(0, noise_len - self.time_samples)
            
        noise_tensor = self.get_padded_tensor(noise_w, start_index=start_index)
        speech_tensor = self.get_padded_tensor(speech_w, start_index=start_index)
        mix_tensor = self.get_padded_tensor(mix_w, start_index=start_index)

        return mix_tensor.unsqueeze(0), speech_tensor.unsqueeze(0), noise_tensor.unsqueeze(0)


def test_generator():
    def get_snr(tensor_1, tensor_2):
        return 10. * torch.log10((tensor_1**2).sum(-1) / ((tensor_2**2).sum(-1) + 1e-9))

    batch_size = 3
    sample_rate = 16000
    timelength = 4.0
    fixed_n_sources = -1
    split = 'dev'
    time_samples = int(sample_rate * timelength)
    data_loader = Dataset(
        sample_rate=sample_rate, fixed_n_sources=fixed_n_sources,
        timelength=timelength, augment=True,
        zero_pad=True, split=split,
        normalize_audio=False, n_samples=-1)
    
    generator = data_loader.get_generator(
        batch_size=batch_size, num_workers=1)
    
    print(f"Obtained: {len(generator)} files with fixed n_sources: {fixed_n_sources}")
    
    for mixtures, sources, noise in generator:
        print(mixtures.shape)
        print(sources.shape)
        print(noise.shape)
        print("Input-SNRs", get_snr(sources, noise))
        assert noise.shape == (batch_size, 1, time_samples)
        assert sources.shape == (batch_size, 1, time_samples)
        break

    # test the testing set with batch size 1 only
    batch_size = 1
    sample_rate = 16000
    timelength = 4.0
    fixed_n_sources = -1
    split = 'dev'
    time_samples = int(sample_rate * timelength)
    data_loader = Dataset(
        sample_rate=sample_rate, fixed_n_sources=fixed_n_sources,
        timelength=timelength, augment=True,
        zero_pad=True, split=split,
        normalize_audio=False, n_samples=-1)
    
    generator = data_loader.get_generator(
        batch_size=batch_size, num_workers=1)
    
    print(f"Obtained: {len(generator)} files with fixed n_sources: {fixed_n_sources}")
    
    for mixtures, sources, noise in generator:
        print(mixtures.shape)
        print(sources.shape)
        print(noise.shape)
        print("Input-SNRs", get_snr(sources, noise))
        assert noise.shape == (batch_size, 1, time_samples)
        assert sources.shape == (batch_size, 1, time_samples)
        break

if __name__ == "__main__":
    test_generator()
