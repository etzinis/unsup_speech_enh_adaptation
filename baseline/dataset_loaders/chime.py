"""!
@brief Pytorch dataloader for CHiME dataset.

@author Mostafa Sadeghi
@author Efthymios Tzinis {etzinis2@illinois.edu}
@copyright University of illinois at Urbana Champaign
"""

import torch
import os
import numpy as np
import glob2


import baseline.dataset_loaders.abstract_dataset as abstract_dataset
from __config__ import CHiME_ROOT_PATH

import torchaudio


class Dataset(torch.utils.data.Dataset, abstract_dataset.Dataset):
    """ Dataset class for the CHiME dataset for one and multi-speaker
    speech enhancement problems.
    """
    def __init__(self, **kwargs):
        super(Dataset, self).__init__()
        self.kwargs = kwargs

        self.zero_pad = self.get_arg_and_check_validness(
            'zero_pad', known_type=bool)

        self.normalize_audio = self.get_arg_and_check_validness(
            'normalize_audio', known_type=bool)

        self.use_vad = self.get_arg_and_check_validness(
            'use_vad', known_type=bool)

        self.get_only_active_speakers = self.get_arg_and_check_validness(
            'get_only_active_speakers', known_type=bool)

        self.split = self.get_arg_and_check_validness(
            'split', known_type=str, choices=['dev', 'eval', 'train'])

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

        self.dataset_dirpath = self.get_path()

        if self.split == 'train' and not self.use_vad:
            self.available_filenames = [
                os.path.join(os.path.split(os.path.split(f)[0])[1], os.path.basename(f))
                for f in glob2.glob(self.dataset_dirpath + '/unlabeled_10s/*.wav')]
        elif self.split == 'train' and self.use_vad:
            self.available_filenames = [
                os.path.join(os.path.split(os.path.split(f)[0])[1], os.path.basename(f))
                for f in glob2.glob(self.dataset_dirpath + '/unlabeled_vad_10s/*.wav')]
        elif self.get_only_active_speakers and self.fixed_n_sources < 0:
            self.available_filenames = []
            for i in range(1, 4):
                self.available_filenames += \
                    [os.path.join(os.path.split(os.path.split(f)[0])[1], os.path.basename(f))
                     for f in glob2.glob(self.dataset_dirpath + '/' + str(i) + '/*.wav')]
        elif self.fixed_n_sources < 0:
            self.available_filenames = [
                os.path.join(os.path.split(os.path.split(f)[0])[1], os.path.basename(f))
                for f in glob2.glob(self.dataset_dirpath + '/**/*.wav')]
        else:    
            self.available_filenames = [
                os.path.join(os.path.split(os.path.split(f)[0])[1], os.path.basename(f))
                for f in glob2.glob(self.dataset_dirpath + '/' + str(self.fixed_n_sources) + '/*.wav')]

        # Check that all files are available.
        for fname in self.available_filenames:
            this_path = os.path.join(self.dataset_dirpath, fname)
            if not os.path.lexists(this_path):
                raise IOError(f"File not found in: {this_path}")

        if self.n_samples <= 0:
            self.n_samples = len(self.available_filenames)
        else:
            self.available_filenames = self.available_filenames[:self.n_samples]

        self.n_samples = len(self.available_filenames)
        self.time_samples = int(self.sample_rate * self.timelength)

    def get_path(self):
        path = os.path.join(CHiME_ROOT_PATH, self.split)
        if os.path.lexists(path):
            return path
        else:
            raise IOError('Dataset path: {} not found!'.format(path))

    def wavread(self, path):
        waveform, fs = torchaudio.load(path)
        # Resample in case of a given sample rate
        if self.sample_rate < fs:
            waveform = torchaudio.functional.resample(
                waveform, fs, self.sample_rate, resampling_method="kaiser_window")
        elif self.sample_rate > fs:
            raise ValueError("Cannot upsample.")

        # Convert to single-channel
        if len(waveform.shape) > 1:
            waveform = waveform.sum(0)

        return waveform - waveform.mean()
        # return (1. * waveform - waveform.mean()) / (waveform.std() + 1e-8)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        mixture_path = self.available_filenames[idx]

        mixture_w = self.wavread(os.path.join(self.dataset_dirpath, mixture_path))
        max_len = mixture_w.shape[-1]

        start_index = 0
        if self.augment and max_len > self.time_samples > 0:
            start_index = np.random.randint(0, max_len - self.time_samples)
        mixture_tensor = self.get_padded_tensor(mixture_w - mixture_w.mean(),
                                                start_index=start_index)

        return mixture_tensor


def test_generator():
    batch_size = 1
    sample_rate = 16000
    timelength = 3.0
    fixed_n_sources = -1
    use_vad = False
    get_only_active_speakers = False
    split = 'train'
    time_samples = int(sample_rate * timelength)
    data_loader = Dataset(
        sample_rate=sample_rate, fixed_n_sources=fixed_n_sources,
        timelength=timelength, augment='train' in split, use_vad=use_vad,
        zero_pad=True, split=split, get_only_active_speakers=get_only_active_speakers,
        normalize_audio=False, n_samples=-1)
    generator = data_loader.get_generator(
        batch_size=batch_size, num_workers=1)
    print(f"Obtained: {len(generator)} files with fixed n_sources: {fixed_n_sources}")
    for mixture in generator:
        print(mixture.shape)
        assert mixture.shape == (batch_size, time_samples)
        break


if __name__ == "__main__":
    test_generator()
