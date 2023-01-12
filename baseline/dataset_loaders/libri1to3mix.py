"""!
@brief Pytorch dataloader for Libri2mix dataset.

@author Efthymios Tzinis {etzinis2@illinois.edu}
@copyright University of illinois at Urbana Champaign
"""

import torch
import os
import numpy as np
import glob2


import baseline.dataset_loaders.abstract_dataset as abstract_dataset
from __config__ import LIBRI3MIX_ROOT_PATH
import torchaudio


class Dataset(torch.utils.data.Dataset, abstract_dataset.Dataset):
    """ Dataset class for Librimix  1 to 3 for one and multi-speaker
    speech enhancement problems.

    Example of kwargs:
        root_dirpath='/mnt/data/wham', task='enh_single',
        split='tr', sample_rate=8000, timelength=4.0,
        normalize_audio=False, n_samples=0, zero_pad=False
    """
    def __init__(self, **kwargs):
        super(Dataset, self).__init__()
        self.kwargs = kwargs

        self.zero_pad = self.get_arg_and_check_validness(
            'zero_pad', known_type=bool)

        self.normalize_audio = self.get_arg_and_check_validness(
            'normalize_audio', known_type=bool)

        self.min_or_max = self.get_arg_and_check_validness(
            'min_or_max', known_type=str, choices=['min', 'max'])

        self.split = self.get_arg_and_check_validness(
            'split', known_type=str, choices=['dev', 'test', 'train-100',
                                              'train-360'])

        self.n_samples = self.get_arg_and_check_validness(
            'n_samples', known_type=int, extra_lambda_checks=[lambda x: x >= -1])

        self.timelength = self.get_arg_and_check_validness(
            'timelength', known_type=float)

        self.fixed_n_sources = self.get_arg_and_check_validness(
            'fixed_n_sources', known_type=int, choices=[-1, 1, 2, 3])

        self.augment = self.get_arg_and_check_validness(
            'augment', known_type=bool)

        self.n_speakers_priors = self.get_arg_and_check_validness(
            'n_speakers_priors', known_type=list,
            extra_lambda_checks=[lambda y: (sum(y) - 1.) < 1e-5,
                                 lambda y: len(y) == 3])
        self.n_speakers_cdf = np.cumsum(self.n_speakers_priors).tolist()

        self.sample_rate = self.get_arg_and_check_validness(
            'sample_rate', known_type=int, choices=[8000, 16000])

        self.dataset_dirpath = self.get_path()

        self.available_filenames = [
            os.path.basename(f) for f in
            glob2.glob(os.path.join(self.dataset_dirpath, 's1') + '/*.wav')]

        self.source_types = ['s1', 's2', 's3', 'noise']

        # Check that all files are available.
        for fname in self.available_filenames:
            for s_type in self.source_types:
                this_path = os.path.join(self.dataset_dirpath, s_type, fname)
                if not os.path.lexists(this_path):
                    raise IOError(f"File not found in: {this_path}")

        if self.n_samples <= 0:
            self.n_samples = len(self.available_filenames)
        else:
            self.available_filenames = self.available_filenames[:self.n_samples]

        # If selected choose the number of files with 1 - 2 - 3 speakers.
        self.available_filenames_dic = {}
        cnt = 0
        for idx, fname in enumerate(sorted(self.available_filenames)):
            this_n_sources = 3
            for j in range(3):
                if idx / self.n_samples <= self.n_speakers_cdf[j]:
                    this_n_sources = j + 1
                    break

            if self.fixed_n_sources < 0 or (this_n_sources == self.fixed_n_sources):
                self.available_filenames_dic[cnt] = {
                    'sources_paths': [
                        os.path.join(self.dataset_dirpath, s, fname)
                        for s in self.source_types],
                    'noise_path': os.path.join(self.dataset_dirpath, 'noise', fname),
                    'n_active_sources': this_n_sources
                }
                cnt += 1

        self.n_samples = len(self.available_filenames_dic)
        self.time_samples = int(self.sample_rate * self.timelength)

    def get_path(self):
        path = os.path.join(LIBRI3MIX_ROOT_PATH,
                            'wav{}k'.format(int(self.sample_rate / 1000)),
                            self.min_or_max, self.split)
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
        max_len = noise_w.shape[-1]

        start_index = 0
        if self.augment and max_len > self.time_samples > 0:
            start_index = np.random.randint(0, max_len - self.time_samples)
        noise_tensor = self.get_padded_tensor(noise_w - noise_w.mean(),
                                              start_index=start_index)

        src_tensor_list = []
        for src_idx in range(3):
            if src_idx + 1 <= file_info['n_active_sources']:
                src_w = self.wavread(file_info['sources_paths'][src_idx])
                source_tensor = self.get_padded_tensor(
                    src_w - src_w.mean(), start_index=start_index)
            else:
                source_tensor = torch.zeros_like(noise_tensor)
            src_tensor_list.append(source_tensor)
        sources_tensor = torch.stack(src_tensor_list, 0)

        return sources_tensor, noise_tensor.unsqueeze(0)


def test_generator():
    batch_size = 3
    sample_rate = 8000
    timelength = 3.0
    fixed_n_sources = -1
    split = 'train-360'
    min_or_max = 'min'
    n_speakers_priors = [0.34, 0.33, 0.33]
    time_samples = int(sample_rate * timelength)
    data_loader = Dataset(
        sample_rate=sample_rate, fixed_n_sources=fixed_n_sources,
        timelength=timelength, augment='train' in split,
        zero_pad=True, min_or_max=min_or_max, split=split,
        normalize_audio=False, n_samples=-1,
        n_speakers_priors=n_speakers_priors)
    generator = data_loader.get_generator(
        batch_size=batch_size, num_workers=1)
    print(f"Obtained: {len(generator)} files with fixed n_sources: {fixed_n_sources}")
    for sources, noise in generator:
        print(sources.shape)
        print(noise.shape)
        assert noise.shape == (batch_size, 1, time_samples)
        assert sources.shape == (batch_size, 3, time_samples)
        break

    # test the testing set with batch size 1 only
    batch_size = 1
    sample_rate = 8000
    timelength = - 1.
    fixed_n_sources = -1
    split = 'train-360'
    min_or_max = 'min'
    n_speakers_priors = [0.34, 0.33, 0.33]
    time_samples = int(sample_rate * timelength)
    data_loader = Dataset(
        sample_rate=sample_rate, fixed_n_sources=fixed_n_sources,
        timelength=timelength, augment='train' in split,
        zero_pad=True, min_or_max=min_or_max, split=split,
        normalize_audio=False, n_samples=-1,
        n_speakers_priors=n_speakers_priors)
    generator = data_loader.get_generator(
        batch_size=batch_size, num_workers=1)
    print(f"Obtained: {len(generator)} files with fixed n_sources: {fixed_n_sources}")
    for sources, noise in generator:
        print(sources.shape)
        print(noise.shape)
        assert noise.shape[:-1] == (batch_size, 1)
        assert sources.shape[:-1] == (batch_size, 3)
        break

    # test that all the fixed sources are working
    batch_size = 3
    sample_rate = 8000
    timelength = 3.0
    fixed_n_sources = 1
    split = 'dev'
    min_or_max = 'min'
    n_speakers_priors = [0.34, 0.33, 0.33]
    time_samples = int(sample_rate * timelength)
    data_loader = Dataset(
        sample_rate=sample_rate, fixed_n_sources=fixed_n_sources,
        timelength=timelength, augment='train' in split,
        zero_pad=True, min_or_max=min_or_max, split=split,
        normalize_audio=False, n_samples=-1,
        n_speakers_priors=n_speakers_priors)
    generator = data_loader.get_generator(
        batch_size=batch_size, num_workers=1)
    print(f"Obtained: {len(generator)} files with fixed n_sources: {fixed_n_sources}")
    for sources, noise in generator:
        assert not np.allclose(sources[:, 0, :], np.zeros_like(sources[:, 0, :]))
        assert np.allclose(sources[:, 1, :], np.zeros_like(sources[:, 1, :]))
        assert np.allclose(sources[:, 2, :], np.zeros_like(sources[:, 2, :]))
        break

    batch_size = 3
    sample_rate = 8000
    timelength = 3.0
    fixed_n_sources = 2
    split = 'dev'
    min_or_max = 'min'
    n_speakers_priors = [0.34, 0.33, 0.33]
    time_samples = int(sample_rate * timelength)
    data_loader = Dataset(
        sample_rate=sample_rate, fixed_n_sources=fixed_n_sources,
        timelength=timelength, augment='train' in split,
        zero_pad=True, min_or_max=min_or_max, split=split,
        normalize_audio=False, n_samples=-1,
        n_speakers_priors=n_speakers_priors)
    generator = data_loader.get_generator(
        batch_size=batch_size, num_workers=1)
    print(f"Obtained: {len(generator)} files with fixed n_sources: {fixed_n_sources}")
    for sources, noise in generator:
        assert not np.allclose(sources[:, 0, :], np.zeros_like(sources[:, 0, :]))
        assert not np.allclose(sources[:, 1, :], np.zeros_like(sources[:, 1, :]))
        assert np.allclose(sources[:, 2, :], np.zeros_like(sources[:, 2, :]))
        break

    batch_size = 3
    sample_rate = 8000
    timelength = 3.0
    fixed_n_sources = 3
    split = 'dev'
    min_or_max = 'min'
    n_speakers_priors = [0.34, 0.33, 0.33]
    time_samples = int(sample_rate * timelength)
    data_loader = Dataset(
        sample_rate=sample_rate, fixed_n_sources=fixed_n_sources,
        timelength=timelength, augment='train' in split,
        zero_pad=True, min_or_max=min_or_max, split=split,
        normalize_audio=False, n_samples=-1,
        n_speakers_priors=n_speakers_priors)
    generator = data_loader.get_generator(
        batch_size=batch_size, num_workers=1)
    print(f"Obtained: {len(generator)} files with fixed n_sources: {fixed_n_sources}")
    for sources, noise in generator:
        assert not np.allclose(sources[:, 0, :], np.zeros_like(sources[:, 0, :]))
        assert not np.allclose(sources[:, 1, :], np.zeros_like(sources[:, 1, :]))
        assert not np.allclose(sources[:, 2, :], np.zeros_like(sources[:, 2, :]))
        break

if __name__ == "__main__":
    test_generator()
