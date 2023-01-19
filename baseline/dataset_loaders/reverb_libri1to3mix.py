"""!
@brief Pytorch dataloader for Libri1to3chime dataset with reverb augmentation.

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
import pyroomacoustics as pra


class Dataset(torch.utils.data.Dataset, abstract_dataset.Dataset):
    """ Dataset class for Librimix  1 to 3 for one and multi-speaker
    speech enhancement problems with the addition of reverb.
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

        self.room_l_region = [8., 10.]
        self.room_w_region = [7., 10.]
        self.room_h_region = [2.6, 3.5]
        self.rt60_region = [0.175, 0.45]
        self.distance_region = [0.1, 3.]
        self.source_h_region = [0.5, 2.5]

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

    def sample_room_parameters(self):
        room_params = {
            "length": np.random.uniform(self.room_l_region[0], self.room_l_region[1]),
            "width": np.random.uniform(self.room_w_region[0], self.room_w_region[1]),
            "height": np.random.uniform(self.room_h_region[0], self.room_h_region[1]),
            "rt60": np.random.uniform(self.rt60_region[0], self.rt60_region[1]),
        }
        return room_params

    def simulate_a_source_in_a_room(self, waveform, room_params):
        # Online room simulation copied and modified from:
        # https://github.com/etzinis/heterogeneous_separation/blob/main/heterogeneous_separation/dataset_loader/spatial_librispeech.py
        room_dims = [room_params["length"], room_params["width"], room_params["height"]]
        e_absorption, max_order = pra.inverse_sabine(room_params["rt60"], room_dims)
        room = pra.ShoeBox(
            room_dims, fs=int(self.sample_rate), materials=pra.Material(e_absorption),
            max_order=max_order
        )

        # Put the microphone in the middle of the room
        mic_loc_x = room_params["length"] / 2.
        mic_loc_y = room_params["width"] / 2.
        mic_loc_z = room_params["height"] / 2.
        mic_locs = np.c_[[mic_loc_x, mic_loc_y, mic_loc_z],]
        room.add_microphone_array(mic_locs)

        theta = np.random.uniform(0, np.pi)
        dist = np.random.uniform(self.distance_region[0], self.distance_region[-1])

        source_x_loc = np.cos(theta) * dist + mic_loc_x
        source_y_loc = np.sin(theta) * dist + mic_loc_y
        # A random speaker height sampling
        source_z_loc = np.random.uniform(self.source_h_region[0],
                                         self.source_h_region[-1])

        room.add_source([source_x_loc, source_y_loc, source_z_loc],
                        signal=waveform, delay=0.0)
        room.simulate()

        return room.mic_array.signals[-1, :]

    def __getitem__(self, idx):
        file_info = self.available_filenames_dic[idx]

        # Sample the room parameters
        if not self.augment:
            np.random.seed(len(self.split) + idx)
        room_params = self.sample_room_parameters()

        noise_w = self.wavread(file_info['noise_path'])
        max_len = noise_w.shape[-1]

        start_index = 0
        if self.augment and max_len > self.time_samples > 0:
            start_index = np.random.randint(0, max_len - self.time_samples)
        noise_tensor = self.get_padded_tensor(noise_w, start_index=start_index)

        src_tensor_list = []
        for src_idx in range(3):
            if src_idx + 1 <= file_info['n_active_sources']:
                src_w = self.wavread(file_info['sources_paths'][src_idx])
                src_w = self.simulate_a_source_in_a_room(src_w.numpy(), room_params)
                src_w = src_w[:max_len]
                source_tensor = self.get_padded_tensor(src_w, start_index=start_index)
            else:
                source_tensor = torch.zeros_like(noise_tensor)
            src_tensor_list.append(source_tensor)
        sources_tensor = torch.stack(src_tensor_list, 0)

        return sources_tensor, noise_tensor.unsqueeze(0)


def test_generator():
    import time
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
        batch_size=batch_size, num_workers=batch_size)
    print(f"Obtained: {len(generator)} files with fixed n_sources: {fixed_n_sources}")
    before = time.time()
    for sources, noise in generator:
        print(sources.shape)
        print(noise.shape)
        assert noise.shape == (batch_size, 1, time_samples)
        assert sources.shape == (batch_size, 3, time_samples)
        break
    this_time = time.time() - before
    print(f"It took me: {this_time} secs to fetch the batch")
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
