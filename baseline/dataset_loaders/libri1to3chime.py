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


import baseline.dataset_loaders.abstract_dataset as abstract_dataset
from __config__ import LIBRI3MIX_ROOT_PATH
from __config__ import CHiME_ROOT_PATH

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

        self.min_or_max = self.get_arg_and_check_validness(
            'min_or_max', known_type=str, choices=['min', 'max'])

        self.split = self.get_arg_and_check_validness(
            'split', known_type=str, choices=['dev', 'test'])

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

        self.speaker_mix_dataset_dirpath = self.get_speaker_mix_path()
        noise_split = 'eval' if self.split == 'test' else 'dev'
        self.noise_dataset_dirpath = self.get_noise_path(split=noise_split)

        self.available_speech_filenames = [
            os.path.basename(f) for f in
            glob2.glob(os.path.join(self.speaker_mix_dataset_dirpath, 's1') + '/*.wav')]
        self.available_noise_filenames = [
            os.path.join(os.path.split(os.path.split(f)[0])[1], os.path.basename(f))
            for f in glob2.glob(self.noise_dataset_dirpath + '/0/*.wav')]

        speaker_mix_samples = len(self.available_speech_filenames)
        noise_samples = len(self.available_noise_filenames)

        # Populate noise files to create more mixtures
        n_populate = speaker_mix_samples // noise_samples + 1
        self.available_noise_filenames = self.available_noise_filenames * n_populate
        self.available_noise_filenames = self.available_noise_filenames[:speaker_mix_samples]

        self.source_types = ['s1', 's2', 's3']

        # Check that all files are available.
        for fname in self.available_speech_filenames:
            for s_type in self.source_types:
                this_path = os.path.join(self.speaker_mix_dataset_dirpath, s_type, fname)
                if not os.path.lexists(this_path):
                    raise IOError(f"File not found in: {this_path}")

        for fname in self.available_noise_filenames:
            this_path = os.path.join(self.noise_dataset_dirpath, fname)
            if not os.path.lexists(this_path):
                raise IOError(f"File not found in: {this_path}")

        if self.n_samples <= 0:
            self.n_samples = speaker_mix_samples
        else:
            self.available_speech_filenames = self.available_speech_filenames[:self.n_samples]
            self.available_noise_filenames = self.available_noise_filenames[:self.n_samples]

        # If selected choose the number of files with 1 - 2 - 3 speakers.
        self.available_filenames_dic = {}
        cnt = 0
        for idx, fname in enumerate(sorted(self.available_speech_filenames)):
            this_n_sources = 3
            for j in range(3):
                if idx / self.n_samples <= self.n_speakers_cdf[j]:
                    this_n_sources = j + 1
                    break

            this_noise_path = os.path.join(self.noise_dataset_dirpath,
                                           self.available_noise_filenames[idx])

            if self.fixed_n_sources < 0 or (this_n_sources == self.fixed_n_sources):
                self.available_filenames_dic[cnt] = {
                    'sources_paths': [
                        os.path.join(self.speaker_mix_dataset_dirpath, s, fname)
                        for s in self.source_types],
                    'noise_path': this_noise_path,
                    'n_active_sources': this_n_sources
                }
                cnt += 1

        self.n_samples = len(self.available_filenames_dic)
        self.time_samples = int(self.sample_rate * self.timelength)

    def get_speaker_mix_path(self):
        path = os.path.join(LIBRI3MIX_ROOT_PATH,
                            'wav{}k'.format(int(self.sample_rate / 1000)),
                            self.min_or_max, self.split)
        if os.path.lexists(path):
            return path
        else:
            raise IOError('Dataset path: {} not found!'.format(path))

    def get_noise_path(self, split):
        path = os.path.join(CHiME_ROOT_PATH, split)
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
        noise_len = noise_w.shape[-1]

        start_index = 0
        if self.augment and noise_len > self.time_samples > 0:
            start_index = np.random.randint(0, noise_len - self.time_samples)
        noise_tensor = self.get_padded_tensor(noise_w, start_index=start_index)

        src_tensor_list = []
        for src_idx in range(3):
            if src_idx + 1 <= file_info['n_active_sources']:
                src_w = self.wavread(file_info['sources_paths'][src_idx])
                start_index = 0
                if self.augment and src_w.shape[-1] > self.time_samples > 0:
                    start_index = np.random.randint(0, src_w.shape[-1] - self.time_samples)
                source_tensor = self.get_padded_tensor(src_w, start_index=start_index)
            else:
                source_tensor = torch.zeros_like(noise_tensor)
            src_tensor_list.append(source_tensor)

        if self.time_samples < 0:
            # If the whole waveform needs to be returned, trim the tensors to align them
            min_len = noise_tensor.shape[-1]
            min_len = min([min_len] + [src_tensor_list[j].shape[-1] for j in range(3)])
            for src_idx in range(3):
                src_tensor_list[src_idx] = src_tensor_list[src_idx][:min_len]
            noise_tensor = noise_tensor[:min_len]

        sources_tensor = torch.stack(src_tensor_list, 0)

        # Mix the speaker mix with noise at a random input-SNR
        speaker_mix = sources_tensor.sum(0)
        if not self.augment:
            np.random.seed(len(self.split) + idx)
        sampled_input_snr = np.random.uniform(-5., 15.)
        new_speaker_mix_tensor, new_noise_tensor = self.mix_2_with_specified_snr(
            speaker_mix, noise_tensor, snr_ratio=sampled_input_snr)

        return new_speaker_mix_tensor.unsqueeze(0), new_noise_tensor.unsqueeze(0)


def test_generator():
    def get_snr(tensor_1, tensor_2):
        return 10. * torch.log10((tensor_1**2).sum(-1) / ((tensor_2**2).sum(-1) + 1e-9))

    batch_size = 3
    sample_rate = 16000
    timelength = 3.0
    fixed_n_sources = -1
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
        print(sources.shape)
        print(noise.shape)
        print("Input-SNRs", get_snr(sources, noise))
        assert noise.shape == (batch_size, 1, time_samples)
        assert sources.shape == (batch_size, 1, time_samples)
        break

    # test the testing set with batch size 1 only
    batch_size = 1
    sample_rate = 16000
    timelength = - 1.
    fixed_n_sources = -1
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
        print(sources.shape)
        print(noise.shape)
        print("Input-SNRs", get_snr(sources, noise))
        assert noise.shape[:-1] == (batch_size, 1)
        assert sources.shape[:-1] == (batch_size, 1)
        break

if __name__ == "__main__":
    test_generator()
