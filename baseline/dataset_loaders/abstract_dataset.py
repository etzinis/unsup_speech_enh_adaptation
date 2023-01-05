"""
Pytorch abstract dataset class for inheritance.

This code has been shamelessly copied from under the proper licence:
https://github.com/etzinis/fedenhance/blob/master/fedenhance/dataset_loader/abstract_dataset.py
"""

from abc import abstractmethod
import inspect
import numpy as np
import torch


class Dataset:

    @abstractmethod
    def mix_2_with_specified_snr(self, wav_tensor_1, wav_tensor_2, snr_ratio):
        power_1 = torch.sqrt(torch.sum(wav_tensor_1 ** 2))
        power_2 = torch.sqrt(torch.sum(wav_tensor_2 ** 2))
        new_power_ratio = np.sqrt(np.power(10., snr_ratio / 10.))
        new_wav_tensor_1 = new_power_ratio * wav_tensor_1 / (power_1 + 10e-8)
        new_wav_tensor_2 = wav_tensor_2 / (power_2 + 10e-8)
        return new_wav_tensor_1, new_wav_tensor_2

    @abstractmethod
    def normalize_tensor_wav(self, wav_tensor, eps=1e-8, std=None):
        mean = wav_tensor.mean(-1, keepdim=True)
        if std is None:
            std = wav_tensor.std(-1, keepdim=True)
        return (wav_tensor - mean) / (std + eps)

    def get_generator(self, batch_size=4, num_workers=4, pin_memory=True):
        generator_params = {'batch_size': batch_size,
                            'shuffle': self.augment,
                            'num_workers': num_workers,
                            'drop_last': True}
        return torch.utils.data.DataLoader(self, **generator_params,
                                           pin_memory=pin_memory)

    def get_padded_tensor(self, numpy_waveform, start_index=0):
        max_len = numpy_waveform.shape[0]
        rand_start = start_index
        if start_index is None and max_len > self.time_samples:
            rand_start = np.random.randint(0, max_len - self.time_samples)

        if self.time_samples > 0:
            tensor_wav = torch.tensor(
                numpy_waveform[rand_start:rand_start + self.time_samples],
                dtype=torch.float32)
            return self.safe_pad(tensor_wav)
        else:
            return torch.tensor(numpy_waveform, dtype=torch.float32)

    @abstractmethod
    def safe_pad(self, tensor_wav, dtype=torch.float32):
        if self.zero_pad and tensor_wav.shape[0] < self.time_samples:
            appropriate_shape = tensor_wav.shape
            padded_wav = torch.zeros(
                list(appropriate_shape[:-1]) + [self.time_samples], dtype=dtype)
            padded_wav[:tensor_wav.shape[0]] = tensor_wav
            return padded_wav[:self.time_samples]
        elif self.time_samples > 0:
            return tensor_wav[:self.time_samples]
        else:
            return tensor_wav


    @abstractmethod
    def get_arg_and_check_validness(self,
                                    key,
                                    choices=None,
                                    known_type=None,
                                    extra_lambda_checks=None):
        try:
            value = self.kwargs[key]
        except Exception as e:
            print(e)
            raise KeyError("Argument: <{}> does not exist in pytorch "
                           "dataloader keyword arguments".format(key))

        if known_type is not None:
            if not isinstance(value, known_type):
                raise TypeError("Value: <{}> for key: <{}> is not an "
                                "instance of "
                                "the known selected type: <{}>"
                                "".format(value, key, known_type))

        if choices is not None:
            if isinstance(value, list):
                if not all([v in choices for v in value]):
                    raise ValueError("Values: <{}> for key: <{}>  "
                                     "contain elements in a"
                                     "regime of non appropriate "
                                     "choices instead of: <{}>"
                                     "".format(value, key, choices))
            else:
                if value not in choices:
                    raise ValueError("Value: <{}> for key: <{}> is "
                                     "not in the "
                                     "regime of the appropriate "
                                     "choices: <{}>"
                                     "".format(value, key, choices))

        if extra_lambda_checks is not None:
            all_checks_passed = all([f(value)
                                     for f in extra_lambda_checks])
            if not all_checks_passed:
                raise ValueError(
                    "Value(s): <{}> for key: <{}>  "
                    "does/do not fulfill the predefined checks: "
                    "<{}>".format(value, key,
                    [inspect.getsourcelines(c)[0][0].strip()
                     for c in extra_lambda_checks
                     if not c(value)]))
        return value
