"""!
@brief Library for experiment cometml functionality

@author Efthymios Tzinis {etzinis2@illinois.edu}
@copyright University of Illinois at Urbana-Champaign
"""

import numpy as np


class AudioLogger(object):
    def __init__(self, fs=8000, n_sources=2):
        """
        :param fs: The sampling rate of the audio in Hz
        :param n_sources: The number of sources
        """
        self.fs = int(fs)
        self.n_sources = int(n_sources)

    def log_sp_enh_batch(self,
                         pr_speaker, pr_noise,
                         t_speaker, t_noise,
                         mix_batch,
                         experiment,
                         tag='',
                         step=None,
                         max_batch_items=4):
        print('Logging audio online...\n')

        tensors_with_names = zip(
            [pr_speaker, pr_noise, t_speaker, t_noise, mix_batch],
            ["est_speaker", "est_noise", "gt_speaker", "gt_noise", "mixture"]
        )

        for tensor_waveform, name in tensors_with_names:
            waveform = tensor_waveform.detach().cpu().numpy()
            waveform = waveform / np.abs(waveform).max(-1, keepdims=True)

            for b_ind in range(min(waveform.shape[0], max_batch_items)):
                experiment.log_audio(waveform[b_ind].squeeze(),
                                     sample_rate=self.fs,
                                     file_name=tag+'batch_{}_{}'.format(b_ind+1, name),
                                     metadata=None, overwrite=True,
                                     copy_to_tmp=True, step=step)

    def log_sp_enh_no_gt_batch(self,
                               pr_speaker, pr_noise,
                               mix_batch,
                               experiment,
                               tag='',
                               step=None,
                               max_batch_items=4):
        print('Logging audio online...\n')

        tensors_with_names = zip(
            [pr_speaker, pr_noise, mix_batch],
            ["est_speaker", "est_noise", "mixture"]
        )

        for tensor_waveform, name in tensors_with_names:
            waveform = tensor_waveform.detach().cpu().numpy()
            waveform = waveform / np.abs(waveform).max(-1, keepdims=True)

            for b_ind in range(min(waveform.shape[0], max_batch_items)):
                experiment.log_audio(waveform[b_ind].squeeze(),
                                     sample_rate=self.fs,
                                     file_name=tag+'batch_{}_{}'.format(b_ind+1, name),
                                     metadata=None, overwrite=True,
                                     copy_to_tmp=True, step=step)

    def log_batch(self,
                  pr_batch,
                  t_batch,
                  mix_batch,
                  experiment,
                  tag='',
                  step=None):
        """!
        :param pr_batch: Reconstructed wavs: Torch Tensor of size:
                         batch_size x num_sources x length_of_wavs
        :param t_batch: Target wavs: Torch Tensor of size:
                        batch_size x num_sources x length_of_wavs
        :param mix_batch: Batch of the mixtures: Torch Tensor of size:
                          batch_size x 1 x length_of_wavs
        :param experiment: Cometml experiment object
        :param step: The step that this batch belongs
        """
        print('Logging audio online...\n')
        mixture = mix_batch.detach().cpu().numpy()
        true_sources = t_batch.detach().cpu().numpy()
        pred_sources = pr_batch.detach().cpu().numpy()

        # Normalize the audio
        mixture = mixture / np.abs(mixture).max(-1, keepdims=True)
        true_sources = true_sources / np.abs(true_sources).max(-1, keepdims=True)
        pred_sources = pred_sources / np.abs(pred_sources).max(-1, keepdims=True)

        for b_ind in range(mixture.shape[0]):
            experiment.log_audio(mixture[b_ind].squeeze(),
                                 sample_rate=self.fs,
                                 file_name=tag+'batch_{}_mixture'.format(b_ind+1),
                                 metadata=None, overwrite=True,
                                 copy_to_tmp=True, step=step)
            for s_ind in range(self.n_sources):
                experiment.log_audio(
                    true_sources[b_ind][s_ind].squeeze(),
                    sample_rate=self.fs,
                    file_name=tag+'batch_{}_source_{}_true.wav'.format(b_ind+1,
                                                                       s_ind+1),
                    metadata=None, overwrite=True,
                    copy_to_tmp=True, step=step)
                experiment.log_audio(
                    pred_sources[b_ind][s_ind].squeeze(),
                    sample_rate=self.fs,
                    file_name=tag+'batch_{}_source_{}_est.wav'.format(b_ind+1,
                                                                      s_ind+1),
                    metadata=None, overwrite=True,
                    copy_to_tmp=True, step=step)


def report_losses_mean_and_std(res_dic, experiment, tr_step, val_step):
    """Wrapper for cometml loss report functionality.
    Reports the mean and the std of each loss by inferring the train and the
    val string and it assigns it accordingly.
    Args:
        losses_dict: Python Dict with the following structure:
                     res_dic[loss_name] = {'mean': 0., 'std': 0., 'acc': []}
        experiment:  A cometml experiment object
        tr_step:     The step/epoch index for training
        val_step:     The step/epoch index for validation
    Returns:
        The updated losses_dict with the current mean and std
    """

    for d_name in res_dic:
        for l_name in res_dic[d_name]:
            values = res_dic[d_name][l_name]['acc']
            mean_metric = np.mean(values)
            median_metric = np.median(values)
            std_metric = np.std(values)

            with experiment.validate():
                experiment.log_metric(
                    f'{d_name}_{l_name}_mean', mean_metric, step=val_step)
                experiment.log_metric(
                    f'{d_name}_{l_name}_median', median_metric, step=val_step)
                experiment.log_metric(
                    f'{d_name}_{l_name}_std', std_metric, step=val_step)

            res_dic[d_name][l_name]['mean'] = mean_metric
            res_dic[d_name][l_name]['median'] = median_metric
            res_dic[d_name][l_name]['std'] = std_metric

    return res_dic
