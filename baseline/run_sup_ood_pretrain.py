"""!
@brief Running an experiment with a supervised Sudo rm -rf teacher
for one- and multi-speaker speech enhancement settings

@author Efthymios Tzinis {etzinis2@illinois.edu}
@copyright University of Illinois at Urbana-Champaign
"""

import os

from __config__ import API_KEY
from comet_ml import Experiment

import torch
import numpy as np

from tqdm import tqdm
from pprint import pprint
import baseline.utils.cmd_parser as parser
import baseline.utils.cometml_logger as cometml_logger
import baseline.utils.dataset_setup as dataset_setup
import baseline.utils.mixture_consistency as mixture_consistency
import baseline.models.improved_sudormrf as improved_sudormrf
import baseline.metrics.dnnmos_metric as dnnmos_metric
from asteroid.losses import pairwise_neg_sisdr
from multiprocessing import Pool


def compute_dnsmos_process(est_speech):
    return dnnmos_metric.compute_dnsmos(est_speech, fs=16000)


args = parser.get_args()
hparams = vars(args)
generators = dataset_setup.supervised_setup(hparams)

audio_logger = cometml_logger.AudioLogger(fs=hparams["fs"], n_sources=2)

experiment = Experiment(API_KEY, project_name=hparams["project_name"])
experiment.log_parameters(hparams)
experiment_name = '_'.join(hparams['cometml_tags'])

for tag in hparams['cometml_tags']:
    experiment.add_tag(tag)
if hparams['experiment_name'] is not None:
    experiment.set_name(hparams['experiment_name'])
else:
    experiment.set_name(experiment_name)

checkpoint_storage_path = os.path.join(hparams["checkpoint_storage_path"],
                                       experiment_name)
if checkpoint_storage_path is not None:
    if hparams["save_models_every"] <= 0:
        raise ValueError("Expected a value greater than 0 for checkpoint storing.")
    if not os.path.exists(checkpoint_storage_path):
        os.makedirs(checkpoint_storage_path)
        print(f"Created directory: {checkpoint_storage_path}")


os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(
    [cad for cad in hparams['cuda_available_devices']])

train_loss_name, train_loss = "train_neg_sisdr", pairwise_neg_sisdr

val_losses = {
    "train_speaker": {"sisdr": pairwise_neg_sisdr},
    "train_noise": {"sisdr": pairwise_neg_sisdr},
    "train_total": {"sisdr": pairwise_neg_sisdr},
}
for val_set in [x for x in generators if not x == 'train']:
    if generators[val_set] is None:
        continue
    if val_set in ['val_chime_1sp', 'test_chime_1sp']:
        val_losses[val_set] = {
            "sig_mos": None,
            "bak_mos": None,
            "ovr_mos": None,
        }
    else:
        val_losses[val_set] = {
            "sisdr": pairwise_neg_sisdr, "sisdri": pairwise_neg_sisdr
        }

model = improved_sudormrf.SuDORMRF(out_channels=hparams['out_channels'],
                                   in_channels=hparams['in_channels'],
                                   num_blocks=hparams['num_blocks'],
                                   upsampling_depth=hparams['upsampling_depth'],
                                   enc_kernel_size=hparams['enc_kernel_size'],
                                   enc_num_basis=hparams['enc_num_basis'],
                                   num_sources=hparams['max_num_sources'])

numparams = 0
for f in model.parameters():
    if f.requires_grad:
        numparams += f.numel()
experiment.log_parameter('Parameters', numparams)
print('Trainable Parameters: {}'.format(numparams))

model = torch.nn.DataParallel(model).cuda()
opt = torch.optim.Adam(model.parameters(), lr=hparams['learning_rate'])


def apply_output_transform(rec_sources_wavs, input_mix_std,
                           input_mix_mean, input_mom, hparams):
    if hparams["rescale_to_input_mixture"]:
        rec_sources_wavs = (rec_sources_wavs * input_mix_std) + input_mix_mean
    if hparams["apply_mixture_consistency"]:
        rec_sources_wavs = mixture_consistency.apply(rec_sources_wavs, input_mom)
    return rec_sources_wavs

tr_step = 0
val_step = 0
sum_loss = 0.
for i in range(hparams['n_epochs']):
    res_dic = {}
    for d_name in val_losses:
        res_dic[d_name] = {}
        for loss_name in val_losses[d_name]:
            res_dic[d_name][loss_name] = {'mean': 0., 'std': 0., 'acc': []}
    print("Supervised teacher Sudo-RM RF: {} - {} || Epoch: {}/{}".format(
        experiment.get_key(), experiment.get_tags(), i + 1,
        hparams['n_epochs']))
    model.train()
    train_tqdm_gen = tqdm(generators['train'], desc='Training')

    sum_loss = 0.0
    for cnt, (speakers, noise) in enumerate(train_tqdm_gen):
        opt.zero_grad()
        gt_speaker_mix = speakers.sum(1, keepdims=True).cuda()
        noise = noise.cuda()

        input_mix = noise + gt_speaker_mix
        input_mix_std = input_mix.std(-1, keepdim=True)
        input_mix_mean = input_mix.mean(-1, keepdim=True)
        input_mix = (input_mix - input_mix_mean) / (input_mix_std + 1e-9)

        rec_sources_wavs = model(input_mix)
        rec_sources_wavs = apply_output_transform(
            rec_sources_wavs, input_mix_std, input_mix_mean, input_mix, hparams)
        teacher_est_active_speakers = rec_sources_wavs[:, 0:1]
        teacher_est_noises = rec_sources_wavs[:, 1:]

        speaker_l = torch.mean(
            torch.clamp(train_loss(teacher_est_active_speakers, gt_speaker_mix),
                                min=-30., max=+30.))

        noise_l = torch.mean(torch.clamp(train_loss(teacher_est_noises, noise),
                              min=-30., max=+30.))

        l = 0.5 * speaker_l + 0.5 * noise_l
        l.backward()
        if hparams['clip_grad_norm'] > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), hparams['clip_grad_norm'])

        opt.step()

        np_loss_value = l.detach().item()
        sum_loss += np_loss_value
        train_tqdm_gen.set_description(
            f"Training, Running Avg Loss: {round(sum_loss / (cnt + 1), 2)}")
        res_dic['train_total']['sisdr']['acc'] += [- l.detach().cpu()]
        res_dic['train_speaker']['sisdr']['acc'] += [- speaker_l.detach().cpu()]
        res_dic['train_noise']['sisdr']['acc'] += [- noise_l.detach().cpu()]

    if hparams['patience'] > 0:
        if tr_step % hparams['patience'] == 0:
            new_lr = (hparams['learning_rate'] / (hparams['divide_lr_by'] ** (
                    tr_step // hparams['patience'])))
            print('Reducing Learning rate to: {}'.format(new_lr))
            for param_group in opt.param_groups:
                param_group['lr'] = new_lr
    tr_step += 1

    for val_d_name in [x for x in generators if not x == 'train']:
        if generators[val_d_name] is None:
            continue
        if hparams["save_models_every"] > 0 and not tr_step % hparams["save_models_every"] == 0:
            continue
        if val_d_name in ['val_chime_1sp', 'test_chime_1sp']:
            model.eval()
            with torch.no_grad():
                for mixture in tqdm(generators[val_d_name],
                                    desc='Validation on {}'.format(val_d_name)):
                    input_mix = mixture.unsqueeze(1).cuda()

                    input_mix_std = input_mix.std(-1, keepdim=True)
                    input_mix_mean = input_mix.mean(-1, keepdim=True)
                    input_mix = (input_mix - input_mix_mean) / (input_mix_std + 1e-9)

                    student_estimates = model(input_mix)
                    student_estimates = apply_output_transform(
                        student_estimates, input_mix_std, input_mix_mean,
                        input_mix, hparams)

                    s_est_speech = student_estimates[:, 0].detach().cpu().numpy()
                    s_est_speech -= s_est_speech.mean(-1, keepdims=True)
                    s_est_speech /= np.abs(s_est_speech).max(-1, keepdims=True) + 1e-9

                    # Parallelize the DNS-MOS computation.
                    num_of_workers = max(os.cpu_count() // (hparams["n_jobs"] * 2), 1)
                    with Pool(num_of_workers) as p:
                        args_list = [s_est_speech[b_ind]
                                     for b_ind in range(s_est_speech.shape[0])]
                        for dnsmos_values in p.map(compute_dnsmos_process, args_list):
                            for k1, v1 in dnsmos_values.items():
                                res_dic[val_d_name][k1]['acc'].append(v1)
        else:
            model.eval()
            with torch.no_grad():
                for speakers, noise in tqdm(generators[val_d_name],
                                           desc='Validation on {}'.format(val_d_name)):
                    gt_speaker_mix = speakers.sum(1, keepdims=True).cuda()
                    noise = noise.cuda()

                    input_mix = noise + gt_speaker_mix
                    input_mix_std = input_mix.std(-1, keepdim=True)
                    input_mix_mean = input_mix.mean(-1, keepdim=True)
                    input_mix = (input_mix - input_mix_mean) / (input_mix_std + 1e-9)

                    rec_sources_wavs = model(input_mix)
                    rec_sources_wavs = apply_output_transform(
                        rec_sources_wavs, input_mix_std, input_mix_mean, input_mix, hparams)
                    teacher_est_active_speakers = rec_sources_wavs[:, 0:1]
                    teacher_est_noises = rec_sources_wavs[:, 1:]

                    sisdr = - pairwise_neg_sisdr(
                        teacher_est_active_speakers, gt_speaker_mix).detach().cpu()
                    mix_sisdr = sisdr + pairwise_neg_sisdr(
                        input_mix, gt_speaker_mix).detach().cpu()
                    res_dic[val_d_name]['sisdr']['acc'] += sisdr.tolist()
                    res_dic[val_d_name]['sisdri']['acc'] += mix_sisdr.tolist()

            if hparams["log_audio"]:
                audio_logger.log_sp_enh_batch(
                    teacher_est_active_speakers.detach(),
                    teacher_est_noises.detach(),
                    gt_speaker_mix.detach(),
                    noise.detach(),
                    input_mix.detach(),
                    experiment, step=val_step, tag=val_d_name, max_batch_items=4)

    val_step += 1

    if hparams["save_models_every"] > 0 and not tr_step % hparams["save_models_every"] == 0:
        for d_name in res_dic:
            for loss_name in res_dic[d_name]:
                res_dic[d_name][loss_name]['acc'] = []
    else:
        res_dic = cometml_logger.report_losses_mean_and_std(
            res_dic, experiment, tr_step, val_step)

        for d_name in res_dic:
            for loss_name in res_dic[d_name]:
                res_dic[d_name][loss_name]['acc'] = []
        pprint(res_dic)

    if hparams["save_models_every"] > 0:
        if tr_step % hparams["save_models_every"] == 0:
            torch.save(
                model.module.cpu().state_dict(),
                os.path.join(checkpoint_storage_path,
                             f"sup_teacher_epoch_{tr_step}.pt"),
            )
            # Restore the model in the proper device.
            model = model.cuda()
