"""!
@brief Running an experiment with RemixIT for unsupervised domain adaptation
for one- and multi-speaker speech enhancement as described in:

Tzinis, E., Adi, Y., Ithapu, V.K., Xu, B., Smaragdis, P. and Kumar, A.
RemixIT: Continual self-training of speech enhancement models via bootstrapped remixing.
IEEE Journal of Selected Topics in Signal Processing, 16(6), 2022, pp.1329-1341.

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
from asteroid.losses import pairwise_neg_snr


args = parser.get_args()
hparams = vars(args)
generators = dataset_setup.unsupervised_setup(hparams)


def get_new_student(hparams, depth_growth):
    student = improved_sudormrf.SuDORMRF(
        out_channels=hparams["out_channels"],
        in_channels=hparams["in_channels"],
        num_blocks=int(depth_growth * hparams["num_blocks"]),
        upsampling_depth=hparams["upsampling_depth"],
        enc_kernel_size=hparams["enc_kernel_size"],
        enc_num_basis=hparams["enc_num_basis"],
        num_sources=2,
    )
    return student


def freeze_model(model):
    for f in model.parameters():
        if f.requires_grad:
            f.requires_grad = False


def apply_output_transform(rec_sources_wavs, input_mix_std,
                           input_mix_mean, input_mom, hparams):
    if hparams["rescale_to_input_mixture"]:
        rec_sources_wavs = (rec_sources_wavs * input_mix_std) + input_mix_mean
    if hparams["apply_mixture_consistency"]:
        rec_sources_wavs = mixture_consistency.apply(rec_sources_wavs, input_mom)
    return rec_sources_wavs


def normalize_waveform(x):
    return (x - x.mean(-1, keepdim=True)) / (x.std(-1, keepdim=True) + 1e-9)


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

# Get initial teacher and student models
student = get_new_student(hparams, depth_growth=1)
teacher = get_new_student(hparams, depth_growth=1)
teacher.load_state_dict(torch.load(hparams["warmup_checkpoint"]))
student = torch.nn.DataParallel(student).cuda()
teacher = torch.nn.DataParallel(teacher).cuda()
freeze_model(teacher)

opt = torch.optim.Adam(student.parameters(), lr=hparams['learning_rate'])

initial_seed = 17

tr_step = 0
val_step = 0
sum_loss = 0.
student_step = 1
student_order = 1
for i in range(hparams['n_epochs']):
    # Set seeds for reproducability
    torch.manual_seed(initial_seed + i)
    np.random.seed(initial_seed + i)

    res_dic = {}
    for d_name in val_losses:
        res_dic[d_name] = {}
        for loss_name in val_losses[d_name]:
            res_dic[d_name][loss_name] = {'mean': 0., 'std': 0., 'acc': []}
    print("RemixIT w Sudo-RM-RF: {} - {} | Epoch: {}/{} | St step: {}".format(
          experiment.get_key(), experiment.get_tags(), i + 1, hparams['n_epochs'],
          student_step))

    # Figure out which student order is and replace teacher if needed
    if hparams["n_epochs_teacher_update"] is not None:
        if i // hparams["n_epochs_teacher_update"] + 1 > student_order:
            # Replace old teacher with the newest student and update order
            del teacher
            teacher = student.module.cpu()
            del student
            old_student_depth = hparams["student_depth_growth"] ** (student_order - 1)
            new_student_growth = hparams["student_depth_growth"] ** student_order
            student = get_new_student(hparams, depth_growth=new_student_growth)
            student = torch.nn.DataParallel(student).cuda()
            teacher = torch.nn.DataParallel(teacher).cuda()
            opt = torch.optim.Adam(student.parameters(), lr=hparams["learning_rate"])
            print(f"Replaced old teacher with latest student: {old_student_depth} -> {new_student_growth}")
            student_step = 1
            student_order = i // hparams["n_epochs_teacher_update"]

    student.train()
    teacher.eval()
    train_tqdm_gen = tqdm(generators['train'], desc='Training')

    sum_loss = 0.0
    for cnt, input_mix in enumerate(train_tqdm_gen):
        opt.zero_grad()

        input_mix = input_mix.unsqueeze(1).cuda()
        input_mix_std = input_mix.std(-1, keepdim=True)
        input_mix_mean = input_mix.mean(-1, keepdim=True)
        input_mix = (input_mix - input_mix_mean) / (input_mix_std + 1e-9)

        with torch.no_grad():
            # Teacher's estimates
            teacher_estimates = teacher(input_mix).detach()
            teacher_estimates = apply_output_transform(
                teacher_estimates, input_mix_std, input_mix_mean, input_mix, hparams)
            t_est_speech, t_est_noise = teacher_estimates[:, 0:1], teacher_estimates[:, 1:]
            batch_size, n_noises, _ = t_est_noise.shape

            # Bootstrapped remixing
            permuted_t_est_noise = t_est_noise[torch.randperm(batch_size)]
            # permuted_t_est_noise -= permuted_t_est_noise.mean(-1, keepdim=True)
            # t_est_speech -= t_est_speech.mean(-1, keepdim=True)
            bootstrapped_mix = t_est_speech + permuted_t_est_noise

            bootstrapped_mix_std = bootstrapped_mix.std(-1, keepdim=True)
            bootstrapped_mix_mean = bootstrapped_mix.mean(-1, keepdim=True)
            bootstrapped_mix = (bootstrapped_mix - bootstrapped_mix_mean) / (
                    bootstrapped_mix_std + 1e-9)

        # Apply the student model and regress over teacher's estimates
        student_estimates = student(bootstrapped_mix)
        student_estimates = apply_output_transform(
                student_estimates, bootstrapped_mix_std, bootstrapped_mix_mean,
                bootstrapped_mix, hparams)
        s_est_speech, s_est_noise = student_estimates[:, 0:1], student_estimates[:, 1:]

        # Regress over the teacher estimated speech and the permuted noise estiamtes
        speaker_l = torch.mean(
            torch.clamp(train_loss(s_est_speech, t_est_speech.detach()),
                        min=-30., max=+30.))

        noise_l = torch.mean(
            torch.clamp(train_loss(s_est_noise, permuted_t_est_noise.detach()),
                        min=-30., max=+30.))

        l = 0.5 * speaker_l + 0.5 * noise_l
        l.backward()
        if hparams['clip_grad_norm'] > 0:
            torch.nn.utils.clip_grad_norm_(student.parameters(), hparams['clip_grad_norm'])

        opt.step()

        np_loss_value = l.detach().item()
        sum_loss += np_loss_value
        train_tqdm_gen.set_description(
            f"Training - Avg Loss: {round(sum_loss / (cnt + 1), 2)} ")
        res_dic['train_total']['sisdr']['acc'] += [- l.detach().cpu()]
        res_dic['train_speaker']['sisdr']['acc'] += [- speaker_l.detach().cpu()]
        res_dic['train_noise']['sisdr']['acc'] += [- noise_l.detach().cpu()]

    if hparams['patience'] > 0:
        if student_step % hparams['patience'] == 0:
            new_lr = (hparams['learning_rate'] / (hparams['divide_lr_by'] ** (
                      student_step // hparams['patience'])))
            print('Reducing Learning rate to: {}'.format(new_lr))
            for param_group in opt.param_groups:
                param_group['lr'] = new_lr
    tr_step += 1
    student_step += 1

    for val_d_name in [x for x in generators if not x == 'train']:
        if generators[val_d_name] is not None and val_set in ['val_chime_1sp', 'test_chime_1sp']:
            student.eval()
            with torch.no_grad():
                for mixture in tqdm(generators[val_d_name],
                                    desc='Validation on {}'.format(val_d_name)):
                    input_mix = mixture.unsqueeze(1).cuda()

                    input_mix_std = input_mix.std(-1, keepdim=True)
                    input_mix_mean = input_mix.mean(-1, keepdim=True)
                    input_mix = (input_mix - input_mix_mean) / (input_mix_std + 1e-9)

                    student_estimates = student(input_mix)
                    student_estimates = apply_output_transform(
                        student_estimates, input_mix_std, input_mix_mean,
                        input_mix, hparams)
                    new_mix = student_estimates[:, 0:1] + student_estimates[:, 1:]
                    new_mix_std = new_mix.std(-1, keepdim=True)
                    new_mix_mean = new_mix.mean(-1, keepdim=True)
                    student_estimates = (student_estimates - new_mix_mean) / (new_mix_std + 1e-9)
                    s_est_speech = student_estimates[:, 0].detach().cpu().numpy()

                    for b_ind in range(s_est_speech.shape[0]):
                        dnsmos_val = dnnmos_metric.compute_dnsmos(s_est_speech[b_ind], fs=16000)
                        for k, v in dnsmos_val.items():
                            res_dic[val_d_name][k]['acc'].append(v)

            if hparams["log_audio"]:
                audio_logger.log_sp_enh_no_gt_batch(
                    student_estimates[:, 0:1].detach(),
                    student_estimates[:, 1:2].detach(),
                    input_mix.detach(),
                    experiment, step=val_step, tag=f"{val_d_name}_stud_{student_order}",
                    max_batch_items=4)

    pprint(res_dic)
    val_step += 1

    res_dic = cometml_logger.report_losses_mean_and_std(
        res_dic, experiment, tr_step, val_step)

    for d_name in res_dic:
        for loss_name in res_dic[d_name]:
            res_dic[d_name][loss_name]['acc'] = []
    pprint(res_dic)

    if hparams["save_models_every"] > 0:
        if tr_step % hparams["save_models_every"] == 0:
            torch.save(
                student.module.cpu().state_dict(),
                os.path.join(
                    checkpoint_storage_path,
                    f"remixit_student_{student_order}_epoch_{student_step}_global_{tr_step}.pt"),
            )
            # Restore the model in the proper device.
            student = student.cuda()



