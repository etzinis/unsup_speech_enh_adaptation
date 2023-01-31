# Unsupervised domain adaptation for speech enhancement with RemixIT
CHiME 2023 task: Unsupervised domain adaptation for conversational speech enhancement baseline

We pre-train a supervised Sudo rm- rf [1, 2] teacher on some out-of-domain data (e.g. Libri1to3mix) and try to adapt a student model with the RemixIT [3] method on the unlabeled CHiME data.

### Results under mean aggregation on single-speaker segments of the CHiME-5 test dataset - 3130 files.
We use peak normalization at the waveform which needs to be evaluated). OOD stands for out-of-domain, EMA stands for expontially moving average teacher and SU stands for the sequentially updated teacher model. VAD annotations means that the model was adapted using only the CHiME training data where the output of the VAD corresponds to at least one speaker active.

|                        Method                          | OVR_MOS | BAK_MOS | SIG_MOS | [Checkpoint](https://github.com/etzinis/unsup_speech_enh_adaptation/tree/main/pretrained_checkpoints) |
| ---------------------------------------------------- | ------- | ------- | ------- | --------- |
| unprocessed                                          |  2.73   |     2.35  |   **3.64**    | |
| Sudo rm -rf (fully-supervised OOD teacher) |   2.81    |   3.25   |   3.46    | ```libri1to3mix_supervised_teacher_w_mixconsist.pt```
| RemixIT (self-supervised student with SU teacher |   **2.83**   |   3.28   |  3.43    | ```remixit_chime_adapted_student_sequentially_updated_teacher_ep_33.py```
| RemixIT (self-supervised student with EMA teacher |    2.75   |   **3.35**   |   3.30   | ```remixit_chime_adapted_student_besmos_ep35.pt```
| RemixIT (self-supervised student with EMA teacher) + VAD annotations    |    2.78   |   **3.35**    |   3.33    | ```remixit_chime_adapted_student_bestbak_ep85_using_vad.pt```


## Table of contents

- [Datasets Generation](#datasets-generation)
- [Repo and paths Configurations](#repo-and-paths-configurations)
- [How to train the supervised teacher](#how-to-train-the-supervised-teacher)
- [How to adapt the RemixIT student](#how-to-adapt-the-remixit-student)
- [How to load a pretrained checkpoint](#how-to-load-a-pretrained-checkpoint)
- [How to evaluate a checkpoint](#how-to-evaluate-a-checkpoint)
- [References](#references)

## Datasets generation
Two datasets are required for generation, namely, Libri3mix and CHiME.

For the generation of Libri3Mix one can follow the instructions [here](https://github.com/JorisCos/LibriMix) or just follow this:
```shell
cd {path_to_generate_Libri3mix}
git clone https://github.com/JorisCos/LibriMix
cd LibriMix 
./generate_librimix.sh storage_dir
```

For the generation of the CHiME data follow the instructions [here](https://github.com/UDASE-CHiME2023/unlabeled_data) or just follow these steps (this step requires the existence of CHiME-5 data under some path, [apply-and-get-CHiME5-here](https://chimechallenge.github.io/chime6/download.html)):
```shell
cd {path_to_generate_CHiME_processed_data}
# clone data repository
git clone https://github.com/UDASE-CHiME2023/data.git
cd unlabeled_data

# create CHiME conda environment
conda env create -f environment.yml
conda activate CHiME

# Run the appropriate scripts to create the training, dev and eval datasets
python create_audio_segments.py {insert_path_of_CHiME5_data} json_files {insert_path_of_processed_10s_CHiME5_data} --train_10s

# Create the training data with VAD annotations - might somewhat help with the adaptation
python create_audio_segments.py {insert_path_of_CHiME5_data} json_files {insert_path_of_processed_10s_CHiME5_data} --train_10s --train_vad --train_only
```


## Repo and paths configurations
Set the paths for the aforementioned datasets and include the path of this repo.

```shell
git clone https://github.com/etzinis/unsup_speech_enh_adaptation.git
export PYTHONPATH={the path that you stored the github repo}:$PYTHONPATH
cd unsup_speech_enh_adaptation
python -m pip install --user -r requirements.txt
vim __config__.py
```

You should change the following:
```shell
LIBRI3MIX_ROOT_PATH = '{inset_path_to_Libri3mix}'
CHiME_ROOT_PATH = '{insert_path_of_processed_10s_CHiME5_data}'

API_KEY = 'your_comet_ml_key'
```

## How to train the supervised teacher
Running the out-of-domain supervised teacher with SI-SNR loss is as easy as: 
```shell
cd {the path that you stored the github repo}/baseline
python -Wignore run_sup_ood_pretrain.py --train libri1to3mix --val libri1to3mix libri1to3chime --test libri1to3mix \
-fs 16000 --enc_kernel_size 81 --num_blocks 8 --out_channels 256 --divide_lr_by 3. \
--upsampling_depth 7 --patience 15  -tags supervised_ood_teacher --n_epochs 81 \
--project_name uchime_baseline_v3 --clip_grad_norm 5.0 --save_models_every 10 --audio_timelength 8.0 \
--p_single_speaker 0.5 --min_or_max min --max_num_sources 2 \
--checkpoint_storage_path {insert_path_to_save_models} --log_audio --apply_mixture_consistency \
--n_jobs 12 -cad 2 3 -bs 24
```

Don't forget to set _n_jobs_ to the number of CPUs to use, _cad_ to the cuda ids to be used and _bs_ to the batch size used w.r.t. your system. Also you need to set the _checkpoint_storage_path_ to a valid path.

## How to adapt the RemixIT student
If you want to adapt your model to the CHiME data you can use as a warm-up checkpoint the previous teacher model and perform RemixIT using the CHiME mixture dataset (in order to use the annotated with VAD data just simple use the *--use_vad* at the end of the followin command): 
```shell
cd {the path that you stored the github repo}/baseline
python -Wignore run_remixit.py --train chime --val chime libri1to3chime --test libri1to3mix \
-fs 16000 --enc_kernel_size 81 --num_blocks 8 --out_channels 256 --divide_lr_by 3. \
--student_depth_growth 1 --n_epochs_teacher_update 1 --teacher_momentum 0.99 \
--upsampling_depth 7 --patience 10 --learning_rate 0.0003 -tags remixit student allData \
--n_epochs 100 --project_name uchime_baseline_v3 --clip_grad_norm 5.0 --audio_timelength 8.0 \
--min_or_max min --max_num_sources 2 --save_models_every 1 --initialize_student_from_checkpoint \
--checkpoint_storage_path /home/thymios/UCHIME_checkpoints \
--warmup_checkpoint ../pretrained_checkpoints/libri1to3mix_supervised_teacher_w_mixconsist.pt \
--checkpoint_storage_path {insert_path_to_save_models} --log_audio --apply_mixture_consistency \
--n_jobs 12 -cad 2 3 -bs 24
```

If you want to use a sequentially updated teacher instead of EMA:
```shell
cd {the path that you stored the github repo}/baseline
python -Wignore run_remixit.py --train chime --val chime libri1to3chime --test libri1to3mix \
-fs 16000 --enc_kernel_size 81 --num_blocks 8 --out_channels 256 --divide_lr_by 3. \
--student_depth_growth 1 --n_epochs_teacher_update 10 --teacher_momentum 0.0 \
--upsampling_depth 7 --patience 15 --learning_rate 0.0001 -tags remixit student allData \
--n_epochs 100 --project_name uchime_baseline_v3 --clip_grad_norm 5.0 --audio_timelength 8.0 \
--min_or_max min --max_num_sources 2 --save_models_every 1 --initialize_student_from_checkpoint \
--checkpoint_storage_path /home/thymios/UCHIME_checkpoints \
--warmup_checkpoint ../pretrained_checkpoints/libri1to3mix_supervised_teacher_w_mixconsist.pt \
--checkpoint_storage_path {insert_path_to_save_models} --log_audio --apply_mixture_consistency \
--n_jobs 12 -cad 2 3 -bs 24
```

## How to load a pretrained checkpoint
```python
import baseline.utils.mixture_consistency as mixture_consistency
import baseline.models.improved_sudormrf as improved_sudormrf

model = improved_sudormrf.SuDORMRF(
        out_channels=256,
        in_channels=512,
        num_blocks=8,
        upsampling_depth=7,
        enc_kernel_size=81,
        enc_num_basis=512,
        num_sources=2,
    )
# You can load the state_dict as here:
model.load_state_dict(torch.load('.../unsup_speech_enh_adaptation/pretrained_checkpoints/remixit_chime_adapted_student_bestbak_ep85_using_vad.pt'))
model = torch.nn.DataParallel(model).cuda()

# Scale the input mixture, perform inference and apply mixture consistency
input_mix = input_mix.unsqueeze(1).cuda() 
# input_mix.shape = (batch, 1, time_samples)
input_mix_std = input_mix.std(-1, keepdim=True)
input_mix_mean = input_mix.mean(-1, keepdim=True)
input_mix = (input_mix - input_mix_mean) / (input_mix_std + 1e-9)

estimates = model(input_mix)
estimates = mixture_consistency.apply(estimates, input_mix)
```

## How to evaluate a checkpoint
If you want to perform full evaluation of a pre-trained checkpoint simply use our script: 
```shell
cd {the path that you stored the github repo}/baseline/utils
python -Wignore final_eval.py --model_checkpoint ../../pretrained_checkpoints/remixit_chime_adapted_student_bestbak_ep85_using_vad.pt \
--save_results_dir ./ --normalize_with_max_absolute_value --dataset_split eval
```


## References

[1] Tzinis, E., Wang, Z., & Smaragdis, P. (2020, September). Sudo rm-rf: Efficient networks for universal audio source separation. In 2020 IEEE 30th International Workshop on Machine Learning for Signal Processing (MLSP). <https://arxiv.org/abs/2007.06833>

[2] Tzinis, E., Wang, Z., Jiang, X., and Smaragdis, P., Compute and memory efficient universal sound source separation. In Journal of Signal Processing Systems, vol. 9, no. 2, pp. 245–259, 2022, Springer. <https://arxiv.org/pdf/2103.02644.pdf>

[3] Tzinis, E., Adi, Y., Ithapu, V. K., Xu, B., Smaragdis, P., & Kumar, A. (October, 2022). RemixIT: Continual self-training of speech enhancement models via bootstrapped remixing. In IEEE Journal of Selected Topics in Signal Processing. <https://arxiv.org/abs/2202.08862>
