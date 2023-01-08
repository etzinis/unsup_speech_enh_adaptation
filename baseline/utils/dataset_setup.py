"""!
@brief Infer Dataset Specific parameters and return generators

@author Efthymios Tzinis {etzinis2@illinois.edu}
@copyright University of Illinois at Urbana-Champaign
"""

import baseline.dataset_loaders.libri1to3mix as libri1to3mix


def create_loader_for_simple_dataset(dataset_name=None,
                                     hparams=None,
                                     fixed_n_sources=None,
                                     n_speakers_priors=None,
                                     split=None):
    if dataset_name == 'libri1to3mix':
        if split == "train":
            this_dataset_split = "train-360"
        elif split == "val":
            this_dataset_split = "dev"
        else:
            this_dataset_split = split
        data_loader = libri1to3mix.Dataset(
            sample_rate=hparams['fs'], fixed_n_sources=fixed_n_sources,
            timelength=hparams['audio_timelength'],
            augment='train' in split, zero_pad=True,
            min_or_max=hparams['min_or_max'], split=this_dataset_split,
            normalize_audio=False, n_samples=-1,
            n_speakers_priors=n_speakers_priors)
    else:
        raise ValueError('Dataset: {} is not yet supported!'.format(
            dataset_name))

    return data_loader


def supervised_setup(hparams):
    # Create all generators
    generators = {}
    for data_split in ['train', 'val', 'test']:
        if hparams[data_split] is None:
            generators[data_split] = None
            continue

        if len(hparams[data_split]) > 1:
            raise ValueError('Current implementation does not support '
                             'training using multiple datasets.')

        dataset_name = hparams[data_split][0]

        if data_split == "train":
            p_single_speaker = hparams['p_single_speaker']
            p_multispeaker = 1. - hparams['p_single_speaker']
            data_loader = create_loader_for_simple_dataset(
                    dataset_name=hparams[data_split][0],
                    hparams=hparams,
                    fixed_n_sources=-1,
                    n_speakers_priors=[
                        p_single_speaker, p_multispeaker / 2, p_multispeaker / 2],
                    split=data_split)
            this_dataset_name = data_split
            generators[this_dataset_name] = data_loader.get_generator(
                batch_size=hparams['batch_size'], num_workers=hparams['n_jobs'])
        else:
            # This is only for validation and testing
            for fixed_n_sources in [1, 2, 3]:
                data_loader = create_loader_for_simple_dataset(
                    dataset_name=hparams[data_split][0],
                    hparams=hparams,
                    fixed_n_sources=fixed_n_sources,
                    n_speakers_priors=[0.34, 0.33, 0.33],
                    split=data_split)

                this_dataset_name = f"{data_split}_{dataset_name}_{fixed_n_sources}sp"
                generators[this_dataset_name] = data_loader.get_generator(
                    batch_size=hparams['batch_size'], num_workers=hparams['n_jobs'])

    return generators
