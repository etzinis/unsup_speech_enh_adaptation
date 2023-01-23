"""!
@brief Infer Dataset Specific parameters and return generators

@author Efthymios Tzinis {etzinis2@illinois.edu}
@copyright University of Illinois at Urbana-Champaign
"""

import baseline.dataset_loaders.libri1to3mix as libri1to3mix
import baseline.dataset_loaders.reverb_libri1to3mix as reverb_libri1to3mix
import baseline.dataset_loaders.libri1to3chime as libri1to3chime
import baseline.dataset_loaders.chime as chime


def create_loader_for_simple_dataset(dataset_name=None,
                                     hparams=None,
                                     fixed_n_sources=None,
                                     n_speakers_priors=None,
                                     split=None,
                                     get_only_active_speakers=False,
                                     n_samples=-1):
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
            normalize_audio=False, n_samples=n_samples,
            n_speakers_priors=n_speakers_priors)
    elif dataset_name == 'reverb_libri1to3mix':
        if split == "train":
            this_dataset_split = "train-360"
        elif split == "val":
            this_dataset_split = "dev"
        else:
            this_dataset_split = split
        data_loader = reverb_libri1to3mix.Dataset(
            sample_rate=hparams['fs'], fixed_n_sources=fixed_n_sources,
            timelength=hparams['audio_timelength'],
            augment='train' in split, zero_pad=True,
            min_or_max=hparams['min_or_max'], split=this_dataset_split,
            normalize_audio=False, n_samples=n_samples,
            n_speakers_priors=n_speakers_priors)
    elif dataset_name == 'libri1to3chime':
        if split == "val":
            this_dataset_split = "dev"
        else:
            this_dataset_split = split
        data_loader = libri1to3chime.Dataset(
            sample_rate=hparams['fs'], fixed_n_sources=fixed_n_sources,
            timelength=hparams['audio_timelength'],
            augment='train' in split, zero_pad=True,
            get_only_active_speakers=get_only_active_speakers,
            min_or_max=hparams['min_or_max'], split=this_dataset_split,
            n_speakers_priors=n_speakers_priors,
            normalize_audio=False, n_samples=n_samples)
    elif dataset_name == 'chime':
        if split == "test":
            this_dataset_split = "eval"
        elif split == "val":
            this_dataset_split = "dev"
        else:
            this_dataset_split = split
        data_loader = chime.Dataset(
            sample_rate=hparams['fs'], fixed_n_sources=fixed_n_sources,
            timelength=hparams['audio_timelength'],
            augment='train' in split, zero_pad=True, use_vad=hparams['use_vad'],
            get_only_active_speakers=get_only_active_speakers, split=this_dataset_split,
            normalize_audio=False, n_samples=n_samples)
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

        for dataset_name in hparams[data_split]:

            if data_split == "train":
                p_single_speaker = hparams['p_single_speaker']
                p_multispeaker = 1. - hparams['p_single_speaker']
                data_loader = create_loader_for_simple_dataset(
                        dataset_name=dataset_name,
                        hparams=hparams,
                        fixed_n_sources=-1,
                        n_speakers_priors=[
                            p_single_speaker, p_multispeaker / 2, p_multispeaker / 2],
                        split=data_split)
                this_dataset_name = data_split
                generators[this_dataset_name] = data_loader.get_generator(
                    batch_size=hparams['batch_size'], num_workers=hparams['n_jobs'])
            else:
                if dataset_name == 'chime':
                    for fixed_n_sources in [1]:
                        data_loader = create_loader_for_simple_dataset(
                            dataset_name=dataset_name,
                            hparams=hparams,
                            fixed_n_sources=fixed_n_sources,
                            get_only_active_speakers=False,
                            split=data_split,
                            n_samples=250)

                        this_dataset_name = f"{data_split}_{dataset_name}_{fixed_n_sources}sp"
                        generators[this_dataset_name] = data_loader.get_generator(
                            batch_size=hparams['batch_size'], num_workers=hparams['n_jobs'])
                else:
                    # non-chime
                    for fixed_n_sources in [1, 2, 3]:
                        data_loader = create_loader_for_simple_dataset(
                            dataset_name=dataset_name,
                            hparams=hparams,
                            fixed_n_sources=fixed_n_sources,
                            n_speakers_priors=[0.34, 0.33, 0.33],
                            split=data_split)

                        this_dataset_name = f"{data_split}_{dataset_name}_{fixed_n_sources}sp"
                        generators[this_dataset_name] = data_loader.get_generator(
                            batch_size=hparams['batch_size'], num_workers=hparams['n_jobs'])

    return generators


def unsupervised_setup(hparams):
    # Create all generators
    generators = {}
    for data_split in ['train', 'val', 'test']:
        if hparams[data_split] is None:
            generators[data_split] = None
            continue

        for dataset_name in hparams[data_split]:
            if data_split == "train":
                data_loader = create_loader_for_simple_dataset(
                        dataset_name=dataset_name,
                        hparams=hparams,
                        fixed_n_sources=-1,
                        get_only_active_speakers=False,
                        split=data_split)
                this_dataset_name = data_split
                generators[this_dataset_name] = data_loader.get_generator(
                    batch_size=hparams['batch_size'], num_workers=hparams['n_jobs'])
            else:
                # This is only for validation and testing
                if dataset_name == 'chime':
                    for fixed_n_sources in [1]:
                        data_loader = create_loader_for_simple_dataset(
                            dataset_name=dataset_name,
                            hparams=hparams,
                            fixed_n_sources=fixed_n_sources,
                            get_only_active_speakers=False,
                            split=data_split,
                            n_samples=250)

                        this_dataset_name = f"{data_split}_{dataset_name}_{fixed_n_sources}sp"
                        generators[this_dataset_name] = data_loader.get_generator(
                            batch_size=hparams['batch_size'], num_workers=hparams['n_jobs'])
                else:
                    # non-chime
                    for fixed_n_sources in [1, 2, 3]:
                        data_loader = create_loader_for_simple_dataset(
                            dataset_name=dataset_name,
                            hparams=hparams,
                            fixed_n_sources=fixed_n_sources,
                            n_speakers_priors=[0.34, 0.33, 0.33],
                            split=data_split)

                        this_dataset_name = f"{data_split}_{dataset_name}_{fixed_n_sources}sp"
                        generators[this_dataset_name] = data_loader.get_generator(
                            batch_size=hparams['batch_size'], num_workers=hparams['n_jobs'])

    return generators
