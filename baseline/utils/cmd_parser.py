"""!
@brief Argument parser for all experiments.

Inspired from: https://github.com/etzinis/fedenhance/blob/master/fedenhance/experiments/utils/improved_cmd_args_parser.py

@author Efthymios Tzinis {etzinis2@illinois.edu}
@copyright University of Illinois at Urbana-Champaign
"""

import argparse


def tuple_availavle_speech(s):
    try:
        x, y = map(float, s.split(","))
        return (x, y)
    except Exception as e:
        print(e)
        raise argparse.ArgumentTypeError("Tuples must be in the form x,y")


def get_args():
    """! Command line parser"""
    parser = argparse.ArgumentParser(description="Experiment Argument Parser")
    # ===============================================
    # Datasets arguments
    parser.add_argument(
        "--train",
        type=str,
        help="""The dataset(s) used for training.""",
        nargs="+",
        choices=["libri1to3mix", "chime", "reverb_libri1to3mix"],
    )
    parser.add_argument(
        "--val",
        type=str,
        help="""The dataset(s) used for validation.""",
        default=["libri1to3mix"],
        nargs="*",
        choices=["libri1to3mix", "chime", "libri1to3chime", "reverb_libri1to3mix"],
    )
    parser.add_argument(
        "--test",
        type=str,
        help="""The dataset(s) used for testing.""",
        default=["libri1to3mix"],
        nargs="*",
        choices=["libri1to3mix", "chime", "libri1to3chime", "reverb_libri1to3mix"],
    )
    parser.add_argument(
        "--audio_timelength",
        type=float,
        help="""The timelength of the audio that you want
                                to load in seconds.""",
        default=4.0,
    )
    parser.add_argument(
        "--p_single_speaker",
        type=float,
        help="""The probability of training with single
                speaker noisy speech samples.""",
        default=0.34,
    )
    parser.add_argument(
        "--use_vad",
        action='store_true',
        help="""Use the output of the VAD model to get only active speaker segments when training with the CHiME 
        data.""",
        default=False,
    )
    parser.add_argument(
        "-fs", type=int, help="""Sampling rate of the audio.""", default=16000
    )
    parser.add_argument(
        "--min_or_max",
        type=str,
        help="""The min or max version of Libri and WSJ-like dataset""",
        default=["min"],
        choices=["min", "max"],
    )
    # ===============================================
    # Separation task arguments
    parser.add_argument(
        "--min_num_sources",
        type=int,
        help="""The minimum number of sources in a mixture.""",
        default=1,
    )
    parser.add_argument(
        "--max_num_sources",
        type=int,
        help="""The maximum number of sources in a mixture.""",
        default=2,
    )

    # ===============================================
    # Teacher / Student parameters in RemixIT
    parser.add_argument(
        "--initialize_student_from_checkpoint",
        action='store_true',
        help="""When set RemixIT's student will get initialized from the same
            checkpoint as the student. Needs to be used with momentum.""",
        default=False,
    )

    parser.add_argument(
        "--teacher_momentum",
        type=float,
        help=""" If this is set to higher than 0, that means that RemixIT's
        student needs to be the same as the teacher and get initialized from the same
        checkpoint. A good value would be somewhere near the region of [0.9-0.99].
        The momentum of the teacher weight model updates
        in the update of the form t_w = momentum * t_w + (1 - momentum) * s_w.""",
        default=0.,
    )

    parser.add_argument(
        "--n_epochs_teacher_update",
        type=int,
        help="""The number of epochs that the teacher model is going to get updated
        using the weights from the student model.""",
        default=None,
    )

    parser.add_argument(
        "--student_depth_growth",
        type=float,
        help="""The growth factor for the student model for the number of UConv Blocks.""",
        default=1,
    )

    # ===============================================
    # Training parameters
    parser.add_argument(
        "--rescale_to_input_mixture",
        action='store_true',
        help="""Rescale the output estimates using the mean and the std of the input mixture.""",
        default=False,
    )
    parser.add_argument(
        "--apply_mixture_consistency",
        action='store_true',
        help="""Use/ no use mixture consistency at the output of the models.""",
        default=False,
    )
    parser.add_argument(
        "-bs",
        "--batch_size",
        type=int,
        help="""The number of samples in each batch.
                Warning: Cannot be less than the number of
                the validation samples""",
        default=4,
    )
    parser.add_argument(
        "--n_epochs",
        type=int,
        help="""The number of epochs that the model will be trained for.""",
        default=500,
    )
    parser.add_argument(
        "--uniform_snr",
        type=float,
        nargs="+",
        help="""SNR range to augment the samples in supervised cases.""",
        default=[None, None],
    )
    parser.add_argument(
        "-lr",
        "--learning_rate",
        type=float,
        help="""Initial Learning rate""",
        default=1e-3,
    )
    parser.add_argument(
        "--divide_lr_by",
        type=float,
        help="""The factor that the learning rate
                            would be divided by""",
        default=2.0,
    )
    parser.add_argument(
        "--patience",
        type=int,
        help="""Patience until reducing the learning rate .""",
        default=10,
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        help="""The optimizer that you want to use""",
        default="adam",
        choices=["adam"],
    )
    parser.add_argument(
        "--clip_grad_norm",
        type=float,
        help="""The norm value which all gradients
        are going to be clipped, 0 means that no
        grads are going to be clipped""",
        default=5.0,
    )
    parser.add_argument(
        "--max_abs_snr",
        type=float,
        help="""The maximum absolute value of the SNR of
                                the mixtures.""",
        default=5.0,
    )
    parser.add_argument(
        "--warmup_checkpoint",
        type=str,
        help="""The absolute path of a pre-trained separation model
        that will be used for warm start for the teacher network.""",
        default=None,
    )
    # ===============================================
    parser.add_argument(
        "-tags", "--cometml_tags",
        type=str,
        nargs="+",
        help="""A list of tags for the cometml experiment.""",
        default=[])
    parser.add_argument(
        "--experiment_name",
        type=str,
        help="""Name of current experiment""",
        default=None,
    )
    parser.add_argument(
        "--project_name",
        type=str,
        help="""Name of current experiment""",
        default="yolo_experiment")
    parser.add_argument(
        "--save_models_every",
        type=int,
        help="""Save models every how many epochs.""",
        default=0,
    )
    parser.add_argument(
        "--checkpoint_storage_path",
        type=str,
        help="""The absolute path for storing the model checkpoints.""",
        default=None,
    )
    parser.add_argument(
        "--full_eval_every",
        type=int,
        help="""The number of epochs per which STOI and PESQ are also going to be
        evaluated. Be aware that those metrics need to be computed on CPU and are
        going to be time consuming.""",
        default=5,
    )
    parser.add_argument("--log_audio", action='store_true',
                        help="""Save some example audio files.""",
                        default=False)
    # ===============================================
    # Device params
    parser.add_argument(
        "-cad",
        "--cuda_available_devices",
        type=str,
        nargs="+",
        help="""A list of Cuda IDs that would be
                            available for running this experiment""",
        default=["0"],
        choices=["0", "1", "2", "3", "4", "5", "6", "7"],
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        help="""The number of cpu workers for loading the data, etc.""",
        default=4,
    )

    # ===============================================
    # Separation model (SuDO-RM-RF) params
    parser.add_argument(
        "--out_channels",
        type=int,
        help="The number of channels of the internal "
        "representation outside the U-Blocks.",
        default=512,
    )
    parser.add_argument(
        "--in_channels",
        type=int,
        help="The number of channels of the internal "
        "representation inside the U-Blocks.",
        default=512,
    )
    parser.add_argument(
        "--num_blocks", type=int, help="Number of the successive U-Blocks.", default=8
    )
    parser.add_argument(
        "--upsampling_depth",
        type=int,
        help="Number of successive upsamplings and "
        "effectively downsampling inside each U-Block. "
        "The aggregation of all scales is performed by "
        "addition.",
        default=7,
    )
    parser.add_argument(
        "--group_size",
        type=int,
        help="The number of individual computation groups "
        "applied if group communication module is used.",
        default=16,
    )
    parser.add_argument(
        "--enc_kernel_size",
        type=int,
        help="The width of the encoder and decoder kernels.",
        default=41,
    )
    parser.add_argument(
        "--enc_num_basis",
        type=int,
        help="Number of the encoded basis representations.",
        default=512,
    )

    return parser.parse_args()
