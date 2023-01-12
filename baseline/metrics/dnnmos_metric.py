"""
Copied from https://github.com/UDASE-CHiME2023/baseline/blob/main/metrics/dnnmos_metric.py
Author: Manuel Pariente

Functional version of https://github.com/microsoft/DNS-Challenge/blob/master/DNSMOS/dnsmos_local.py
Licences : 
- https://github.com/microsoft/DNS-Challenge/blob/master/LICENSE
- https://github.com/microsoft/DNS-Challenge/blob/master/LICENSE-CODE

Corresponding paper : https://arxiv.org/pdf/2110.01763.pdf
"""

import os
import numpy as np
import librosa
import onnxruntime as ort
import numpy.polynomial.polynomial as poly

# Coefficients for polynomial fitting
COEFS_SIG = np.array([9.651228012789436761e-01, 6.592637550310214145e-01, 7.572372955623894730e-02])
COEFS_BAK = np.array([-3.733460011101781717e+00,2.700114234092929166e+00, -1.721332907340922813e-01]) 
COEFS_OVR = np.array([8.924546794696789354e-01, 6.609981731940616223e-01, 7.600269530243179694e-02])

SIG_MODEL_PATH = "local/sig.onnx"
BAK_OVR_MODEL_PATH = "local/bak_ovr.onnx"


def audio_logpowspec(audio, nfft=320, hop_length=160):
    powspec = (np.abs(librosa.core.stft(audio, n_fft=nfft, hop_length=hop_length)))**2
    logpowspec = np.log10(np.maximum(powspec, 10**(-12)))
    return logpowspec.T


# Function version of dnsmos_local.py with cached sessions
def compute_dnsmos(audio, fs=16000, input_length=9, _sessions={}):
    """ Compute the DNSMOS P.835 metric"""
    # Cache the sessions
    base = os.path.dirname(compute_dnsmos.__code__.co_filename)
    if not _sessions:
        _sessions.update({
            'sig': ort.InferenceSession(os.path.join(base, SIG_MODEL_PATH)),
            'bak_ovr': ort.InferenceSession(os.path.join(base, BAK_OVR_MODEL_PATH))
        })
    if len(audio) < 2*fs:
        print('Audio clip is too short. Skipped processing ')
        return 0
    if fs != 16000:
        print("DNS-MOS only works at 16kHZ, resampling.")
        audio = librosa.core.resample(audio, fs, 16000)
    new_fs = 16000

    len_samples = int(input_length*new_fs)
    while len(audio) < len_samples:
        audio = np.append(audio, audio)
    
    num_hops = int(np.floor(len(audio)/new_fs) - input_length)+1
    hop_len_samples = new_fs
    predicted_mos_sig_seg = []
    predicted_mos_bak_seg = []
    predicted_mos_ovr_seg = []

    for idx in range(num_hops):
        audio_seg = audio[int(idx*hop_len_samples) : int((idx+input_length)*hop_len_samples)]
        input_features = np.array(audio_logpowspec(audio=audio_seg)).astype('float32')[np.newaxis,:,:]

        onnx_inputs_sig = {inp.name: input_features for inp in _sessions["sig"].get_inputs()}
        mos_sig = poly.polyval(_sessions["sig"].run(None, onnx_inputs_sig), COEFS_SIG)
            
        onnx_inputs_bak_ovr = {inp.name: input_features for inp in _sessions["bak_ovr"].get_inputs()}
        mos_bak_ovr = _sessions["bak_ovr"].run(None, onnx_inputs_bak_ovr)

        mos_bak = poly.polyval(mos_bak_ovr[0][0][1], COEFS_BAK)
        mos_ovr = poly.polyval(mos_bak_ovr[0][0][2], COEFS_OVR)

        predicted_mos_sig_seg.append(mos_sig)
        predicted_mos_bak_seg.append(mos_bak)
        predicted_mos_ovr_seg.append(mos_ovr)
    
    return dict(
        sig_mos=np.mean(predicted_mos_sig_seg),
        bak_mos=np.mean(predicted_mos_bak_seg),
        ovr_mos=np.mean(predicted_mos_ovr_seg)
        )
