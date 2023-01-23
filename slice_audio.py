#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import os
import soundfile as sf
import argparse
from pathlib import Path
from tqdm import tqdm
import soundfile as sf
import numpy as np


def string_to_float_list(s):
    return [float(f) for f in s[1:-1].split(sep=', ')]

def parse_args():
    parser=argparse.ArgumentParser(description=''' 
        Slice the audio files into several chunk.
        ''')

    parser.add_argument("data_dir", help="Data directory containing the audio files to slice.", type=Path)
    parser.add_argument("out_dir", help="Output directory to store the chunks.", type=Path)
    parser.add_argument("--max_chunk_time", help="Maximum chunk time in seconds.", 
                        type=float, default=None)
    parser.add_argument("--brouhaha_vad_snr", help="Brouhaha result file, used to select only the chunks where speech is detected (works only on CHiME train dataset).", 
                        type=Path, default=None)


    args=parser.parse_args()
    return args

def main():
    # load arguments
    args = parse_args()
    print(args)
    data_dir = args.data_dir
    max_chunk_time = 10000 if args.max_chunk_time is None else args.max_chunk_time
    out_dir = args.out_dir
    if not args.data_dir.exists():
        raise ValueError("data_dir doesn't exist.")
    if not args.out_dir.exists():
        os.makedirs(args.out_dir)
    
    if args.brouhaha_vad_snr is not None:
        if not args.brouhaha_vad_snr.exists():
            raise ValueError("brouhaha_vad_snr doesn't exist.")
        brouhaha_results = pd.read_csv(args.brouhaha_vad_snr, index_col=0)
        brouhaha_results = brouhaha_results.applymap(string_to_float_list)
        brouhaha_sr = 1/0.016875

    # search for audio files
    files = []
    for file in os.listdir(data_dir):
        if file.endswith('.wav') and 'CH' not in file:
            files.append(os.path.join(file))

    # slice audio into chunks
    for file in tqdm(files):
        # load audio
        audio, sr = sf.read(os.path.join(data_dir, file))

        # Extract only the left channel.
        audio = audio[:, 1:]
        
        # load brouhaha results
        if args.brouhaha_vad_snr is not None:
            brouhaha_idx = f'train/unlabeled/{file}'
            vad = brouhaha_results.loc[brouhaha_idx, 'vad']
            speech_start_ind = []
            speech_end_ind = []
            for i in range(len(vad)):
                if vad[i] == 1:
                    if i == 0:
                        speech_start_ind.append(i)
                    elif i == len(vad)-1:
                        speech_end_ind.append(i)
                    else:
                        if vad[i-1] == 0:
                            speech_start_ind.append(i)
                        if vad[i+1] == 0:
                            speech_end_ind.append(i+1)
            audios = []
            for start, end in zip(speech_start_ind, speech_end_ind):
                start_time = start/brouhaha_sr
                end_time = end/brouhaha_sr
                audios.append(audio[int(np.round(start_time*sr)):int(np.round(end_time*sr))])
        else:
            audios = [audio]
        
        # save chunks
        chunk = 0
        for audio in audios:

            duration = audio.shape[0]/sr
            for i in range(int(np.ceil(duration/max_chunk_time))):

                start_time = max_chunk_time*(i)
                end_time = max_chunk_time*(i+1)
                
                audio_chunk = audio[int(np.round(start_time*sr)):int(np.round(end_time*sr))]

                chunk_duration = audio_chunk.shape[0]/sr
                if chunk_duration < 3:
                    continue

                sf.write(os.path.join(out_dir, f'{file[:-4]}_{chunk}.wav'), audio_chunk, sr)
                chunk += 1
            
                                
if __name__ == '__main__':
    main()    

