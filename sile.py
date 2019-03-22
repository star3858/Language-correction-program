#!/usr/bin/env python
# -*- coding: utf-8 -*-

import itertools
from pydub import AudioSegment
from pydub.utils import db_to_float
import librosa
import librosa.display
import os

def load_file_list(file_path):
	file_list = os.listdir(file_path)
	return file_list

def detect_silence(audio_segment, min_silence_len=10, silence_thresh=-45, seek_step=1):
    seg_len = len(audio_segment)

    if seg_len < min_silence_len:
        return []

    silence_thresh = db_to_float(silence_thresh) * audio_segment.max_possible_amplitude

    silence_starts = []

    last_slice_start = seg_len - min_silence_len
    slice_starts = range(0, last_slice_start + 1, seek_step)

    if last_slice_start % seek_step:
        slice_starts = itertools.chain(slice_starts, [last_slice_start])

    for i in slice_starts:
        audio_slice = audio_segment[i:i + min_silence_len]
        if audio_slice.rms <= silence_thresh:
            silence_starts.append(i)

    if not silence_starts:
        return []

    silent_ranges = []

    prev_i = silence_starts.pop(0)
    current_range_start = prev_i

    for silence_start_i in silence_starts:
        continuous = (silence_start_i == prev_i + seek_step)

        silence_has_gap = silence_start_i > (prev_i + min_silence_len)

        if not continuous and silence_has_gap:
            silent_ranges.append([current_range_start,
                                  prev_i + min_silence_len])
            current_range_start = silence_start_i
        prev_i = silence_start_i

    silent_ranges.append([current_range_start,
                          prev_i + min_silence_len])

    return silent_ranges


def detect_nonsilent(audio_segment, min_silence_len=10, silence_thresh=-45, seek_step=1):
    silent_ranges = detect_silence(audio_segment, min_silence_len, silence_thresh, seek_step)
    len_seg = len(audio_segment)

    if not silent_ranges:
        return [[0, len_seg]]

    if silent_ranges[0][0] == 0 and silent_ranges[0][1] == len_seg:
        return []

    prev_end_i = 0
    nonsilent_ranges = []
    for start_i, end_i in silent_ranges:
        nonsilent_ranges.append([prev_end_i, start_i])
        prev_end_i = end_i

    if end_i != len_seg:
        nonsilent_ranges.append([prev_end_i, len_seg])

    if nonsilent_ranges[0] == [0, 0]:
        nonsilent_ranges.pop(0)

    return nonsilent_ranges


def split_on_silence(fname, audio_segment, min_silence_len=10, silence_thresh=-45, keep_silence=0, seek_step=1):
    not_silence_ranges = detect_nonsilent(audio_segment, min_silence_len, silence_thresh, seek_step)
    chunks = [] ## 
    num = 0
    #t_val = 0

    one_sec_segment = AudioSegment.silent(duration=1000)
    for start_i, end_i in not_silence_ranges: ##
        start_i = max(0, start_i - keep_silence) ##

        end_i += keep_silence #0.1초 더해줌
        print('start:',start_i)
        print('end:',end_i)
        '''
        file_length = end_i - start_i

        if file_length < 10000:
                t_val = start_i
                continue

        if file_length > 15000:
                split_on_silence(fname,audio_segment)
        '''
        chunks.append(audio_segment[start_i:end_i]) ##

        fname2 = '/home/kwan/Desktop/filesegment/'+fname + '_' + str(num) + '.wav'
        seg = audio_segment[start_i:end_i]
        final_seg = one_sec_segment + seg
        final_seg.export(fname2, format="wav")
        num  += 1
    return chunks

if __name__ == "__main__":
        sound = load_file_list('/home/kwan/Desktop/test/')
        print("sound")
        print(len(sound))

        for fname in sound:
                print(fname)
                file_name = '/home/kwan/Desktop/test/' + fname
                wav_file = AudioSegment.from_wav(file_name)
                tmp = split_on_silence(fname ,wav_file)
                print("tmp")
                print(len(tmp))
                count = 0
                total = 0

                for i in tmp:
                        length = len(i)
                        total += length
                        print('Length of element:', length)
                        count+=1

                print('Count:',count)
                print('Total:', total)
