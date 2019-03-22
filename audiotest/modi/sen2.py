#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
import numpy as np
import librosa
from fastdtw import fastdtw
from dtaidistance import dtw
from dtaidistance import dtw_visualisation as dtwvis

for step in range(1,41): # mfcc의 수
    y1, sr1 = librosa.load("/home/kwan/Desktop/audio/sen1/ted.wav")
    mfcc_feat_1 = librosa.feature.mfcc(y=y1, sr=sr1, n_mfcc=step)

    y2, sr2 = librosa.load("/home/kwan/Desktop/audio/sen1/ted_BK.wav")
    mfcc_feat_2 = librosa.feature.mfcc(y=y2, sr=sr2, n_mfcc=step)

    y3, sr3 = librosa.load("/home/kwan/Desktop/audio/sen1/ted_GH_1.wav")
    mfcc_feat_3 = librosa.feature.mfcc(y=y3, sr=sr3, n_mfcc=step)

    rc_1 = mfcc_feat_1.shape
    rc_2 = mfcc_feat_2.shape
    rc_3 = mfcc_feat_3.shape

    sum1_2 = 0
    sum1_3 = 0

    for i in range(0, step):
        tot_list_1 = []
        tot_list_2 = []
        tot_list_3 = []

        list_1 = []
        list_2 = []
        list_3 = []

        for j in range(0, rc_1[1]):
            list_1.append(mfcc_feat_1[i][j])
            tot_list_1.append(list_1)

        for j in range(0, rc_2[1]):
            list_2.append(mfcc_feat_2[i][j])
            tot_list_2.append(list_2)

        for j in range(0, rc_3[1]):
            list_3.append(mfcc_feat_3[i][j])
            tot_list_3.append(list_3)

        arr_1 = np.array(list_1)
        arr_2 = np.array(list_2)
        arr_3 = np.array(list_3)

        cost1_2 = dtw.distance(arr_1, arr_2)
        cost1_3 = dtw.distance(arr_1, arr_3)

        path1_2 = dtw.warping_path(arr_1, arr_2)
        path1_3 = dtw.warping_path(arr_1, arr_3)
        
        f_name1_2 = str(i) + "1_2.png"
        f_name1_3 = str(i) + "1_3.png"
        dtwvis.plot_warping(arr_1, arr_2, path1_2, filename=f_name1_2)
        dtwvis.plot_warping(arr_1, arr_3, path1_3, filename=f_name1_3)
        if i != 0: # 0번째 mfcc vector버림
            sum1_2 += cost1_2
            sum1_3 += cost1_3

    dis1_2 = sum1_2 / step-1
    dis1_3 = sum1_3 / step-1

    print("coefficient갯수:",step," ted음성vs녹음파일(경환) 평균 거리 : ", dis1_2)
    print("coefficient갯수:",step," ted음성vs녹음파일(경환)(느리게) 평균 거리 : ", dis1_3)
    print("\n")
