#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
import numpy as np
import librosa
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from dtaidistance import dtw
from dtaidistance import dtw_visualisation as dtwvis
import time

num = 4
path = "/home/kwan/Desktop/my/"

start = time.time()

for step in range(13,14): # step = mfcc의 수 13
    y, sr = librosa.load(path+str(num)+"/WanisKabbaj_2018S_5_0.wav")
    mfcc_feat = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=step)
    
    y1, sr1 = librosa.load(path+str(num)+"/MyVoice1.wav")
    mfcc_feat_1 = librosa.feature.mfcc(y=y1, sr=sr1, n_mfcc=step)

    y2, sr2 = librosa.load(path+str(num)+"/MyVoice2.wav")
    mfcc_feat_2 = librosa.feature.mfcc(y=y2, sr=sr2, n_mfcc=step)

    y3, sr3 = librosa.load(path+str(num)+"/MyVoice3.wav")
    mfcc_feat_3 = librosa.feature.mfcc(y=y3, sr=sr3, n_mfcc=step)

    y4, sr4 = librosa.load(path+str(num)+"/MyVoice4.wav")
    mfcc_feat_4 = librosa.feature.mfcc(y=y4, sr=sr4, n_mfcc=step)

    y5, sr5 = librosa.load(path+str(num)+"/MyVoice5.wav")
    mfcc_feat_5 = librosa.feature.mfcc(y=y5, sr=sr5, n_mfcc=step)

    y6, sr6 = librosa.load(path+str(num)+"/MyVoice6.wav")
    mfcc_feat_6 = librosa.feature.mfcc(y=y6, sr=sr6, n_mfcc=step)

    y7, sr7 = librosa.load(path+str(num)+"/MyVoice7.wav")
    mfcc_feat_7 = librosa.feature.mfcc(y=y7, sr=sr7, n_mfcc=step)

    rc = mfcc_feat.shape
    rc_1 = mfcc_feat_1.shape
    rc_2 = mfcc_feat_2.shape
    rc_3 = mfcc_feat_3.shape
    rc_4 = mfcc_feat_4.shape
    rc_5 = mfcc_feat_5.shape
    rc_6 = mfcc_feat_6.shape
    rc_7 = mfcc_feat_7.shape

    print("rc:",rc)
    print("rc_1:",rc_1)
    print("rc_2:",rc_2)
    print("rc_3:",rc_3)
    print("rc_4:",rc_4)
    print("rc_5:",rc_5)
    print("rc_6:",rc_6)
    print("rc_7:",rc_7)

    sum_1 = 0
    sum_2 = 0
    sum_3 = 0
    sum_4 = 0
    sum_5 = 0
    sum_6 = 0
    sum_7 = 0

    for i in range(0, step):
        list = []
        list_1 = []
        list_2 = []
        list_3 = []
        list_4 = []
        list_5 = []
        list_6 = []
        list_7 = []

        for j in range(0, rc[1]):
            list.append(mfcc_feat[i][j])

        for j in range(0, rc_1[1]):
            list_1.append(mfcc_feat_1[i][j])

        for j in range(0, rc_2[1]):
            list_2.append(mfcc_feat_2[i][j])

        for j in range(0, rc_3[1]):
            list_3.append(mfcc_feat_3[i][j])

        for j in range(0, rc_4[1]):
            list_4.append(mfcc_feat_4[i][j])

        for j in range(0, rc_5[1]):
            list_5.append(mfcc_feat_5[i][j])

        for j in range(0, rc_6[1]):
            list_6.append(mfcc_feat_6[i][j])

        for j in range(0, rc_7[1]):
            list_7.append(mfcc_feat_7[i][j])

        _arr = np.array(list)
        m = np.mean(_arr)
        d = np.std(_arr)
        arr = (_arr - m) / d

        _arr_1 = np.array(list_1)
        m1 = np.mean(_arr_1)
        d1 = np.std(_arr_1)
        arr_1 = (_arr_1 - m1) / d1

        _arr_2 = np.array(list_2)
        m2 = np.mean(_arr_2)
        d2 = np.std(_arr_2)
        arr_2 = (_arr_2 - m2) / d2

        _arr_3 = np.array(list_3)
        m3 = np.mean(_arr_3)
        d3 = np.std(_arr_3)
        arr_3 = (_arr_3 - m3) / d3

        _arr_4 = np.array(list_4)
        m4 = np.mean(_arr_4)
        d4 = np.std(_arr_4)
        arr_4 = (_arr_4 - m4) / d4

        _arr_5 = np.array(list_5)
        m5 = np.mean(_arr_5)
        d5 = np.std(_arr_5)
        arr_5 = (_arr_5 - m5) / d5

        _arr_6 = np.array(list_6)
        m6 = np.mean(_arr_6)
        d6 = np.std(_arr_6)
        arr_6 = (_arr_6 - m6) / d6

        _arr_7 = np.array(list_7)
        m7 = np.mean(_arr_7)
        d7 = np.std(_arr_7)
        arr_7 = (_arr_7 - m7) / d7

        cost_1, path_1 = fastdtw(arr, arr_1, dist=euclidean)
        cost_2, path_2 = fastdtw(arr, arr_2, dist=euclidean)
        cost_3, path_3 = fastdtw(arr, arr_3, dist=euclidean)
        cost_4, path_4 = fastdtw(arr, arr_4, dist=euclidean)
        cost_5, path_5 = fastdtw(arr, arr_5, dist=euclidean)
        cost_6, path_6 = fastdtw(arr, arr_6, dist=euclidean)
        cost_7, path_7 = fastdtw(arr, arr_7, dist=euclidean)

        if i != 0: # 0번째 mfcc vector버림
            sum_1 += cost_1
            sum_2 += cost_2
            sum_3 += cost_3
            sum_4 += cost_4
            sum_5 += cost_5
            sum_6 += cost_6
            sum_7 += cost_7

    dis_1 = sum_1 / 12
    dis_2 = sum_2 / 12
    dis_3 = sum_3 / 12
    dis_4 = sum_4 / 12
    dis_5 = sum_5 / 12
    dis_6 = sum_6 / 12
    dis_7 = sum_7 / 12

    print("mfcc(12) + normalization + fastdtw")
    print("ted음성vs1번 음성 평균 거리 : ", dis_1)
    print("ted음성vs2번 음성 평균 거리 : ", dis_2)
    print("ted음성vs3번 음성 평균 거리 : ", dis_3)
    print("ted음성vs4번 음성 평균 거리 : ", dis_4)
    print("ted음성vs5번 음성 평균 거리 : ", dis_5)
    print("ted음성vs6번 음성 평균 거리 : ", dis_6)
    print("ted음성vs7번 음성 평균 거리 : ", dis_7)
    print("\n")

end=time.time()-start
print("실행시간:",end)
