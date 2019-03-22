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

num = 5
path = "/home/kwan/Desktop/my/"

start=time.time()

for step in range(13,14): # step = mfcc의 수 13
    y, sr = librosa.load(path+str(num)+"/ElizabethStreb_2018_11_0.wav")
    mfcc_feat = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=step)
    mfcc_d = librosa.feature.delta(mfcc_feat)
    mfcc_dd = librosa.feature.delta(mfcc_feat, order=2)

    y1, sr1 = librosa.load(path+str(num)+"/MyVoice1.wav")
    mfcc_feat_1 = librosa.feature.mfcc(y=y1, sr=sr1, n_mfcc=step)
    mfcc_d1 = librosa.feature.delta(mfcc_feat_1)
    mfcc_dd1 = librosa.feature.delta(mfcc_feat_1, order=2)

    y2, sr2 = librosa.load(path+str(num)+"/MyVoice2.wav")
    mfcc_feat_2 = librosa.feature.mfcc(y=y2, sr=sr2, n_mfcc=step)
    mfcc_d2 = librosa.feature.delta(mfcc_feat_2)
    mfcc_dd2 = librosa.feature.delta(mfcc_feat_2, order=2)

    y3, sr3 = librosa.load(path+str(num)+"/MyVoice3.wav")
    mfcc_feat_3 = librosa.feature.mfcc(y=y3, sr=sr3, n_mfcc=step)
    mfcc_d3 = librosa.feature.delta(mfcc_feat_3)
    mfcc_dd3 = librosa.feature.delta(mfcc_feat_3, order=2)

    y4, sr4 = librosa.load(path+str(num)+"/MyVoice4.wav")
    mfcc_feat_4 = librosa.feature.mfcc(y=y4, sr=sr4, n_mfcc=step)
    mfcc_d4 = librosa.feature.delta(mfcc_feat_4)
    mfcc_dd4 = librosa.feature.delta(mfcc_feat_4, order=2)

    y5, sr5 = librosa.load(path+str(num)+"/MyVoice5.wav")
    mfcc_feat_5 = librosa.feature.mfcc(y=y5, sr=sr5, n_mfcc=step)
    mfcc_d5 = librosa.feature.delta(mfcc_feat_5)
    mfcc_dd5 = librosa.feature.delta(mfcc_feat_5, order=2)

    y6, sr6 = librosa.load(path+str(num)+"/MyVoice6.wav")
    mfcc_feat_6 = librosa.feature.mfcc(y=y6, sr=sr6, n_mfcc=step)
    mfcc_d6 = librosa.feature.delta(mfcc_feat_6)
    mfcc_dd6 = librosa.feature.delta(mfcc_feat_6, order=2)

    y7, sr7 = librosa.load(path+str(num)+"/MyVoice7.wav")
    mfcc_feat_7 = librosa.feature.mfcc(y=y7, sr=sr7, n_mfcc=step)
    mfcc_d7 = librosa.feature.delta(mfcc_feat_7)
    mfcc_dd7 = librosa.feature.delta(mfcc_feat_7, order=2)

    y8, sr8 = librosa.load(path+str(num)+"/MyVoice8.wav")
    mfcc_feat_8 = librosa.feature.mfcc(y=y8, sr=sr8, n_mfcc=step)
    mfcc_d8 = librosa.feature.delta(mfcc_feat_8)
    mfcc_dd8 = librosa.feature.delta(mfcc_feat_8, order=2)

    rc = mfcc_feat.shape
    rc_1 = mfcc_feat_1.shape
    rc_2 = mfcc_feat_2.shape
    rc_3 = mfcc_feat_3.shape
    rc_4 = mfcc_feat_4.shape
    rc_5 = mfcc_feat_5.shape
    rc_6 = mfcc_feat_6.shape
    rc_7 = mfcc_feat_7.shape
    rc_8 = mfcc_feat_8.shape

    rcd = mfcc_d.shape
    rcd_1 = mfcc_d1.shape
    rcd_2 = mfcc_d2.shape
    rcd_3 = mfcc_d3.shape
    rcd_4 = mfcc_d4.shape
    rcd_5 = mfcc_d5.shape
    rcd_6 = mfcc_d6.shape
    rcd_7 = mfcc_d7.shape
    rcd_8 = mfcc_d8.shape

    rcdd = mfcc_dd.shape
    rcdd_1 = mfcc_dd1.shape
    rcdd_2 = mfcc_dd2.shape
    rcdd_3 = mfcc_dd3.shape
    rcdd_4 = mfcc_dd4.shape
    rcdd_5 = mfcc_dd5.shape
    rcdd_6 = mfcc_dd6.shape
    rcdd_7 = mfcc_dd7.shape
    rcdd_8 = mfcc_dd8.shape

    print("rc:",rc)
    print("rc_1:",rc_1)
    print("rc_2:",rc_2)
    print("rc_3:",rc_3)
    print("rc_4:",rc_4)
    print("rc_5:",rc_5)
    print("rc_6:",rc_6)
    print("rc_7:",rc_7)
    print("rc_8:",rc_8)

    print("ted delta:",rcd)
    print("delta_1:",rcd_1)
    print("delta_2:",rcd_2)
    print("delta_3:",rcd_3)
    print("delta_4:",rcd_4)
    print("delta_5:",rcd_5)
    print("delta_6:",rcd_6)
    print("delta_7:",rcd_7)
    print("delta_8:",rcd_8)

    print("ted delta-delta:",rcdd)
    print("delta-delta_1:",rcdd_1)
    print("delta-delta_2:",rcdd_2)
    print("delta-delta_3:",rcdd_3)
    print("delta-delta_4:",rcdd_4)
    print("delta-delta_5:",rcdd_5)
    print("delta-delta_6:",rcdd_6)
    print("delta-delta_7:",rcdd_7)
    print("delta-delta_8:",rcdd_8)

    sum_1 = 0
    sum_2 = 0
    sum_3 = 0
    sum_4 = 0
    sum_5 = 0
    sum_6 = 0
    sum_7 = 0
    sum_8 = 0

    for i in range(0, step):
        list = []
        list_1 = []
        list_2 = []
        list_3 = []
        list_4 = []
        list_5 = []
        list_6 = []
        list_7 = []
        list_8 = []

        listd = []
        listd_1 = []
        listd_2 = []
        listd_3 = []
        listd_4 = []
        listd_5 = []
        listd_6 = []
        listd_7 = []
        listd_8 = []

        listdd = []
        listdd_1 = []
        listdd_2 = []
        listdd_3 = []
        listdd_4 = []
        listdd_5 = []
        listdd_6 = []
        listdd_7 = []
        listdd_8 = []

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

        for j in range(0, rc_8[1]):
            list_8.append(mfcc_feat_8[i][j])
        #===================================
        for j in range(0, rcd[1]):
            listd.append(mfcc_d[i][j])

        for j in range(0, rcd_1[1]):
            listd_1.append(mfcc_d1[i][j])

        for j in range(0, rcd_2[1]):
            listd_2.append(mfcc_d2[i][j])

        for j in range(0, rcd_3[1]):
            listd_3.append(mfcc_d3[i][j])

        for j in range(0, rcd_4[1]):
            listd_4.append(mfcc_d4[i][j])

        for j in range(0, rcd_5[1]):
            listd_5.append(mfcc_d5[i][j])

        for j in range(0, rcd_6[1]):
            listd_6.append(mfcc_d6[i][j])

        for j in range(0, rcd_7[1]):
            listd_7.append(mfcc_d7[i][j])

        for j in range(0, rcd_8[1]):
            listd_8.append(mfcc_d8[i][j])
        #=====================================
        for j in range(0, rcdd[1]):
            listdd.append(mfcc_dd[i][j])

        for j in range(0, rcdd_1[1]):
            listdd_1.append(mfcc_dd1[i][j])

        for j in range(0, rcdd_2[1]):
            listdd_2.append(mfcc_dd2[i][j])

        for j in range(0, rcdd_3[1]):
            listdd_3.append(mfcc_dd3[i][j])

        for j in range(0, rcdd_4[1]):
            listdd_4.append(mfcc_dd4[i][j])

        for j in range(0, rcdd_5[1]):
            listdd_5.append(mfcc_dd5[i][j])

        for j in range(0, rcdd_6[1]):
            listdd_6.append(mfcc_dd6[i][j])

        for j in range(0, rcdd_7[1]):
            listdd_7.append(mfcc_dd7[i][j])

        for j in range(0, rcdd_8[1]):
            listdd_8.append(mfcc_dd8[i][j])

        arr = np.array(list)
        arr_1 = np.array(list_1)
        arr_2 = np.array(list_2)
        arr_3 = np.array(list_3)
        arr_4 = np.array(list_4)
        arr_5 = np.array(list_5)
        arr_6 = np.array(list_6)
        arr_7 = np.array(list_7)
        arr_8 = np.array(list_8)

        arrd = np.array(listd)
        arrd_1 = np.array(listd_1)
        arrd_2 = np.array(listd_2)
        arrd_3 = np.array(listd_3)
        arrd_4 = np.array(listd_4)
        arrd_5 = np.array(listd_5)
        arrd_6 = np.array(listd_6)
        arrd_7 = np.array(listd_7)
        arrd_8 = np.array(listd_8)
   
        arrdd = np.array(listdd)
        arrdd_1 = np.array(listdd_1)
        arrdd_2 = np.array(listdd_2)
        arrdd_3 = np.array(listdd_3)
        arrdd_4 = np.array(listdd_4)
        arrdd_5 = np.array(listdd_5)
        arrdd_6 = np.array(listdd_6)
        arrdd_7 = np.array(listdd_7)
        arrdd_8 = np.array(listdd_8)

        cost_1, path_1 = dtw.warping_paths(arr, arr_1)
        cost_2, path_2 = dtw.warping_paths(arr, arr_2)
        cost_3, path_3 = dtw.warping_paths(arr, arr_3)
        cost_4, path_4 = dtw.warping_paths(arr, arr_4)
        cost_5, path_5 = dtw.warping_paths(arr, arr_5)
        cost_6, path_6 = dtw.warping_paths(arr, arr_6)
        cost_7, path_7 = dtw.warping_paths(arr, arr_7)
        cost_8, path_8 = dtw.warping_paths(arr, arr_8)

        costd_1, pathd_1 = dtw.warping_paths(arrd, arrd_1)
        costd_2, pathd_2 = dtw.warping_paths(arrd, arrd_2)
        costd_3, pathd_3 = dtw.warping_paths(arrd, arrd_3)
        costd_4, pathd_4 = dtw.warping_paths(arrd, arrd_4)
        costd_5, pathd_5 = dtw.warping_paths(arrd, arrd_5)
        costd_6, pathd_6 = dtw.warping_paths(arrd, arrd_6)
        costd_7, pathd_7 = dtw.warping_paths(arrd, arrd_7)
        costd_8, pathd_8 = dtw.warping_paths(arrd, arrd_8)

        costdd_1, pathdd_1 = dtw.warping_paths(arrdd, arrdd_1)
        costdd_2, pathdd_2 = dtw.warping_paths(arrdd, arrdd_2)
        costdd_3, pathdd_3 = dtw.warping_paths(arrdd, arrdd_3)
        costdd_4, pathdd_4 = dtw.warping_paths(arrdd, arrdd_4)
        costdd_5, pathdd_5 = dtw.warping_paths(arrdd, arrdd_5)
        costdd_6, pathdd_6 = dtw.warping_paths(arrdd, arrdd_6)
        costdd_7, pathdd_7 = dtw.warping_paths(arrdd, arrdd_7)
        costdd_8, pathdd_8 = dtw.warping_paths(arrdd, arrdd_8)
        if i != 0: # 0번째 mfcc vector버림
            sum_1 += cost_1
            sum_2 += cost_2
            sum_3 += cost_3
            sum_4 += cost_4
            sum_5 += cost_5
            sum_6 += cost_6
            sum_7 += cost_7
            sum_8 += cost_8

        sum_1 = sum_1 + costd_1 + costdd_1
        sum_2 = sum_2 + costd_2 + costdd_2
        sum_3 = sum_3 + costd_3 + costdd_3
        sum_4 = sum_4 + costd_4 + costdd_4
        sum_5 = sum_5 + costd_5 + costdd_5
        sum_6 = sum_6 + costd_6 + costdd_6
        sum_7 = sum_7 + costd_7 + costdd_7
        sum_8 = sum_8 + costd_8 + costdd_8

    dis_1 = sum_1 / 38
    dis_2 = sum_2 / 38
    dis_3 = sum_3 / 38
    dis_4 = sum_4 / 38
    dis_5 = sum_5 / 38
    dis_6 = sum_6 / 38
    dis_7 = sum_7 / 38
    dis_8 = sum_8 / 38

    print("mfcc(12) + delta(13) + delta-delta(13) + fastdtw")
    print(dis_1)
    print(dis_2)
    print(dis_3)
    print(dis_4)
    print(dis_5)
    print(dis_6)
    print(dis_7)
    print(dis_8)
    print("\n")

end=time.time() - start
print("실행시간:",end)
