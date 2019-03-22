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

for step in range(13,14): # step = mfcc의 수 13
    y1, sr1 = librosa.load("/home/kwan/Desktop/TestSet/2/CharlesCMann_2018_11_1.wav")
    mfcc_feat_1 = librosa.feature.mfcc(y=y1, sr=sr1, n_mfcc=step)
    mfcc_delta_1 = librosa.feature.delta(mfcc_feat_1)
    mfcc_delta_1_2 = librosa.feature.delta(mfcc_feat_1, order=2)

    y2, sr2 = librosa.load("/home/kwan/Desktop/t/MyVoice.wav")
    mfcc_feat_2 = librosa.feature.mfcc(y=y2, sr=sr2, n_mfcc=step)
    mfcc_delta_2 = librosa.feature.delta(mfcc_feat_2)
    mfcc_delta_2_2 = librosa.feature.delta(mfcc_feat_2, order=2)

    y3, sr3 = librosa.load("/home/kwan/Desktop/t/MyVoice_1.wav")
    mfcc_feat_3 = librosa.feature.mfcc(y=y3, sr=sr3, n_mfcc=step)
    mfcc_delta_3 = librosa.feature.delta(mfcc_feat_3)
    mfcc_delta_3_2 = librosa.feature.delta(mfcc_feat_3, order=2)

    y4, sr4 = librosa.load("/home/kwan/Desktop/t/MyVoice_2.wav")
    mfcc_feat_4 = librosa.feature.mfcc(y=y4, sr=sr4, n_mfcc=step)
    mfcc_delta_4 = librosa.feature.delta(mfcc_feat_4)
    mfcc_delta_4_2 = librosa.feature.delta(mfcc_feat_4, order=2)
    '''
    y5, sr5 = librosa.load("/home/kwan/Desktop/TestSet/2/2.1/MyVoice_2.wav")
    mfcc_feat_5 = librosa.feature.mfcc(y=y5, sr=sr5, n_mfcc=step + 1)

    y6, sr6 = librosa.load("/home/kwan/Desktop/TestSet/2/2.1/nber2.wav")
    mfcc_feat_6 = librosa.feature.mfcc(y=y6, sr=sr6, n_mfcc=step + 1)
    '''
    rc_1 = mfcc_feat_1.shape
    rc_d_1 = mfcc_delta_1.shape
    rc_d_1_2 = mfcc_delta_1_2.shape
    rc_2 = mfcc_feat_2.shape
    rc_d_2 = mfcc_delta_2.shape
    rc_d_2_2 = mfcc_delta_2_2.shape
    rc_3 = mfcc_feat_3.shape
    rc_d_3 = mfcc_delta_3.shape
    rc_d_3_2 = mfcc_delta_3_2.shape
    rc_4 = mfcc_feat_4.shape
    rc_d_4 = mfcc_delta_4.shape
    rc_d_4_2 = mfcc_delta_4_2.shape
    print("rc_1:",rc_1," delta:",rc_d_1," delta2:",rc_d_1_2)
    print("rc_2:",rc_2," delta:",rc_d_2," delta2:",rc_d_2_2)
    print("rc_3:",rc_3," delta:",rc_d_3," delta2:",rc_d_3_2)
    print("rc_4:",rc_4," delta:",rc_d_4," delta2:",rc_d_4_2)
    sum1_2 = 0
    sum1_3 = 0
    sum1_4 = 0

    for i in range(0, step):
        list_1 = []
        list_2 = []
        list_3 = []
        list_4 = []

        list_d_1 = []
        list_d_2 = []
        list_d_3 = []
        list_d_4 = []

        list_d2_1 = []
        list_d2_2 = []
        list_d2_3 = []
        list_d2_4 = []

        #sum4_6 = 0
        for j in range(0, rc_1[1]):
            list_1.append(mfcc_feat_1[i][j])
            list_d_1.append(mfcc_delta_1[i][j]) 
            list_d2_1.append(mfcc_delta_1_2[i][j]) 

        for j in range(0, rc_2[1]):
            list_2.append(mfcc_feat_2[i][j])
            list_d_2.append(mfcc_delta_2[i][j]) 
            list_d2_2.append(mfcc_delta_2_2[i][j]) 

        for j in range(0, rc_3[1]):
            list_3.append(mfcc_feat_3[i][j])
            list_d_3.append(mfcc_delta_3[i][j]) 
            list_d2_3.append(mfcc_delta_3_2[i][j]) 

        for j in range(0, rc_4[1]):
            list_4.append(mfcc_feat_4[i][j])
            list_d_4.append(mfcc_delta_4[i][j]) 
            list_d2_4.append(mfcc_delta_4_2[i][j]) 

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


        _arr_d_1 = np.array(list_d_1)
        d_m1 = np.mean(_arr_d_1)
        d_d1 = np.std(_arr_d_1)
        arr_d_1 = (_arr_d_1 - d_m1) / d_d1
        _arr_d_2 = np.array(list_d_2)
        d_m2 = np.mean(_arr_d_2)
        d_d2 = np.std(_arr_d_2)
        arr_d_2 = (_arr_d_2 - d_m2) / d_d2
        _arr_d_3 = np.array(list_d_3)
        d_m3 = np.mean(_arr_d_3)
        d_d3 = np.std(_arr_d_3)
        arr_d_3 = (_arr_d_3 - d_m3) / d_d3
        _arr_d_4 = np.array(list_d_4)
        d_m4 = np.mean(_arr_d_4)
        d_d4 = np.std(_arr_d_4)
        arr_d_4 = (_arr_d_4 - d_m4) / d_d4

        _arr_d2_1 = np.array(list_d2_1)
        d2_m1 = np.mean(_arr_d2_1)
        d2_d1 = np.std(_arr_d2_1)
        arr_d2_1 = (_arr_d2_1 - d2_m1) / d2_d1
        _arr_d2_2 = np.array(list_d2_2)
        d2_m2 = np.mean(_arr_d2_2)
        d2_d2 = np.std(_arr_d2_2)
        arr_d2_2 = (_arr_d2_2 - d2_m2) / d2_d2
        _arr_d2_3 = np.array(list_d2_3)
        d2_m3 = np.mean(_arr_d2_3)
        d2_d3 = np.std(_arr_d2_3)
        arr_d2_3 = (_arr_d2_3 - d2_m3) / d2_d3
        _arr_d2_4 = np.array(list_d2_4)
        d2_m4 = np.mean(_arr_d2_4)
        d2_d4 = np.std(_arr_d2_4)
        arr_d2_4 = (_arr_d2_4 - d2_m4) / d2_d4
        '''
        cost1_2 = dtw.distance(arr_1, arr_2)
        cost1_2_d = dtw.distance(arr_d_1, arr_d_2)
        cost1_2_d2 = dtw.distance(arr_d2_1, arr_d2_2)
        cost1_3 = dtw.distance(arr_1, arr_3)
        cost1_3_d = dtw.distance(arr_d_1, arr_d_3)
        cost1_3_d2 = dtw.distance(arr_d2_1, arr_d2_3)
        cost1_4 = dtw.distance(arr_1, arr_4)
        cost1_4_d = dtw.distance(arr_d_1, arr_d_4)
        cost1_4_d2 = dtw.distance(arr_d2_1, arr_d2_4)
        '''
        cost1_2, path1_2 = fastdtw(arr_1, arr_2, radius=30, dist=euclidean)
        cost1_2_d, path1_2_d = fastdtw(arr_d_1, arr_d_2, radius=30, dist=euclidean)
        cost1_2_d2, path1_2_d2 = fastdtw(arr_d2_1, arr_d2_2, radius=30, dist=euclidean)
        cost1_3, path1_3 = fastdtw(arr_1, arr_3, radius=30, dist=euclidean)
        cost1_3_d, path1_3_d = fastdtw(arr_d_1, arr_d_3, radius=30, dist=euclidean)
        cost1_3_d2, path1_3_d2 = fastdtw(arr_d2_1, arr_d2_3, radius=30, dist=euclidean)
        cost1_4, path1_4 = fastdtw(arr_1, arr_4, radius=30, dist=euclidean)
        cost1_4_d, path1_4_d = fastdtw(arr_d_1, arr_d_4, radius=30, dist=euclidean)
        cost1_4_d2, path1_4_d2 = fastdtw(arr_d2_1, arr_d2_4, radius=30, dist=euclidean)

        #print(step,"번째", i, " 1-2 cost:",cost1_2)
        #print(step,"번째", i, " 1-3 cost:",cost1_3)
        #print(step,"번째", i, " 1-4 cost:",cost1_4)
        #print("\n")
        '''
        p1 = dtw.warping_path(arr_1, arr_2)
        p2 = dtw.warping_path(arr_1, arr_3)
        p3 = dtw.warping_path(arr_1, arr_4)

        dtwvis.plot_warping(arr_1, arr_2, p1, filename="/home/kwan/Desktop/tmp/12_"+str(step)+"_"+str(i)+".png")
        dtwvis.plot_warping(arr_1, arr_3, p2, filename="/home/kwan/Desktop/tmp/13_"+str(step)+"_"+str(i)+".png")
        dtwvis.plot_warping(arr_1, arr_4, p3, filename="/home/kwan/Desktop/tmp/14_"+str(step)+"_"+str(i)+".png")
        '''
        #cost4_6 = dtw.distance(arr_4, arr_6)

        if i != 0: # 0번째 mfcc vector버림
            sum1_2 += cost1_2
            sum1_3 += cost1_3
            sum1_4 += cost1_4
            #sum4_6 += cost4_6
        
        sum1_2 += cost1_2_d
        sum1_3 += cost1_3_d
        sum1_4 += cost1_4_d

        sum1_2 += cost1_2_d2
        sum1_3 += cost1_3_d2
        sum1_4 += cost1_4_d2
        
    dis1_2 = sum1_2 / 38
    dis1_3 = sum1_3 / 38
    dis1_4 = sum1_4 / 38
    #dis4_6 = sum4_6 / step-1

    print("coefficient갯수:",step," 원본vs클리어 평균 거리 : ", dis1_2)
    print("coefficient갯수:",step," 원본vs약간 이상 평균 거리 : ", dis1_3)
    print("coefficient갯수:",step," 원본vs많이 이상 평균 거리 : ", dis1_4)
    #print("coefficient갯수:",step," 나vs느버  평균 거리 : ", dis4_6)
    print("\n")
