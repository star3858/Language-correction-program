#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import time
import os
import io
import contextlib
import pyaudio
import threading
import wave
from pydub import AudioSegment
from playsound import playsound
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtMultimedia import *

from deepspeech import Model
import subprocess

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random
import scipy.io.wavfile as wav
import scipy.signal as signal
import numpy as np

import itertools
from pydub.utils import db_to_float
from sile import detect_silence, detect_nonsilent, split_on_silence
import librosa
import librosa.display

from google.cloud import speech
from google.cloud.speech import enums
from google.cloud.speech import types
from dtw import DTW
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from dtaidistance import dtw
from dtaidistance import dtw_visualisation as dtwvis

from pygame import mixer
import pygame
import threading

class WorkerThread(QThread):
    numberChanged = pyqtSignal(str)
    def __init__(self, sec,ted_timer, parent=None):
        super(WorkerThread, self).__init__(parent)
        self._running = False
        self.sec = sec
        self.ted = ted_timer
        self.timer = QTimer()  # Question: define the QTimer here?
        self.timer.timeout.connect(self.doWork)
        self.timer.start(100)

    def doWork(self):
        if self._running == False:
            for i in np.arange(0.0,self.sec,0.1):  # Question: How to return the i onto the QLable?
                self.ted.setText(str(round(i,2))+'(sec)')
                interval=100
                loop = QEventLoop()
                QTimer.singleShot(interval, loop.quit)
                loop.exec_()
                self.numberChanged.emit(str(i))

            self._running = True

class asset(QWidget):
    def __init__(self,parent=None,py=pyaudio.PyAudio(),chunk = 1024,frmat=pyaudio.paInt16,channels=1,rate=16000):
        super(asset, self).__init__(parent)
        font = QtGui.QFont("맑은 고딕", 15) # 폰트 설정
        sc_font = QtGui.QFont("맑은 고딕", 17) # 스크립트 폰트 설정
        per_font = QtGui.QFont("맑은 고딕",15)
        t_font = QtGui.QFont("맑은 고딕",12)
        rec_font = QtGui.QFont("맑은 고딕",12,QtGui.QFont.Bold)
        com_font = QtGui.QFont("맑은 고딕",10,QtGui.QFont.Bold)
        n_font = QtGui.QFont("맑은 고딕",8,QtGui.QFont.Bold)
        self.TedAudioPath = " "
        self.Tot_len = 0
        self.my_len = 0
        self.ted_sc_path = " "
        self.isrecording=True
        self.st = 1
        self.frames = []
        self.FORMAT = frmat
        self.p=py
        self.RATE = rate
        self.CHUNK = chunk
        self.CHANNELS = channels
        self.t1=0.0
        self.t2=0.0

        p = self.palette()
        p.setColor(self.backgroundRole(), QtCore.Qt.white)
        self.setPalette(p)
        #===================녹음파일 파라미터=====================#
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.CHUNK=1024
        self.RATE = 16000
        self.RECORD_SECONDS = 10
        self.WAVE_OUTPUT_FILENAME = "./MyAudioTmp/MyVoice.wav"
        self.MyVoicePath = "./MyAudioTmp/MyVoice.wav"
        self.stream = self.p.open(format=self.FORMAT, channels=self.CHANNELS, rate=self.RATE, input=True, frames_per_buffer=self.CHUNK)

        #==========================text===========================#
        self.filetext = QTextEdit(self)
        self.filetext.resize(250,40)
        self.filetext.setFont(com_font)
        self.filetext.setText(" ? 따라할 강연의 음성을 한번만 선택하세요.")
        self.filetext.setStyleSheet("color: rgb(255,0,0); border-width: 0px; border-style: solid")
        self.filetext.move(0, 0)

        #==========================테드===========================#
        self.tedtext = QTextEdit(self)
        self.tedtext.resize(80,40)
        self.tedtext.setFont(t_font)
        self.tedtext.setText("TED")
        self.tedtext.move(270,40)

        #======================테드 타이머========================#
        self.ted_timer = QTextEdit(self)
        self.ted_timer.resize(80,40)
        self.ted_timer.setFont(t_font)
        self.ted_timer.setText("0(sec)")
        self.ted_timer.move(350,40)

        #========================내 음성==========================#
        self.mytext = QTextEdit(self)
        self.mytext.resize(80,40)
        self.mytext.setFont(t_font)
        self.mytext.setText("내음성")
        self.mytext.move(270, 240)

        #======================음성 타이머========================#
        self.my_timer = QTextEdit(self)
        self.my_timer.resize(80,40)
        self.my_timer.setFont(t_font)
        self.my_timer.setText("0(sec)")
        self.my_timer.move(350, 240)

        #====================강연파일 리스트======================#
        self.assetList = QListView(self)
        self.assetList.resize(250, 320)
        self.assetList.clicked.connect(self.on_treeView_clicked)
        self.assetList.move(0,40)

        #===================강연파일 그래프=======================#
        self.m = PlotCanvas(self, width=6, height=2)
        self.m.move(460,0)

        #==========================text===========================#
        self.tedtext = QTextEdit(self)
        self.tedtext.resize(460,40)
        self.tedtext.setFont(com_font)
        self.tedtext.setText(" ? 분석하기를 눌러 나온 붉은색 글자 부분에서 발음에 다시 한번 신경써주세요.")
        self.tedtext.setStyleSheet("color: rgb(255,0,0); border-width: 0px; border-style: solid")
        self.tedtext.move(0, 360)

        #==========================text===========================#
        self.tedtext2 = QTextEdit(self)
        self.tedtext2.resize(460,60)
        self.tedtext2.setFont(t_font)
        self.tedtext2.setText("TED SCRIPT>")
        self.tedtext2.setStyleSheet("border-width: 0px; border-style: solid")
        self.tedtext2.move(0, 400)

        #===================강연파일 스크립트=====================#
        self.lineedit = QTextEdit(self)
        self.lineedit.resize(460,190)
        self.lineedit.setFont(sc_font)
        self.lineedit.move(0, 450)

        #==================녹음파일 그래프========================#
        self._m = PlotCanvas(self, width=6, height=2)
        self._m.move(460, 200)

        #=====================분석 그래프=========================#
        self.__m = PlotCanvas(self, width=6, height=2)
        self.__m.move(460, 400)

        #=====================분석 그래프=========================#
        self.anal_g = PlotCanvas(self, width=0, height=0)

        #=====================억양에유의하세요====================#
        self.intotext = QTextEdit(self)
        self.intotext.resize(600,40)
        self.intotext.setFont(com_font)
        self.intotext.setText("                                       ? 분석하기를 눌러 나온 Result의 붉은영역에 억양을 주의하세요.")
        self.intotext.setStyleSheet("color: rgb(255,0,0); border-width: 0px; border-style: solid")
        self.intotext.move(460, 600)

        #====================내음성듣기 버튼======================#
        playVoice = QPushButton('내음성듣기', self)
        playVoice.setFont(font)
        playVoice.move(760, 650)
        playVoice.clicked.connect(self.PlayMyVoice)

        #===================녹음하기 버튼=========================#
        recordButton = QPushButton('녹음하기', self)
        recordButton.setFont(font)
        recordButton.move(960,650)
        recordButton.clicked.connect(self.RecordStart)

        #===================녹음중지 버튼=========================#
        StopButton = QPushButton('녹음중지', self)
        StopButton.setFont(font)
        StopButton.move(870,650)
        StopButton.clicked.connect(self.Stop)
        #===================분석하기 버튼=========================#
        compButton = QPushButton('분석하기', self)
        compButton.setFont(font)
        compButton.move(670, 650)
        compButton.clicked.connect(self.Analysis)

        #==========================n1===========================#
        self.text1 = QTextEdit(self)
        self.text1.resize(30,30)
        self.text1.setFont(n_font)
        self.text1.setStyleSheet("background-color: yellow")
        self.text1.setText("  1")
        self.text1.move(250,0)

        #==========================n2===========================#
        self.text2 = QTextEdit(self)
        self.text2.resize(30,30)
        self.text2.setFont(n_font)
        self.text2.setStyleSheet("background-color: yellow")
        self.text2.setText("  2")
        self.text2.move(980,630)

        #==========================n3===========================#
        self.text3 = QTextEdit(self)
        self.text3.resize(30,30)
        self.text3.setFont(n_font)
        self.text3.setStyleSheet("background-color: yellow")
        self.text3.setText("  3")
        self.text3.move(890,630)

        #==========================n4===========================#
        self.text4 = QTextEdit(self)
        self.text4.resize(30,30)
        self.text4.setFont(n_font)
        self.text4.setStyleSheet("background-color: yellow")
        self.text4.setText("  4")
        self.text4.move(780,630)

        #==========================n5===========================#
        self.text5 = QTextEdit(self)
        self.text5.resize(30,30)
        self.text5.setFont(n_font)
        self.text5.setStyleSheet("background-color: yellow")
        self.text5.setText("  5")
        self.text5.move(690,630)

        #========================녹음상태==========================#
        self.rectext = QTextEdit(self)
        self.rectext.resize(80,40)
        self.rectext.setFont(t_font)
        self.rectext.setText("녹음상태")
        self.rectext.move(270, 280)

        #======================녹음상태2===========================#
        self.recstate = QTextEdit(self)
        self.recstate.resize(80,40)
        self.recstate.setFont(rec_font)
        self.recstate.setText("")
        self.recstate.move(350, 280)

        #======================ADD ITEMS==========================#
        self.list_data = os.listdir("/home/kwan/Desktop/0.8audio")
        self.list_data.sort() # 강연 파일 리스트 소팅
        
        #=========================정확도==========================#
        self.probability = QTextEdit(self)
        self.probability.resize(640,60)
        self.probability.setFont(per_font)
        self.probability.move(10, 640)

        dir = listModel(self.list_data)
        self.assetList.setModel(dir)

        self.setFocus()
    #===============================리스트 아이템 눌렀을때=========================================#
    @QtCore.pyqtSlot(QtCore.QModelIndex)
    def on_treeView_clicked(self, index):
        self.lineedit.clear()
        itms = self.assetList.selectedIndexes()
        for it in itms:
            _str = self.list_data[it.row()] # wav 파일 이름 파싱
            ment=_str+"을 재생합니다."
            self.probability.setText(ment)
            script_name = _str[:-4]
            s = script_name +".txt"
            print(_str,"을 선택했습니다.")
            path = "/home/kwan/Desktop/script/" + s # 스크립트 경로
            self.ted_sc_path = path
            audioPath = "/home/kwan/Desktop/0.8audio/" + self.list_data[it.row()]
            self.TedAudioPath = audioPath
            WavFile = AudioSegment.from_wav(audioPath)
            self.RECORD_SECONDS = len(WavFile)/1000 + 1 # 녹음시간 설정(누른 파일의 길이 + 1만큼)
            f = open(path, mode='rt') # f 스크립트 파일 오픈
            f2 = open(path, mode='rt')
            total_script = f.read(1000)
            tot_len = len(total_script)
            for i in range(0,tot_len):
                tmp = f2.read(1)
                self.lineedit.setTextColor(QtGui.QColor(0,0,0))
                self.lineedit.insertPlainText(tmp)
           
            wav_name = '/home/kwan/Desktop/0.8audio/' + self.list_data[it.row()]
            print(_str,"을 재생합니다.")
            print(wav_name)
            mixer.pre_init(frequency=16000, size=-16, channels=1)
            pygame.init()
            t_y, t_sr = librosa.load(self.TedAudioPath)
            sec = librosa.get_duration(y=t_y, sr=t_sr)
            s=mixer.Sound(wav_name)
            s.play()
            self.thread = WorkerThread(sec,self.ted_timer)
            self.m.ted_plot(self.TedAudioPath)

    def PlayMyVoice(self):
        print("play my voice.")
        t_y, t_sr = librosa.load(self.WAVE_OUTPUT_FILENAME)
        sec = librosa.get_duration(y=t_y, sr=t_sr)
        QSound.play(self.WAVE_OUTPUT_FILENAME)
        self.thread = WorkerThread(sec,self.my_timer) # My Voice timer
        self._m.my_plot(self.WAVE_OUTPUT_FILENAME)

    def RecordStart(self):
        #self.probability.setText("녹음을 시작합니다.")
        self.recstate.setText("녹음중")
        self.recstate.setStyleSheet("color: rgb(255,0,0);")
        self.isrecording=True
        t=threading.Thread(target=self.Record)
        t.start()

    def Record(self):
        while self.isrecording:
            self.st=1
            p = pyaudio.PyAudio()
            stream = self.p.open(format=self.FORMAT,channels=self.CHANNELS,rate=self.RATE,input=True,frames_per_buffer=self.CHUNK)
            frames = []
            print("==============Start to record the audio.==============")
            while self.st==1:
                data = stream.read(self.CHUNK)
                frames.append(data)
    
            stream.close()
            print("================Recording is finished.================")
        
            wf = wave.open(self.WAVE_OUTPUT_FILENAME, 'wb')    
            wf.setnchannels(self.CHANNELS)    
            wf.setsampwidth(p.get_sample_size(self.FORMAT))        
            wf.setframerate(self.RATE)    
            wf.writeframes(b''.join(frames))
            wf.close()

    def Stop(self):
        #self.probability.setText("녹음을 마칩니다.")
        self.isrecording=False
        self.recstate.setText("")
        self.st=0
        print("Stop")

    def Analysis(self):
        self.probability.setText("분석중입니다. 잠시만 기다려 주세요...")
        print("Analysis")
        try:
            self.anal_g.Twoplot(self.TedAudioPath, self.WAVE_OUTPUT_FILENAME, self.ted_sc_path, self.probability, self.lineedit, self.__m)
        except IndexError:
            error=2

class listModel(QAbstractListModel): 
    def __init__(self, datain, parent=None, *args): 
        """ datain: a list where each item is a row
        """
        QAbstractListModel.__init__(self, parent, *args) 
        self.listdata = datain

    def rowCount(self, parent=QModelIndex()): 
        return len(self.listdata) 

    def data(self, index, role): 
        if index.isValid() and role == Qt.DisplayRole:
            return QVariant(self.listdata[index.row()])
        else: 
            return QVariant()

class PlotCanvas(FigureCanvas): 
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
 
        FigureCanvas.__init__(self, fig)
        self.setParent(parent)
 
        FigureCanvas.setSizePolicy(self,
                QSizePolicy.Expanding,
                QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

    def data_gen(self,list):
        cnt = 0
        l_len = len(list)
        print("data_gen-l_len:",l_len)
        while cnt < l_len:
            print("cnt:",cnt," y값:",list[cnt])
            yield cnt, list[cnt]
            cnt += 1

    def init(self):
        print("init")
        self.ax1.set_ylim(-500.0, 500.0)
        self.ax1.set_xlim(0, 450)
        del self.xdata[:]
        del self.ydata[:]
        self.line.set_data(self.xdata, self.ydata)
        return self.line,

    def run(self,data):
        print("run")
        # update the data
        t, y = data
        self.xdata.append(t)
        self.ydata.append(y)
        xmin, xmax = self.ax1.get_xlim()

        if t >= xmax:
            self.ax1.set_xlim(xmin, 2*xmax)
            self.ax1.figure.canvas.draw()
        self.line.set_data(self.xdata, self.ydata)
        return self.line,

    def init2(self):
        self.line2.set_data([], [])
        return self.line2,

    def animate(self,i):
        x = i * 0.01
        y = np.linspace(self.y_min, self.y_max, 1000)
        self.line2.set_data(x, y)
        return self.line2,

    def init3(self):
        self.line3.set_data([], [])
        return self.line3,

    def animate3(self,i):
        x = i * 0.01
        y = np.linspace(self.y_min2, self.y_max2, 1000)
        self.line3.set_data(x, y)
        return self.line3,

    def ted_plot(self,ted):
        print("TED음성의 그래프를 출력합니다.")
        num_mfcc = 13
        t_y, t_sr = librosa.load(ted)
        sec = librosa.get_duration(y=t_y, sr=t_sr)
        print("TED wav duration : ",sec)
        t_mfcc_feat = librosa.feature.mfcc(y=t_y, sr=t_sr, n_mfcc=num_mfcc)
        t_rc = t_mfcc_feat.shape
        t_graph_list = []
        self.y_min = 1000
        self.y_max = 0
        for j in range(0,t_rc[1]):
            tmp = 0
            for i in range(1,13):
                tmp += t_mfcc_feat[i][j]

            #tmp/=12
            if self.y_min > tmp:
                self.y_min = tmp
            if self.y_max < tmp:
                self.y_max = tmp
            t_graph_list.append(tmp)

        A_arr = np.array(t_graph_list)
        self.ax = self.figure.add_subplot(111)
        self.ax.cla()
        self.ax.set_title("TED")
        _x = []
        for i in range(0, len(A_arr)):
            _x.append(i / (len(A_arr)/sec))

        x = np.array(_x)
        print("x축배열길이:",len(x))
        #for i in range(0,len(x)):
            #print(x[i])
        self.line2, = self.ax.plot(x,A_arr)
        self.ax.plot(x, A_arr, 'g')
        self.ax.get_yaxis().set_visible(False)
        self.ax.fill_between(x, self.y_min, A_arr, facecolor='green', interpolate=True, alpha=0.7)
        self.draw()
        _sec = int(sec * 100)
        print("Before animation")
        anim = animation.FuncAnimation(self.figure, self.animate, init_func=self.init2,
                                       frames=_sec, interval=10, repeat=False, blit=True)
        print("After animation")
        self.draw()

    def my_plot(self,my):
        print("my_plot")
        num_mfcc = 13
        m_y, m_sr = librosa.load(my)
        m_mfcc_feat = librosa.feature.mfcc(y=m_y, sr=m_sr, n_mfcc=num_mfcc)
        sec = librosa.get_duration(y=m_y, sr=m_sr)
        m_rc = m_mfcc_feat.shape
        m_graph_list = []
        self.y_min2 = 1000
        self.y_max2 = 0
        for j in range(0,m_rc[1]):
            tmp = 0
            for i in range(1,13):
                tmp += m_mfcc_feat[i][j]
            #tmp /= 12
            if self.y_min2 > tmp:
                self.y_min2 = tmp
            if self.y_max2 < tmp:
                self.y_max2 = tmp
            m_graph_list.append(tmp)

        A = np.array(m_graph_list)
        self.ax = self.figure.add_subplot(111)
        self.ax.cla()
        self.ax.set_title("My Voice")
        _x = []
        for i in range(0,len(A)):
            _x.append(i/(len(A)/sec))
        x = np.array(_x)
        self.line3, = self.ax.plot(x,A)
        self.ax.plot(x,A,'m')
        self.ax.get_yaxis().set_visible(False)
        self.ax.fill_between(x, self.y_min2, A, facecolor='magenta', interpolate=True, alpha=0.7)
        self.draw()
        _sec = int(sec * 100)
        anim2 = animation.FuncAnimation(self.figure, self.animate3, init_func=self.init3,
                                       frames=_sec, interval=10, repeat=False, blit=True)
        self.draw()

    def Pitchplot(self, t, list, t2, list2, a, b, p, offset, y_min, y_min2,_A,_B):
        '''
        fig=plt.figure()
        fig.add_subplot(211)
        plt.title("TED(Green) | My Voice(Purple)")
        plt.plot(t,list,'g')
        plt.fill_between(t, y_min, list, facecolor='green', interpolate=True, alpha=0.7)
        fig.add_subplot(212)
        plt.plot(t2,list2-offset,'m')
        self.ax.fill_between(t2, y_min2, list2-offset, facecolor='magenta', interpolate=True, alpha=0.7)
        '''
        self.ax = self.figure.add_subplot(111)
        self.ax.cla()
        self.ax.set_title("Result(Red : different)")
        list3 = list2-offset
        self.ax.plot(t2,list2-offset,'m')
        self.ax.get_yaxis().set_visible(False)
        for (x1, x2) in p:
            print("list[",x1,"]:",list[x1]," list2[",x2,"]:",list2[x2]-offset)
            try:
                if abs(list[x1] - (list2[x2]-offset)) > 50:
                    list3[x2] = y_min2
            except IndexError:
                    error=3

        self.ax.fill_between(t2,y_min2,list2-offset, facecolor='magenta', interpolate=True, alpha=0.7)
        for (x1, x2) in p:
            self.ax.fill_between(t2, list3, list2-offset, facecolor='red', interpolate=True)
        #plt.show()
        self.draw()

    def Twoplot(self, ted, myvoice, sc_path, pro, lineedit, __m):
        print("TED, MY VOiCE음성의 분석을 시작합니다.")
        f = open(sc_path, mode='rt') # 전체 스크립트
        client = speech.SpeechClient() # Google Cloud Speech API
        my_file = os.path.join(os.path.dirname(__file__),myvoice)

        with io.open(my_file, 'rb') as my_audio_file:
            content = my_audio_file.read()
            audio = types.RecognitionAudio(content=content)

        config = types.RecognitionConfig(encoding=enums.RecognitionConfig.AudioEncoding.LINEAR16,sample_rate_hertz=16000,language_code='en-US')
        _cmd = []
        cmd = ""
        response = client.recognize(config, audio)

        for i in range(0,len(response.results)):
            _cmd.append(response.results[i])

        for i in range(0,len(_cmd)):
            tmp = str(_cmd[i])
            tmp_list = tmp.split('"')
            cmd = cmd + tmp_list[1]
        print("내 음성 Speech -> Text")
        print(cmd) # 내 음성 스크립트
        cmd_len = len(cmd)

        total_script = f.read(1000)
        tot_len = len(total_script)
        tot_visited = [0] * (tot_len+1)
        common_list = []
        dp = np.zeros((tot_len+1, cmd_len+1))
        visited = np.zeros((tot_len+1, cmd_len+1))

        for i in range(1, tot_len+1): # 최장공통부분수열 알고리즘
            for j in range(1, cmd_len+1):
                if total_script[i-1].lower() == cmd[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                    visited[i][j] = 3
                else:
                    dp[i][j] = max(dp[i][j-1],dp[i-1][j])

                    if dp[i][j] == dp[i-1][j]:
                        visited[i][j] = 1
                    else:
                        visited[i][j] = 2
        f2 = open(sc_path, mode='rt')
        a = tot_len
        b = cmd_len
        while visited[a][b] != 0:
             if visited[a][b] == 3:
                 common_list.append(total_script[a - 1])
                 tot_visited[a - 1] = 1
                 a -= 1
                 b -= 1
             elif visited[a][b] == 1:
                 a -= 1
             elif visited[a][b] == 2:
                 b -= 1
        common_list.reverse()
        lineedit.clear()
        for i in range(0,tot_len):
            tmp = f2.read(1)
            if tot_visited[i] == 1 or total_script[i] == '.' or total_script[i] == ',' or total_script[i] == '"' or total_script[i] == '!' or total_script[i] == '?':
                lineedit.setTextColor(QtGui.QColor(0,0,0))
                lineedit.insertPlainText(tmp)
            elif tot_visited[i] == 0:
                lineedit.setTextColor(QtGui.QColor(255,0,0))
                lineedit.insertPlainText(tmp)

        num_mfcc = 13
        t_y, t_sr = librosa.load(ted)
        t_mfcc_feat = librosa.feature.mfcc(y=t_y, sr=t_sr, n_mfcc=num_mfcc)
        m_y, m_sr = librosa.load(myvoice)
        m_mfcc_feat = librosa.feature.mfcc(y=m_y, sr=m_sr, n_mfcc=num_mfcc)

        t_rc = t_mfcc_feat.shape
        m_rc = m_mfcc_feat.shape

        sum = 0
        for i in range(1, num_mfcc):
            t_tmp = []
            m_tmp = []
            for j in range(0, t_rc[1]):
                t_tmp.append(t_mfcc_feat[i][j])
            for j in range(0, m_rc[1]):
                m_tmp.append(m_mfcc_feat[i][j])
            _t_arr = np.array(t_tmp)
            t_m = np.mean(_t_arr)
            t_d = np.std(_t_arr)
            t_arr = (_t_arr - t_m) / t_d
            _m_arr = np.array(m_tmp)
            m_m = np.mean(_m_arr)
            m_d = np.std(_m_arr)
            m_arr = (_m_arr - m_m) / m_d

            dtw_cost, dtw_path = fastdtw(t_arr, m_arr, dist=euclidean) # fastDTW 사용
            sum += dtw_cost

        tot_dis = sum / (num_mfcc - 1)
        tot_per = 0
        ted_len = 0
        my_len = 0

        t_graph_list = []
        m_graph_list = []
        t_graph_list2 = []
        m_graph_list2 = []
        y_min=1000
        y_min2=1000
        ##
        for j in range(0,t_rc[1]):
            tmp = 0
            for i in range(1,13):
                tmp += t_mfcc_feat[i][j]
            #tmp /= 12
            t_graph_list.append(tmp)
            if y_min > tmp:
                y_min = tmp
            tmp2 = tmp / 12
            t_graph_list2.append(tmp2)

        for j in range(0,m_rc[1]):
            tmp = 0
            for i in range(1,13):
                tmp += m_mfcc_feat[i][j]
            #tmp /= 12
            m_graph_list.append(tmp)
            if y_min2 > tmp: 
                y_min2 = tmp
            tmp2 = tmp / 12
            m_graph_list2.append(tmp2)

        with contextlib.closing(wave.open(ted,'r')) as t_f:
            t_frames = t_f.getnframes()
            t_rate = t_f.getframerate()
            ted_len = t_frames / float(t_rate)

        with contextlib.closing(wave.open(myvoice,'r')) as m_f:
            m_frames = m_f.getnframes()
            m_rate = m_f.getframerate()
            my_len = m_frames / float(m_rate)


        A = np.array(t_graph_list)
        B = np.array(m_graph_list)
        _A = np.array(t_graph_list2)
        _B = np.array(m_graph_list2)
        _cost, _path = DTW(A, B, window = 100)
        offset = 200
        __m.Pitchplot([float(x) * ted_len / len(t_graph_list) for x in range(0, len(t_graph_list))], A, [float(x) * my_len / len(m_graph_list) for x in range(0, len(m_graph_list))], B+offset, A,B,_path,offset, y_min, y_min2,_A,_B)

        with contextlib.closing(wave.open(ted,'r')) as t_f:
            t_frames = t_f.getnframes()
            t_rate = t_f.getframerate()
            ted_len = t_frames / float(t_rate)

        with contextlib.closing(wave.open(myvoice,'r')) as m_f:
            m_frames = m_f.getnframes()
            m_rate = m_f.getframerate()
            my_len = m_frames / float(m_rate)

        tmp=False
        probability_text = "스크립트 : " + str(len(common_list)) + " / " + str(tot_len) + " dtw 거리 : " + str(tot_dis)
        print(probability_text)
        tot_per=0.6*(100*(len(common_list)/tot_len))+0.4*(1-(((tot_dis+(ted_len-my_len)*12.491855842)-150)/100))*100
        f_per = int((100*(len(common_list)/tot_len)))
        b_per = 100*(1-(((tot_dis+(ted_len-my_len)*12.491855842)-150)/100))
        b_per2 = int(b_per)
        if tot_per < 0:
            tot_per = 0
        if abs(ted_len - my_len) > 4:
            tmp = True

        if tmp == True:
            pro.setText("정확한 테스트를 위하여 다시 한번 녹음해 주세요.")
        else:
            pro.setText("발음유사도:"+str(f_per)+"(%) 성조유사도:"+str(b_per2)+"(%) 전체유사도:"+str(int(tot_per))+"(%)")
            print("전체스크립트:",total_script)
            print("유효스트링:",len(common_list),"/",tot_len)
            print(cmd)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle('plastique')
    main = asset()
    main.resize(1070,700)
    main.show()
    sys.exit(app.exec_())
