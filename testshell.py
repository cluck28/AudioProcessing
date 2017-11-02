#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed May  3 13:47:53 2017

Audio visualizer by listening with the microphone

Trying to construct the FFT in discrete steps across the frames

Lots of hardcoded numbers so try to keep track

Right now weird factor of two when considering rate

Plots the spectogram for the frames

@author: Chris
"""

import pyaudio
import numpy
import wave
import time
import matplotlib.pyplot as plt

'''
CHUNK = 2**11
RATE = 44100

p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16,channels=1,rate=RATE,input=True,
              frames_per_buffer=CHUNK)

for i in range(int(10*44100/1024)): #go for a few seconds
    data = np.fromstring(stream.read(CHUNK),dtype=np.int16)
    peak = np.average(np.abs(data))*2
    #bars="#"*int(50*peak/2**16)
    print("%04d %05d %s"%(i,peak,bars))
    print("%04d %05d"%(i,peak))

stream.stop_stream()
stream.close()
p.terminate()
'''

def record_audio(time,filename):
    CHUNK = 1024
    FORMAT = pyaudio.paFloat32
    CHANNELS = 2
    RATE = 44100 #Frames per second
    RECORD_SECONDS = time
    WAVE_OUTPUT_FILENAME = filename
    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

    print 'Recording...'
    frames = []
    frames_pt = []
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        decoded = numpy.fromstring(data,dtype=numpy.float32)
        frames.extend(decoded)
        frames_pt.append(data)
    print 'done'
    
    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames_pt))
    wf.close()
    return frames
    
def read_recording(filename):
    CHUNK = 1024
    wf = wave.open(filename, 'rb')
    p = pyaudio.PyAudio()
    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                channels=wf.getnchannels(),
                rate=wf.getframerate(),
                output=True)
    data = wf.readframes(CHUNK)
    frames = []
    while data != '':
        stream.write(data)
        data = wf.readframes(CHUNK)
        decoded = numpy.fromstring(data,dtype=numpy.float32)
        frames.extend(decoded)
    stream.stop_stream()
    stream.close()
    p.terminate()
    return frames
    
def play_recording(filename):
    CHUNK = 1024
    wf = wave.open(filename, 'rb')
    p = pyaudio.PyAudio()
    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                channels=wf.getnchannels(),
                rate=wf.getframerate(),
                output=True)
    data = wf.readframes(CHUNK)
    while data != '':
        stream.write(data)
        data = wf.readframes(CHUNK)
    stream.stop_stream()
    stream.close()
    p.terminate()
    
def fft_frames(frames,RATE):
    Y = numpy.fft.fft(frames)/len(frames)
    k = numpy.arange(len(frames))*RATE/len(frames)
    k = k[range(len(frames)/2)]
    Y = Y[range(len(frames)/2)]
    return k, Y

def plot_spectogram(mat):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    plt.imshow(mat, interpolation='nearest', cmap=plt.cm.terrain, aspect='auto')
    plt.colorbar()
    plt.ylim([0,150])
    plt.show()
    
def plot_timeseries(times,frames,freq,fft):
    fig, ax = plt.subplots(2, 1)
    ax[0].plot(times,frames)
    ax[0].set_xlabel('Time')
    ax[0].set_ylabel('Amplitude')
    ax[1].plot(freq,abs(fft),'r')
    ax[1].set_xlabel('Freq (Hz)')
    ax[1].set_ylabel('|Y(freq)|')
    plt.show()
    
    
if __name__ == '__main__':
    #frames = record_audio(10, 'fileout.wav')
    #times = numpy.arange(0,len(frames))/44100./2.
    #Read in audio
    frames = read_recording('fileout_test.wav')
    times = numpy.arange(0,len(frames))/44100./2.
    #Zoom in on the times
    time_step = int(0.02/(1/44100./2.))
    fft_list = []
    times_list = []
    frames_list = []
    for i in range(0,len(frames)/time_step):
        frames_zoom = frames[i*time_step:(i+1)*time_step]
        frames_list.append(frames_zoom)
        times_zoom = times[i*time_step:(i+1)*time_step]
        times_list.append(times_zoom)
        freq, fft = fft_frames(frames_zoom, 44100/2.)
        fft_list.append(fft)
    #Plot
    plot_timeseries(times_list[50],frames_list[50],freq,fft_list[50])
    
    out_mat = numpy.zeros((len(fft_list[0]),len(fft_list)))
    for i in range(len(fft_list[0])):
        for j in range(len(fft_list)):
            out_mat[i,j] = abs(fft_list[j][i])
            
    #Plot
    plot_spectogram(out_mat)       
     
    #I want to find the start of each note
    #Look for where the amplitudes in the fft are biggest
    #Search by columns [:,0] gives zeroth column [0,:] gives zeroth row
    amplist = []
    for i in range(len(out_mat[0,:])):
        amplist.append( numpy.sum(out_mat[:,i]) )
    times = numpy.arange(len(amplist))
    
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    plt.scatter(times,amplist)
    plt.show()
    
    