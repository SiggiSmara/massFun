#!/usr/bin/env python
# -*- coding: utf-8 -*-

from copy import deepcopy
import numpy as np
from scipy import signal

def upsample_cubicSpline(xdata,ydata,numIncreas=2):
    cs = CubicSpline(xdata, ydata)
    newX=[]
    newY=[]
    for i in range(len(xdata)-1):
        x = np.linspace(xdata[i], xdata[i+1], numIncreas*2)
        newX=np.append(newX,x)
        newY=np.append(newY,cs(x))
    return (newX,newY)

def upsample_linear(xdata,ydata,numIncreas=2):
    x = np.linspace(0, 2*np.pi, 10)
    newX=[]
    for i in range(len(xdata)-1):
        x = np.linspace(xdata[i], xdata[i+1], numIncreas*2)
        newX=np.append(newX,x)
    newY=np.interp(newX,xdata,ydata)
    return (newX,newY)

def find_lstsqLine(x,y):
    # find a line model for these points
    A = np.vstack([x, np.ones(len(x))]).T
    return np.linalg.lstsq(A,y)[0]

def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

def find_closest(xnparr, xFind):
    y=abs(xnparr-xFind)
    return np.argwhere(y==min(y)).flatten()[0]

def find_rangeMax(xlist, fromIdx, toIdx):
    return np.argwhere(xlist[fromIdx:toIdx]==max(xlist[fromIdx:toIdx])).flatten()[0]+fromIdx

def find_localMins(xdict):
    xkeys=list()
    x1=list()
    for oneK in sorted(xdict.keys()):
        xkeys.append(oneK)
        x1.append(x[oneK])
    x1=np.array(x1)
    xdata=rolling_window(x1,3)
    minDict=dict()
    for i in range(xdata.shape[0]):
        oneX=xdata[i,:]
        if oneX[1]==min(oneX):
            minDict[xkeys[i]]=oneX[1]
    return minDict

def find_localMaxes(xdict):
    xkeys=list()
    x1=list()
    for oneK in sorted(xdict.keys()):
        xkeys.append(oneK)
        x1.append(x[oneK])
    x1=np.array(x1)
    xdata=rolling_window(x1,3)
    minDict=dict()
    for i in range(xdata.shape[0]):
        oneX=xdata[i,:]
        if oneX[1]==max(oneX):
            minDict[xkeys[i]]=oneX[1]
    return minDict

def get_movingSdev(x,sdev_width):
    if sdev_width % 2 == 0:
        print "stdev width has to be an odd number"
        return x
    movDev=np.std(rolling_window(x,sdev_width),1)
    # adding the begin and end numbers to have the 
    # size the same as the original array
    tails=np.ones((sdev_width-1)/2)
    movDev=np.append(movDev[0]*tails,movDev)
    movDev=np.append(movDev,movDev[movDev.size-1]*tails)
    return movDev

def get_medianDiffarr(xnparr):
    if xnparr.size<2:
        if xnparr.size==1:
            return [0]
        else:
            return []
    else:
        dx=xnparr[1:]-xnparr[0:-1]
        dx=np.append(dx[0],dx)
    return abs(dx-np.median(dx))


def make_weights(centerIdx,fromIdx,toIdx):
    sigmadata=list()
    for i in range(fromIdx,toIdx):
        if i!=centerIdx:
            sigmadata.append(1.0/abs(i-centerIdx))
        else:
            sigmadata.append(1)
    return np.array(sigmadata)

def filter_movingAverage (values, window):
    cumsum_vec = np.cumsum(np.insert(values, 0, 0)) 
    ma_vec = (cumsum_vec[window:] - cumsum_vec[:-window]) / window
    adds=(window-1)/2
    ma_vec=np.append(np.ones(adds)*ma_vec[0],ma_vec)
    ma_vec=np.append(ma_vec,np.ones(adds)*ma_vec[-1])
    return ma_vec

def filter_movingWeightedAverage(values, window):
    adds=(window-1)/2
    weights=make_weights(adds,0,window)
    retVec=np.average(rolling_window(values,window),1,weights=weights)
    retVec= np.append(np.ones(adds)*retVec[0],retVec)
    retVec= np.append(retVec,np.ones(adds)*retVec[-1])
    return retVec

def filter_movingMedian(values, window):
    return signal.medfilt(values,window)

def filter_savitskygolay(values, window, polyorder=2):   
    return signal.savgol_filter(values,window,polyorder)

def function_gauss(x,a,x0,sigma):
        return a*np.exp(-(x-x0)**2/(2.*sigma**2))

def function_bigauss(x,a,x0,sigma1,sigma2):
    y=[function_gauss(onex,a,x0,sigma1) if onex<x0 else function_gauss(onex,a,x0,sigma2) for onex in x]
    return np.array(y)