#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys, os
import numpy as np
from scipy import linalg

sys.path.append(sys.path.join("..","..",".."))
from  massFun.GeneralDataFunctions import rolling_window,moving_sdev,local_minimums,median_diffarr

class baselineCorrection:

    # def rolling_window(self,a, window):
    #     shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    #     strides = a.strides + (a.strides[-1],)
    #     return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

    # def moving_sdev(self,x,sdev_width):
    #     if sdev_width % 2 == 0:
    #         print "stdev width has to be an odd number"
    #         return x

    #     movDev=np.std(self.rolling_window(x,sdev_width),1)

    #     # adding the begin and end numbers to have the 
    #     # size the same as the original array
    #     tails=np.ones((sdev_width-1)/2)
    #     movDev=np.append(movDev[0]*tails,movDev)
    #     movDev=np.append(movDev,movDev[movDev.size-1]*tails)
    #     return movDev

    def stage1_fujchrom2016(self,x):
        return local_minimums(x)

    def stage2_fujchrom2016(self,x,window):
        x1=self.medSNR_elim(x,window)
        x2=self.firstDeriv_elim(x)
        retDict=dict()
        for i in x1:
            retDict[i]=min(x1[i],x2[i])
        return retDict

    def stage3_fujchrom2016(self,x,y,st2wind):
        st1=self.stage1_fujchrom2016(y)
        st2=self.stage2_fujchrom2016(st1,st2wind)
        xf=list()
        yf=list()
        for oneX in sorted(st2):
            xf.append(oneX)
            yf.append(st2[oneX])
        baseline=np.interp(x,xf,yf)
        baseline=np.minimum(baseline,y)
        return baseline

    # def local_minimums(self,x):
    #     x=np.array(x)
    #     xdata=self.rolling_window(x,3)
    #     minDict=dict()
    #     for i in range(xdata.shape[0]):
    #         oneX=xdata[i,:]
    #         #print oneX
    #         if oneX[1]==min(oneX):
    #             minDict[i+1]=oneX[1]
    #     return minDict

    # def median_diffarr(self,x):
    #     dx=x[1:]-x[0:-1]
    #     dx=np.append(dx[0],dx)
    #     return abs(dx-np.median(dx))

    def medSNR_elim(self,x,window=30,prevResult=np.Inf):
        xkeys=x.keys()
        x1=np.array(x.values())
        medx=np.median(x1)
        medDiffArr=median_diffarr(x1)
        sigma=1.483*np.median(medDiffArr)
        xdata=rolling_window(x1,window)

        beginshape=xdata.shape
        addp=(x1.shape[0]-xdata.shape[0])/2
        
        for n in range(addp):
            xdata=np.append(xdata[0,:],xdata)
        xdata.shape=(beginshape[0]+addp,beginshape[1])
        for n in range(addp):
            xdata=np.append(xdata,xdata[-1,:])
        xdata.shape=(beginshape[0]+2*addp,beginshape[1])
        retArr=dict()

        for i in range(1,x1.size-1):
            uno=np.abs(x1[i]-np.median(xdata[i]))/sigma
            due=np.abs(x1[i]-x1[i-1])/sigma
            tre=np.abs(x1[i]-x1[i+1])/sigma
            SNRi = max(uno,due,tre)
            if SNRi>2.5:
                retArr[xkeys[i]]=np.interp(xkeys[i],(xkeys[i-1],xkeys[i+1]),(x1[i-1],x1[i+1]))
            else:
                retArr[xkeys[i]]=x1[i]
        retArr[xkeys[0]]=retArr[xkeys[1]]
        retArr[xkeys[-1]]=retArr[xkeys[-2]]

        result=np.linalg.norm(np.array(retArr.values())-x1)/np.linalg.norm(x1)
        if result==prevResult:
            return retArr
        elif result > 1e-4:
            retArr=self.medSNR_elim(retArr,window, result)
            return retArr
        return x

    def firstDeriv_elim(self,x):
        xkeys=x.keys()
        x1=np.array(x.values())
        medx=np.median(x1)
        medDiffArr=median_diffarr(x1)/x1
        retArr=dict()
        for i in range(x1.size):
            #print medDiffArr[i]
            if medDiffArr[i]>2.5:
                if i==0:
                    retArr[xkeys[i]]=x1[i+1]
                elif i==x1.size-1:
                    retArr[xkeys[i]]=x1[i-1]
                else:
                    retArr[xkeys[i]]=np.interp(xkeys[i],(xkeys[i-1],xkeys[i+1]),(x1[i-1],x1[i+1]))
            else:
                retArr[xkeys[i]]=x1[i]
        return retArr




