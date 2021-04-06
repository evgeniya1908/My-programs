# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import sys
import xlwt
import xlrd
import pandas as pd
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import (QApplication, QMainWindow, QMessageBox)
from PyQt5.QtCore import QAbstractTableModel, Qt
import random
import math
#import matplotlib.pyplot as plt
import scipy.stats as stat
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import numpy as np



class MplCanvas(FigureCanvasQTAgg):

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)


class pandasModel(QAbstractTableModel):

    def __init__(self, data):
        QAbstractTableModel.__init__(self)
        Id = [i for i in range(1, data.shape[0] + 1)]
        self._data = pd.DataFrame()
        self._data['Id'] = Id
        liss = list(data)
        for i in range (data.shape[1]):
            self._data[liss[i]] = data.values[:,i] 
        
        
    def rowCount(self, parent=None):
        return self._data.shape[0]

    def columnCount(self, parnet=None):
        return self._data.shape[1]

    def data(self, index, role=Qt.DisplayRole):
        if index.isValid():
            if role == Qt.DisplayRole:
                return str(self._data.iloc[index.row(), index.column()])
        return None

    def headerData(self, col, orientation, role):
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            return self._data.columns[col]
        return None


class Stat(QMainWindow):
    
    def __init__(self):
        self.fname = "СМО-курс.xls"
        self.snX = "Равномерный"
        self.snY = "Нормальный"
        self.X = []
        self.Y = []
        self.X10 = []
        self.Y10 = []
        self.groupX = pd.DataFrame()
        self.pointsX = pd.DataFrame()
        self.conf_intMGX = pd.DataFrame()
        self.conf_intMTX = pd.DataFrame()
        self.conf_intDGX = pd.DataFrame()
        self.conf_intDTX = pd.DataFrame()
        self.groupY = pd.DataFrame()
        self.pointsY = pd.DataFrame()
        self.conf_intMGY = pd.DataFrame()
        self.conf_intMTY = pd.DataFrame()
        self.conf_intDGY = pd.DataFrame()
        self.conf_intDTY = pd.DataFrame()
        self.agreeX = pd.DataFrame()
        self.agreeY = pd.DataFrame()
        self.hArrX = []
        self.hArrY = []
        self.intervalX = []
        self.intervalY = []
        self.TX9p = []
        self.TX95p = []
        self.TX99p = []
        self.GX9p = []
        self.GX95p = []
        self.GX99p = []
        self.TX9m = []
        self.TX95m = []
        self.TX99m = []
        self.GX9m = []
        self.GX95m = []
        self.GX99m = []
        self.TY9p = []
        self.TY95p = []
        self.TY99p = []
        self.GY9p = []
        self.GY95p = []
        self.GY99p = []
        self.TY9m = []
        self.TY95m = []
        self.TY99m = []
        self.GY9m = []
        self.GY95m = []
        self.GY99m = []
        self.Dx = []
        self.Mx = []
        self.Dy = []
        self.My = []
        self.dTX9p = []
        self.dTX95p = []
        self.dTX99p = []
        self.dGX9p = []
        self.dGX95p = []
        self.dGX99p = []
        self.dTX9m = []
        self.dTX95m = []
        self.dTX99m = []
        self.dGX9m = []
        self.dGX95m = []
        self.dGX99m = []
        self.dTY9p = []
        self.dTY95p = []
        self.dTY99p = []
        self.dGY9p = []
        self.dGY95p = []
        self.dGY99p = []
        self.dTY9m = []
        self.dTY95m = []
        self.dTY99m = []
        self.dGY9m = []
        self.dGY95m = []
        self.dGY99m = []
        self.DMX = pd.DataFrame()
        self.DMY = pd.DataFrame()
        self.ZR = pd.DataFrame()
        self.ZN = pd.DataFrame()
        self.TPX = []
        self.TPY = []
        self.TKY = []
        self.TKX = []
    
    
    def initX(self, n):
        self.x = []
        for i in range(n):
            self.R = random.random()
            self.x.append(round(1.154*self.R+0.42265, 4))
        
        return self.x
    
    
    def initY(self, n):
        self.y = []
        for i in range(n):
            self.R = 0
            for j in range(300):
                self.R = self.R + random.random()
            self.y.append(round(self.R/15-9, 4))
        
        return self.y
    
    
    def variationRange (self, f):
       return sorted(f)
   
    
    def intervalSeries (self, f, n):
        maxi = max(f)
        mini = min(f)
        r = maxi - mini
        k = 1 + 3.32*math.log10(n)
        k = math.ceil(k)
        h = round(r/k, 4)
        hArr = [] #массив частот
        sh = mini #для подсчета попадения в интервал
        midl =  [] #массив средних значений каждого интервала
        interval = []
        
        m = 0
        i = 0
        while i < n-1:
            if f[i] <= sh+h:
                m = m + 1
                i= i+1
                if i == n-1:
                    m = m+1
                    interval.append(round(sh,2))
                    hArr.append(m)
                    midl.append(round((sh+h/2),2))

            else:
                interval.append(round(sh,2))
                hArr.append(m)
                midl.append(round((sh+h/2),2))
                m = 0
                sh = sh+h
        interval.append(maxi)
        return r, interval, hArr , midl 
        
    
    def point (self, f, n, midl, hArr):
        m1 = sum(f)/n
        sum_el = 0
        for i in range(len(hArr)):
            sum_el = sum_el + midl[i]*hArr[i]
        m2 = sum_el/n
        D = 0
        for elem in f:
            D = D+(elem - m1)**2
        D = D/n
        D2 = D*n/(n-1)
        ass = stat.skew(f)
        exc = stat.kurtosis(f)
        var = (D**0.5)/m1
        return m1, m2, D, D2, ass, exc, var 
    

    
    def intervalM1 (self, M, disp, N):
        E = math.sqrt(disp/N)      
        fn1 =  stat.norm.interval(0.9,loc=M, scale = E)
        fn2 =  stat.norm.interval(0.95,loc=M, scale = E)
        fn3 =  stat.norm.interval(0.99,loc=M, scale = E)
        return fn1, fn2, fn3
    
    
    def intervalM2 (self, M, disp, N):
        E = math.sqrt(disp/N)
        ft1 =  stat.t.interval(0.9, N-1, loc=M, scale = E)
        ft2 =  stat.t.interval(0.95, N-1, loc=M, scale = E)
        ft3 =  stat.t.interval(0.99, N-1, loc=M, scale = E)
        return ft1, ft2, ft3
    
    
    def  intervalDN1 (self, disp, D, N):
        scal = math.sqrt(2*(D**2)/(N-1))
        fgd1 = stat.norm.interval(0.9, loc = disp, scale = scal)
        fgd2 = stat.norm.interval(0.95, loc = disp, scale = scal)
        fgd3 = stat.norm.interval(0.99, loc = disp, scale = scal)
        return fgd1, fgd2, fgd3
    
    def  intervalDR1 (self, disp, D, N):
        scal = math.sqrt((0.8*N+1.2)/(N*(N-1)))*D
        fgd1 = stat.norm.interval(0.9, loc = disp, scale = scal)
        fgd2 = stat.norm.interval(0.95, loc = disp, scale = scal)
        fgd3 = stat.norm.interval(0.99, loc = disp, scale = scal)
        return fgd1, fgd2, fgd3
    
    def intervalD2 (self, disp2, N):
        ftd1 = (N*disp2/stat.chi2.ppf(0.95, N), N*disp2/stat.chi2.ppf((0.05),N))
        ftd2 = (N*disp2/stat.chi2.ppf(0.95, N), N*disp2/stat.chi2.ppf((0.05),N))
        ftd3 = (N*disp2/stat.chi2.ppf(0.99, N), N*disp2/stat.chi2.ppf((0.01),N))
        return ftd1, ftd2, ftd3
    

    def creatF (self):
        self.fname = "СМО-курс.xls"
        book = xlwt.Workbook(encoding="utf-8")

        # Add a sheet to the workbook 
        sheet1 = book.add_sheet(self.snX)
        sheet2 = book.add_sheet(self.snY)
        
        # Write to the sheet of the workbook 
        for i in range(10):
            n = (i+1)*100
            x = self.initX(n)
            y = self.initY(n)
            self.X.append(x)
            self.Y.append(y)
            for l in range(n):
                sheet1.write(l, i, x[l])
                sheet2.write(l,i,y[l])
                
        for i in range(10,19):
            n = 1000
            x = self.initX(n)
            y = self.initY(n)
            self.X10.append(x)
            self.Y10.append(y)
            for l in range(n):
                sheet1.write(l, i, x[l])
                sheet2.write(l,i,y[l])
        
        # Save the workbook 
        book.save(self.fname)
        QMessageBox.critical(self, "Звiт", "Файл створен, можна аналiзувати данi", QMessageBox.Ok)
        
    

    def readF (self):
        #try:
        rb = xlrd.open_workbook(self.fname, formatting_info=True)
        sht1 = rb.sheet_by_index(0)
        sht2 = rb.sheet_by_index(1)
        for i in range(10):
            k = (i+1)*100
            x = []
            for j in range (k):
                x.append(sht1.cell_value(j, i))
            self.X.append(x)
                
        for i in range(9, 19):
            x = []
            for j in range (1000):
                x.append(sht1.cell_value(j, i))
            self.X10.append(x)
                
        for i in range(10):
            k = (i+1)*100
            y = []
            for j in range (k):
                y.append(sht2.cell_value(j, i))
            self.Y.append(y)
                
        for i in range(9,19):
            y = []
            for j in range (1000):
                y.append(sht2.cell_value(j, i))
            self.Y10.append(x)
        
        #QMessageBox.critical(self, "Звiт", "Данi завантаженi", QMessageBox.Ok)
        #except:
            #QMessageBox.critical(self, "Помилка ", "Файл не містить таблицю", QMessageBox.Ok)
         
    
    def M_D_10 (self, A):
        aver = np.mean(A) 
        disp = np.var(A) 
        return aver, disp

    
    def PirsonX (self, interv,  m, N, n):
        xn = []
        p = []
        np = []
        xi2 = 0
        i = 0
        mi = []
        u = 1/n
        while i < n-1:
            if m[i] < 5 and i < n-1:
                mi.append(m[i]+m[i+1])
                p.append(2*u)
                np.append(N*u*2)
                xn.append(str(interv[i])+'-'+str(interv[i+2]))
                i = i+1
            else:
                mi.append(m[i])
                p.append(u)
                np.append(u*N)
                xn.append(str(interv[i])+'-'+str(interv[i+1]))
            i= i+1
            if m[i] > 5 and i == n-1:
                mi.append(m[i])
                p.append(u)
                np.append(u*N)
                xn.append(str(interv[i])+'-'+str(interv[i+1]))
            elif m[i] < 5 and i == n-1:
                l = len(mi) - 1
                mi[l] = m[i-1] + m[i]
                p[l] = 2*u
                np[l] = N*u*2
                xn[l] = str(interv[i-1])+'-'+str(interv[i+1])
                i = i+1  
            
        k = len(mi)
        for j in range(k):
            xi2 = xi2 + ((mi[j]-np[j])**2)/np[j]
        result = ''
        xi2pr = stat.chi2.ppf(0.95, k-3)
        if xi2 < xi2pr:
            result = 'H0'
        else:
            result = 'H1'
        mi.append(sum(mi))
        xn.append('Sum')
        p.append(sum(p))
        np.append(sum(np))
        df = pd.DataFrame({'Nx':xn, 'mi':mi, 'pi':p, 'npi':np})
        return  xi2, xi2pr, result,df
        
    
    def PirsonY (self, X, m, D, M, N):
        xn = []
        d2 = (D*N/(N-1))**0.5
        p = []
        pi = []
        h = len(X)-1
        for i in range(h):
            p1 = 0
            p2 = 0
            f1 = (X[i+1]- M)/d2
            p1 = stat.norm.cdf(f1)
            f2 = (X[i]- M)/d2
            p2 = stat.norm.cdf(f2)
            p.append(p1-p2)
        i = 0
        n = len(m)
        mi = []
        np = []
        while i <= n-1:
            if m[i] < 5 and  i == 0:
                mi.append(m[i]+m[i+1])
                pi.append(p[i]+p[i+1])
                np.append(N*(p[i]+p[i+1]))
                xn.append(str(X[i])+'-'+str(X[i+2]))
                i = i+1
            else:
                mi.append(m[i])
                pi.append(p[i])
                np.append(p[i]*N)
                xn.append(str(X[i])+'-'+str(X[i+1]))    
            if i < n-1:
                i= i+1
            if m[i] < 5 and i <= n-1: 
                l = len(mi) -1
                mi[l] = m[i-1] + m[i]
                pi[l] = p[i]+p[i-1]
                np[l] = N*(p[i]+p[i-1])
                xn[l] = str(X[i-1])+'-'+str(X[i+1])
                i = i+1
            elif m[i] >= 5 and i == n-1:
                mi.append(m[i])
                pi.append(p[i])
                np.append(p[i]*N)
                xn.append(str(X[i])+'-'+str(X[i+1]))
                
        k = len(mi)
        xi2 = 0
        for j in range(k):
            xi2 = xi2 + ((mi[j]-np[j])**2)/np[j]
        result = ''
        xi2pr = stat.chi2.ppf(0.95, k-3)
        if xi2 < xi2pr:
            result = 'H0'
        else:
            result = 'H1'
        mi.append(sum(mi))
        xn.append('Sum')
        pi.append(sum(pi))
        np.append(sum(np))
        df = pd.DataFrame({'Nx':xn, 'mi':mi, 'pi':pi, 'npi':np})
        return  xi2, xi2pr, result, df
    
    
    def KolmogorovY (self, m, x, M, D, N):
        d = D**0.5
        k = len(m)
        f1 = 0
        F = []
        for i in range(k):
            f1 = m[i]/N + f1
            f2 = (x[i+1]-M)/d
            f2 = stat.norm.cdf(f2)
            F.append(abs(f1 - f2))
        maxF = max(F)
        la = (N**0.5)*maxF
        result = ''
        if la < 1.36:
            result = 'H0'
        else:
            result = 'H1'
        return  la, result
        
    def KolmogorovX (self, m, n):
        k = len(m)
        f2 = 1/k
        F = []
        for i in range(k):
            f1 = m[i]/n 
            F.append(abs(f1-f2))
        maxF = max(F)
        la = (n**0.5)*maxF
        result = ''
        if la < 1.36:
            result = 'H0'
        else:
            result = 'H1'
        return  la, result
    
    
    
    def analiz(self):
        #Rx = []
        #Ry = []
        
        
        self.pointsX['Параметр'] = ['M1', 'M2', 'D1', 'D2', 'A', 'E', 'V', 'R']
        self.pointsY['Параметр'] = ['M1', 'M2', 'D1', 'D2', 'A', 'E', 'V', 'R']
        self.DMX['Параметр'] = ['M', 'D']
        self.DMY['Параметр'] = ['M', 'D']
        self.ZR['Крiтерiй'] = ['Пiрсона', 'Xi2',  'Гiпотеза', 'Коломогорова', 'Гiпотеза']
        self.ZN['Крiтерiй'] = ['Пiрсона', 'Xi2', 'Гiпотеза', 'Коломогорова', 'Гiпотеза']

        for i in range(10):
            #ArrX = []
            #ArrY = []
            x = self.variationRange(self.X[i])
            y = self.variationRange(self.Y[i])
            n = (i+1)*100
            rx, intervX, hArX, midlX = self.intervalSeries(x,n)
            ry, intervY, hArY, midlY = self.intervalSeries(y,n)
            kx = len(hArX)
    
            mx1, mx2, dx1, dx2, ax, ex, varx = self.point(self.X[i], n, midlX, hArX)
            my1, my2, dy1, dy2, ay, ey, vary = self.point(self.Y[i], n, midlY, hArY)
            
            xi2x, xi2prx, resx, dfx = self.PirsonX(intervX, hArX, n, kx) 
            self.TPX.append(dfx)
            lax, resx2 = self.KolmogorovX(hArX, n)
            self.ZR[str(n)] = [xi2x, xi2prx, resx, lax, resx2]
            
            xi2y, xi2pry, resy, dfy = self.PirsonY(intervY, hArY, dy1, my1, n) 
            self.TPY.append(dfy)
            lay, resy2 = self.KolmogorovY(hArY, intervY, my1, dy1, n)
            self.ZN[str(n)] = [xi2y, xi2pry, resy, lay, resy2]
            
            mx10, dx10 = self.M_D_10(self.X10[i])
            my10, dy10 = self.M_D_10(self.Y10[i])
            #Rx.append(rx)
            #Ry.append(ry)
            self.Mx.append(mx1)
            self.Dx.append(dx1)
            self.My.append(my1)
            self.Dy.append(dy1)
            self.pointsX[str(n)] = [mx1, mx2, dx1, dx2, ax, ex, varx, rx]
            self.pointsY[str(n)] = [my1, my2, dy1, dy2, ay, ey, vary, ry]
            self.DMX[str(i+1)] = [mx10, dx10]
            self.DMY[str(i+1)] = [my10, dy10]
            #ArrX.append()
            self.hArrX.append(hArX)
            self.hArrY.append(hArY)
            self.intervalX.append(intervX)
            self.intervalY.append(intervY)
            vx = len(hArX)
            vy = len(hArY)
            if vy == 8:
                intervX.append(0)
                intervX.append(0)
                intervX.append(0)
                hArX.append(0)
                hArX.append(0)
                hArX.append(0)
            if vx == 9:
                intervX.append(0)
                intervX.append(0)
                hArX.append(0)
                hArX.append(0)
            elif vx == 10:
                intervX.append(0)
                hArX.append(0)
            if vy == 8:
                intervY.append(0)
                intervY.append(0)
                intervY.append(0)
                hArY.append(0)
                hArY.append(0)
                hArY.append(0)
            elif vy == 9:
                intervY.append(0)
                intervY.append(0)
                hArY.append(0)
                hArY.append(0)
            elif vy == 10:
                intervY.append(0)
                hArY.append(0)
           
            lx = len(intervX) 
            self.groupX['- ' + str(n)] = intervX[0:lx-1]
            self.groupX['+ ' + str(n)] = intervX[1:lx]
            self.groupX['N' + str(n)] = hArX
            ly = len(intervY) 
            self.groupY['- ' + str(n)] = intervY[0:ly-1]
            self.groupY['+ ' + str(n)] = intervY[1:ly]
            self.groupY['N' + str(n)] = hArY
            
        maxDx = self.Dx[9]
        maxDy = self.Dy[9]
        self.conf_intMGX['A'] = ['0,9', '0,95', '0,99']
        self.conf_intMGY['A'] = ['0,9', '0,95', '0,99']
        self.conf_intMTX['A'] = ['0,9', '0,95', '0,99']
        self.conf_intMTY['A'] = ['0,9', '0,95', '0,99']
        for i in range(10):
            n = (i+1)*100
            gx1, gx2, gx3 = self.intervalM1(self.Mx[i], maxDx, n)
            gy1, gy2, gy3 = self.intervalM1(self.My[i], maxDy, n)
            tx1, tx2, tx3 = self.intervalM2(self.Mx[i], self.Dx[i], n)
            ty1, ty2, ty3 = self.intervalM2(self.My[i], self.Dy[i], n)
            self.conf_intMGX[str(n)+ '-'] = [gx1[0], gx2[0], gx3[0]] 
            self.conf_intMGX[str(n)+ '+'] = [gx1[1], gx2[1], gx3[1]]
            self.conf_intMTX[str(n)+ '-'] = [tx1[0], tx2[0], tx3[0]] 
            self.conf_intMTX[str(n)+ '+'] = [tx1[1], tx2[1], tx3[1]]
            self.conf_intMGY[str(n)+ '-'] = [gy1[0], gy2[0], gy3[0]] 
            self.conf_intMGY[str(n)+ '+'] = [gy1[1], gy2[1], gy3[1]]
            self.conf_intMTY[str(n)+ '-'] = [ty1[0], ty2[0], ty3[0]] 
            self.conf_intMTY[str(n)+ '+'] = [ty1[1], ty2[1], ty3[1]]
            self.TX9p.append(tx1[1])
            self.TX95p.append(tx2[1])
            self.TX99p.append(tx3[1])
            self.GX9p.append(gx1[1])
            self.GX95p.append(gx2[1])
            self.GX99p.append(gx3[1])
            self.TX9m.append(tx1[0])
            self.TX95m.append(tx2[0])
            self.TX99m.append(tx3[0])
            self.GX9m.append(gx1[0])
            self.GX95m.append(gx2[0])
            self.GX99m.append(gx3[0])
            self.TY9p.append(ty1[1])
            self.TY95p.append(ty2[1])
            self.TY99p.append(ty3[1])
            self.GY9p.append(gy1[1])
            self.GY95p.append(gy2[1])
            self.GY99p.append(gy3[1])
            self.TY9m.append(ty1[0])
            self.TY95m.append(ty2[0])
            self.TY99m.append(ty3[0])
            self.GY9m.append(gy1[0])
            self.GY95m.append(gy2[0])
            self.GY99m.append(gy3[0])
        self.conf_intDGX['A'] = ['0,9', '0,95', '0,99']
        self.conf_intDGY['A'] = ['0,9', '0,95', '0,99']
        self.conf_intDTX['A'] = ['0,9', '0,95', '0,99']
        self.conf_intDTY['A'] = ['0,9', '0,95', '0,99']
        for i in range(10):
            n = (i+1)*100
            gx1, gx2, gx3 = self.intervalDR1(self.Dx[i], maxDx, n)
            gy1, gy2, gy3 = self.intervalDN1(self.Dy[i], maxDy, n)
            tx1, tx2, tx3 = self.intervalD2(self.Dx[i], n)
            ty1, ty2, ty3 = self.intervalD2(self.Dy[i], n)
            self.conf_intDGX[str(n)+ '-'] = [gx1[0], gx2[0], gx3[0]] 
            self.conf_intDGX[str(n)+ '+'] = [gx1[1], gx2[1], gx3[1]]
            self.conf_intDTX[str(n)+ '-'] = [tx1[0], tx2[0], tx3[0]] 
            self.conf_intDTX[str(n)+ '+'] = [tx1[1], tx2[1], tx3[1]]
            self.conf_intDGY[str(n)+ '-'] = [gy1[0], gy2[0], gy3[0]] 
            self.conf_intDGY[str(n)+ '+'] = [gy1[1], gy2[1], gy3[1]]
            self.conf_intDTY[str(n)+ '-'] = [ty1[0], ty2[0], ty3[0]] 
            self.conf_intDTY[str(n)+ '+'] = [ty1[1], ty2[1], ty3[1]]
            self.dTX9p.append(tx1[1])
            self.dTX95p.append(tx2[1])
            self.dTX99p.append(tx3[1])
            self.dGX9p.append(gx1[1])
            self.dGX95p.append(gx2[1])
            self.dGX99p.append(gx3[1])
            self.dTX9m.append(tx1[0])
            self.dTX95m.append(tx2[0])
            self.dTX99m.append(tx3[0])
            self.dGX9m.append(gx1[0])
            self.dGX95m.append(gx2[0])
            self.dGX99m.append(gx3[0])
            self.dTY9p.append(ty1[1])
            self.dTY95p.append(ty2[1])
            self.dTY99p.append(ty3[1])
            self.dGY9p.append(gy1[1])
            self.dGY95p.append(gy2[1])
            self.dGY99p.append(gy3[1])
            self.dTY9m.append(ty1[0])
            self.dTY95m.append(ty2[0])
            self.dTY99m.append(ty3[0])
            self.dGY9m.append(gy1[0])
            self.dGY95m.append(gy2[0])
            self.dGY99m.append(gy3[0])
                
            
            
            
        
        
class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1236, 703)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.creatButton = QtWidgets.QPushButton(self.centralwidget)
        self.creatButton.setGeometry(QtCore.QRect(10, 10, 111, 31))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.creatButton.setFont(font)
        self.creatButton.setObjectName("creatButton")
        self.readButton = QtWidgets.QPushButton(self.centralwidget)
        self.readButton.setGeometry(QtCore.QRect(130, 10, 111, 31))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.readButton.setFont(font)
        self.readButton.setObjectName("readButton")
        self.analysisButton = QtWidgets.QPushButton(self.centralwidget)
        self.analysisButton.setGeometry(QtCore.QRect(250, 10, 111, 31))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.analysisButton.setFont(font)
        self.analysisButton.setObjectName("analysisButton")
        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        self.tabWidget.setGeometry(QtCore.QRect(0, 40, 1251, 641))
        self.tabWidget.setObjectName("tabWidget")
        self.tab = QtWidgets.QWidget()
        self.tab.setObjectName("tab")
        self.tableInterval = QtWidgets.QTableView(self.tab)
        self.tableInterval.setGeometry(QtCore.QRect(10, 0, 1211, 321))
        self.tableInterval.setObjectName("tableInterval")
        self.gridLayoutWidget = QtWidgets.QWidget(self.tab)
        self.gridLayoutWidget.setGeometry(QtCore.QRect(420, 330, 731, 281))
        self.gridLayoutWidget.setObjectName("gridLayoutWidget")
        self.gridLayout = QtWidgets.QGridLayout(self.gridLayoutWidget)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setObjectName("gridLayout")
        self.canvas  = MplCanvas(self, width=5, height=4, dpi=100)
        self.gridLayout.addWidget(self.canvas)
        self.label_3 = QtWidgets.QLabel(self.tab)
        self.label_3.setGeometry(QtCore.QRect(280, 330, 81, 31))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")
        self.spinBox = QtWidgets.QSpinBox(self.tab)
        self.spinBox.setGeometry(QtCore.QRect(290, 390, 61, 31))
        self.spinBox.setMinimum(100)
        self.spinBox.setMaximum(1000)
        self.spinBox.setSingleStep(100)
        self.spinBox.setObjectName("spinBox")
        self.tabWidget.addTab(self.tab, "")
        self.tab_2 = QtWidgets.QWidget()
        self.tab_2.setObjectName("tab_2")
        self.tablePoint = QtWidgets.QTableView(self.tab_2)
        self.tablePoint.setGeometry(QtCore.QRect(0, 30, 1221, 311))
        self.tablePoint.setObjectName("tablePoint")
        self.label_2 = QtWidgets.QLabel(self.tab_2)
        self.label_2.setGeometry(QtCore.QRect(420, 10, 171, 16))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.label_8 = QtWidgets.QLabel(self.tab_2)
        self.label_8.setGeometry(QtCore.QRect(300, 420, 331, 20))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_8.setFont(font)
        self.label_8.setObjectName("label_8")
        self.tableMDX = QtWidgets.QTableView(self.tab_2)
        self.tableMDX.setGeometry(QtCore.QRect(-10, 450, 1231, 151))
        self.tableMDX.setObjectName("tableMDX")
        self.tabWidget.addTab(self.tab_2, "")
        self.tab_3 = QtWidgets.QWidget()
        self.tab_3.setObjectName("tab_3")
        self.tableMG = QtWidgets.QTableView(self.tab_3)
        self.tableMG.setGeometry(QtCore.QRect(10, 40, 601, 191))
        self.tableMG.setObjectName("tableMG")
        self.tableMT = QtWidgets.QTableView(self.tab_3)
        self.tableMT.setGeometry(QtCore.QRect(620, 40, 601, 191))
        self.tableMT.setObjectName("tableMT")
        self.label_4 = QtWidgets.QLabel(self.tab_3)
        self.label_4.setGeometry(QtCore.QRect(90, 10, 421, 21))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_4.setFont(font)
        self.label_4.setObjectName("label_4")
        self.label_5 = QtWidgets.QLabel(self.tab_3)
        self.label_5.setGeometry(QtCore.QRect(730, 10, 421, 21))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_5.setFont(font)
        self.label_5.setObjectName("label_5")
        self.gridLayoutWidget_2 = QtWidgets.QWidget(self.tab_3)
        self.gridLayoutWidget_2.setGeometry(QtCore.QRect(10, 240, 601, 341))
        self.gridLayoutWidget_2.setObjectName("gridLayoutWidget_2")
        self.gridLayout_MG = QtWidgets.QGridLayout(self.gridLayoutWidget_2)
        self.gridLayout_MG.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_MG.setObjectName("gridLayout_MG")
        self.canvas3  = MplCanvas(self, width=5, height=4, dpi=100)
        self.gridLayout_MG.addWidget(self.canvas3)
        self.gridLayoutWidget_3 = QtWidgets.QWidget(self.tab_3)
        self.gridLayoutWidget_3.setGeometry(QtCore.QRect(620, 240, 601, 341))
        self.gridLayoutWidget_3.setObjectName("gridLayoutWidget_3")
        self.gridLayout_MT = QtWidgets.QGridLayout(self.gridLayoutWidget_3)
        self.gridLayout_MT.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_MT.setObjectName("gridLayout_MT")
        self.canvas4  = MplCanvas(self, width=5, height=4, dpi=100)
        self.gridLayout_MT.addWidget(self.canvas4)
        self.tabWidget.addTab(self.tab_3, "")
        self.tab_4 = QtWidgets.QWidget()
        self.tab_4.setObjectName("tab_4")
        self.label_6 = QtWidgets.QLabel(self.tab_4)
        self.label_6.setGeometry(QtCore.QRect(90, 10, 421, 21))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_6.setFont(font)
        self.label_6.setObjectName("label_6")
        self.tableDG = QtWidgets.QTableView(self.tab_4)
        self.tableDG.setGeometry(QtCore.QRect(10, 40, 601, 191))
        self.tableDG.setObjectName("tableDG")
        self.gridLayoutWidget_4 = QtWidgets.QWidget(self.tab_4)
        self.gridLayoutWidget_4.setGeometry(QtCore.QRect(10, 240, 601, 351))
        self.gridLayoutWidget_4.setObjectName("gridLayoutWidget_4")
        self.gridLayout_DG = QtWidgets.QGridLayout(self.gridLayoutWidget_4)
        self.gridLayout_DG.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_DG.setObjectName("gridLayout_DG")
        self.canvas5  = MplCanvas(self, width=5, height=4, dpi=100)
        self.gridLayout_DG.addWidget(self.canvas5)
        self.tableDT = QtWidgets.QTableView(self.tab_4)
        self.tableDT.setGeometry(QtCore.QRect(620, 40, 601, 191))
        self.tableDT.setObjectName("tableDT")
        self.gridLayoutWidget_5 = QtWidgets.QWidget(self.tab_4)
        self.gridLayoutWidget_5.setGeometry(QtCore.QRect(620, 240, 601, 351))
        self.gridLayoutWidget_5.setObjectName("gridLayoutWidget_5")
        self.gridLayout_DT = QtWidgets.QGridLayout(self.gridLayoutWidget_5)
        self.gridLayout_DT.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_DT.setObjectName("gridLayout_DT")
        self.canvas6  = MplCanvas(self, width=5, height=4, dpi=100)
        self.gridLayout_DT.addWidget(self.canvas6)
        self.label_7 = QtWidgets.QLabel(self.tab_4)
        self.label_7.setGeometry(QtCore.QRect(710, 10, 421, 21))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_7.setFont(font)
        self.label_7.setObjectName("label_7")
        self.tabWidget.addTab(self.tab_4, "")
        self.tab_5 = QtWidgets.QWidget()
        self.tab_5.setObjectName("tab_5")
        self.tableInterval_2 = QtWidgets.QTableView(self.tab_5)
        self.tableInterval_2.setGeometry(QtCore.QRect(10, 0, 1191, 321))
        self.tableInterval_2.setObjectName("tableInterval_2")
        self.label_21 = QtWidgets.QLabel(self.tab_5)
        self.label_21.setGeometry(QtCore.QRect(230, 340, 81, 31))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_21.setFont(font)
        self.label_21.setObjectName("label_21")
        self.gridLayoutWidget_11 = QtWidgets.QWidget(self.tab_5)
        self.gridLayoutWidget_11.setGeometry(QtCore.QRect(360, 330, 711, 281))
        self.gridLayoutWidget_11.setObjectName("gridLayoutWidget_11")
        self.gridLayout_3 = QtWidgets.QGridLayout(self.gridLayoutWidget_11)
        self.gridLayout_3.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.canvas2  = MplCanvas(self, width=5, height=4, dpi=100)
        self.gridLayout_3.addWidget(self.canvas2)
        self.spinBox_3 = QtWidgets.QSpinBox(self.tab_5)
        self.spinBox_3.setGeometry(QtCore.QRect(240, 390, 61, 31))
        self.spinBox_3.setMinimum(100)
        self.spinBox_3.setMaximum(1000)
        self.spinBox_3.setSingleStep(100)
        self.spinBox_3.setObjectName("spinBox_3")
        self.tabWidget.addTab(self.tab_5, "")
        self.tab_6 = QtWidgets.QWidget()
        self.tab_6.setObjectName("tab_6")
        self.label_19 = QtWidgets.QLabel(self.tab_6)
        self.label_19.setGeometry(QtCore.QRect(550, 10, 171, 16))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_19.setFont(font)
        self.label_19.setObjectName("label_19")
        self.tablePoint_3 = QtWidgets.QTableView(self.tab_6)
        self.tablePoint_3.setGeometry(QtCore.QRect(0, 30, 1221, 321))
        self.tablePoint_3.setObjectName("tablePoint_3")
        self.label_9 = QtWidgets.QLabel(self.tab_6)
        self.label_9.setGeometry(QtCore.QRect(530, 420, 331, 20))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_9.setFont(font)
        self.label_9.setObjectName("label_9")
        self.tableMDY = QtWidgets.QTableView(self.tab_6)
        self.tableMDY.setGeometry(QtCore.QRect(0, 440, 1221, 151))
        self.tableMDY.setObjectName("tableMDY")
        self.tabWidget.addTab(self.tab_6, "")
        self.tab_7 = QtWidgets.QWidget()
        self.tab_7.setObjectName("tab_7")
        self.tableMG_3 = QtWidgets.QTableView(self.tab_7)
        self.tableMG_3.setGeometry(QtCore.QRect(10, 40, 601, 191))
        self.tableMG_3.setObjectName("tableMG_3")
        self.gridLayoutWidget_12 = QtWidgets.QWidget(self.tab_7)
        self.gridLayoutWidget_12.setGeometry(QtCore.QRect(10, 240, 601, 351))
        self.gridLayoutWidget_12.setObjectName("gridLayoutWidget_12")
        self.gridLayout_MG_3 = QtWidgets.QGridLayout(self.gridLayoutWidget_12)
        self.gridLayout_MG_3.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_MG_3.setObjectName("gridLayout_MG_3")
        self.canvas7  = MplCanvas(self, width=5, height=4, dpi=100)
        self.gridLayout_MG_3.addWidget(self.canvas7)
        self.label_20 = QtWidgets.QLabel(self.tab_7)
        self.label_20.setGeometry(QtCore.QRect(90, 10, 421, 21))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_20.setFont(font)
        self.label_20.setObjectName("label_20")
        self.tableMT_3 = QtWidgets.QTableView(self.tab_7)
        self.tableMT_3.setGeometry(QtCore.QRect(620, 40, 601, 191))
        self.tableMT_3.setObjectName("tableMT_3")
        self.gridLayoutWidget_13 = QtWidgets.QWidget(self.tab_7)
        self.gridLayoutWidget_13.setGeometry(QtCore.QRect(620, 240, 601, 351))
        self.gridLayoutWidget_13.setObjectName("gridLayoutWidget_13")
        self.gridLayout_MT_3 = QtWidgets.QGridLayout(self.gridLayoutWidget_13)
        self.gridLayout_MT_3.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_MT_3.setObjectName("gridLayout_MT_3")
        self.canvas8  = MplCanvas(self, width=5, height=4, dpi=100)
        self.gridLayout_MT_3.addWidget(self.canvas8)
        self.tabWidget.addTab(self.tab_7, "")
        self.tab_8 = QtWidgets.QWidget()
        self.tab_8.setObjectName("tab_8")
        self.tableDG_3 = QtWidgets.QTableView(self.tab_8)
        self.tableDG_3.setGeometry(QtCore.QRect(10, 40, 601, 201))
        self.tableDG_3.setObjectName("tableDG_3")
        self.gridLayoutWidget_14 = QtWidgets.QWidget(self.tab_8)
        self.gridLayoutWidget_14.setGeometry(QtCore.QRect(10, 250, 601, 341))
        self.gridLayoutWidget_14.setObjectName("gridLayoutWidget_14")
        self.gridLayout_DG_3 = QtWidgets.QGridLayout(self.gridLayoutWidget_14)
        self.gridLayout_DG_3.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_DG_3.setObjectName("gridLayout_DG_3")
        self.canvas9  = MplCanvas(self, width=5, height=4, dpi=100)
        self.gridLayout_DG_3.addWidget(self.canvas9)
        self.tableDT_3 = QtWidgets.QTableView(self.tab_8)
        self.tableDT_3.setGeometry(QtCore.QRect(620, 40, 601, 201))
        self.tableDT_3.setObjectName("tableDT_3")
        self.gridLayoutWidget_15 = QtWidgets.QWidget(self.tab_8)
        self.gridLayoutWidget_15.setGeometry(QtCore.QRect(620, 250, 601, 341))
        self.gridLayoutWidget_15.setObjectName("gridLayoutWidget_15")
        self.gridLayout_DT_3 = QtWidgets.QGridLayout(self.gridLayoutWidget_15)
        self.gridLayout_DT_3.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_DT_3.setObjectName("gridLayout_DT_3")
        self.canvas10  = MplCanvas(self, width=5, height=4, dpi=100)
        self.gridLayout_DT_3.addWidget(self.canvas10)
        self.label_23 = QtWidgets.QLabel(self.tab_8)
        self.label_23.setGeometry(QtCore.QRect(110, 10, 421, 21))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_23.setFont(font)
        self.label_23.setObjectName("label_23")
        self.label_24 = QtWidgets.QLabel(self.tab_8)
        self.label_24.setGeometry(QtCore.QRect(760, 20, 421, 21))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_24.setFont(font)
        self.label_24.setObjectName("label_24")
        self.tabWidget.addTab(self.tab_8, "")
        self.tab_9 = QtWidgets.QWidget()
        self.tab_9.setObjectName("tab_9")
        self.tableZR = QtWidgets.QTableView(self.tab_9)
        self.tableZR.setGeometry(QtCore.QRect(0, 20, 1221, 151))
        self.tableZR.setObjectName("tableZR")
        self.label_10 = QtWidgets.QLabel(self.tab_9)
        self.label_10.setGeometry(QtCore.QRect(430, 0, 431, 16))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_10.setFont(font)
        self.label_10.setObjectName("label_10")
        self.label_12 = QtWidgets.QLabel(self.tab_9)
        self.label_12.setGeometry(QtCore.QRect(60, 190, 121, 16))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_12.setFont(font)
        self.label_12.setObjectName("label_12")
        self.label_13 = QtWidgets.QLabel(self.tab_9)
        self.label_13.setGeometry(QtCore.QRect(650, 190, 161, 16))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_13.setFont(font)
        self.label_13.setObjectName("label_13")
        self.horizontalLayoutWidget = QtWidgets.QWidget(self.tab_9)
        self.horizontalLayoutWidget.setGeometry(QtCore.QRect(10, 220, 1211, 381))
        self.horizontalLayoutWidget.setObjectName("horizontalLayoutWidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.tablePX = QtWidgets.QTableView(self.horizontalLayoutWidget)
        self.tablePX.setObjectName("tablePX")
        self.horizontalLayout.addWidget(self.tablePX)
        self.tableKX = QtWidgets.QTableView(self.horizontalLayoutWidget)
        self.tableKX.setObjectName("tableKX")
        self.horizontalLayout.addWidget(self.tableKX)
        self.spinBox_2 = QtWidgets.QSpinBox(self.tab_9)
        self.spinBox_2.setGeometry(QtCore.QRect(180, 180, 61, 31))
        self.spinBox_2.setMinimum(100)
        self.spinBox_2.setMaximum(1000)
        self.spinBox_2.setSingleStep(100)
        self.spinBox_2.setObjectName("spinBox_2")
        self.spinBox_4 = QtWidgets.QSpinBox(self.tab_9)
        self.spinBox_4.setGeometry(QtCore.QRect(810, 180, 61, 31))
        self.spinBox_4.setMinimum(100)
        self.spinBox_4.setMaximum(1000)
        self.spinBox_4.setSingleStep(100)
        self.spinBox_4.setObjectName("spinBox_4")
        self.tabWidget.addTab(self.tab_9, "")
        self.tab_11 = QtWidgets.QWidget()
        self.tab_11.setObjectName("tab_11")
        self.label_11 = QtWidgets.QLabel(self.tab_11)
        self.label_11.setGeometry(QtCore.QRect(460, 10, 431, 16))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_11.setFont(font)
        self.label_11.setObjectName("label_11")
        self.tableZN = QtWidgets.QTableView(self.tab_11)
        self.tableZN.setGeometry(QtCore.QRect(10, 30, 1221, 151))
        self.tableZN.setObjectName("tableZN")
        self.horizontalLayoutWidget_2 = QtWidgets.QWidget(self.tab_11)
        self.horizontalLayoutWidget_2.setGeometry(QtCore.QRect(9, 229, 1211, 371))
        self.horizontalLayoutWidget_2.setObjectName("horizontalLayoutWidget_2")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget_2)
        self.horizontalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.tablePY = QtWidgets.QTableView(self.horizontalLayoutWidget_2)
        self.tablePY.setObjectName("tablePY")
        self.horizontalLayout_2.addWidget(self.tablePY)
        self.tableKY = QtWidgets.QTableView(self.horizontalLayoutWidget_2)
        self.tableKY.setObjectName("tableKY")
        self.horizontalLayout_2.addWidget(self.tableKY)
        self.label_14 = QtWidgets.QLabel(self.tab_11)
        self.label_14.setGeometry(QtCore.QRect(650, 200, 161, 16))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_14.setFont(font)
        self.label_14.setObjectName("label_14")
        self.spinBox_5 = QtWidgets.QSpinBox(self.tab_11)
        self.spinBox_5.setGeometry(QtCore.QRect(170, 190, 61, 31))
        self.spinBox_5.setMinimum(100)
        self.spinBox_5.setMaximum(1000)
        self.spinBox_5.setSingleStep(100)
        self.spinBox_5.setObjectName("spinBox_5")
        self.spinBox_6 = QtWidgets.QSpinBox(self.tab_11)
        self.spinBox_6.setGeometry(QtCore.QRect(800, 190, 61, 31))
        self.spinBox_6.setMinimum(100)
        self.spinBox_6.setMaximum(1000)
        self.spinBox_6.setSingleStep(100)
        self.spinBox_6.setObjectName("spinBox_6")
        self.label_15 = QtWidgets.QLabel(self.tab_11)
        self.label_15.setGeometry(QtCore.QRect(40, 200, 121, 16))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_15.setFont(font)
        self.label_15.setObjectName("label_15")
        self.tabWidget.addTab(self.tab_11, "")
        self.tab_10 = QtWidgets.QWidget()
        self.tab_10.setObjectName("tab_10")
        self.tableKorAn = QtWidgets.QTableView(self.tab_10)
        self.tableKorAn.setGeometry(QtCore.QRect(5, 51, 1221, 251))
        self.tableKorAn.setObjectName("tableKorAn")
        self.label_16 = QtWidgets.QLabel(self.tab_10)
        self.label_16.setGeometry(QtCore.QRect(480, 20, 161, 16))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_16.setFont(font)
        self.label_16.setObjectName("label_16")
        self.label_17 = QtWidgets.QLabel(self.tab_10)
        self.label_17.setGeometry(QtCore.QRect(490, 320, 161, 16))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_17.setFont(font)
        self.label_17.setObjectName("label_17")
        self.tableRegAn = QtWidgets.QTableView(self.tab_10)
        self.tableRegAn.setGeometry(QtCore.QRect(10, 340, 1221, 251))
        self.tableRegAn.setObjectName("tableRegAn")
        self.tabWidget.addTab(self.tab_10, "")
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.creatButton.setText(_translate("MainWindow", "Створити файл"))
        self.readButton.setText(_translate("MainWindow", "Вiдкрити файл"))
        self.analysisButton.setText(_translate("MainWindow", "Проаналiзувати"))
        self.label_3.setText(_translate("MainWindow", "Гiстограма"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), _translate("MainWindow", "Iнт. ряд Х"))
        self.label_2.setText(_translate("MainWindow", "Точковi оцiнки та розмах"))
        self.label_8.setText(_translate("MainWindow", "Точковi оцiнки в залежностi вiд експерименту"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_2), _translate("MainWindow", "Точк. оц. Х"))
        self.label_4.setText(_translate("MainWindow", "Довiрковий iнтервал матиматичного очiкування грубий метод"))
        self.label_5.setText(_translate("MainWindow", "Довiрковий iнтервал матиматичного очiкування точний метод"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_3), _translate("MainWindow", "Дов. iнт. м. о. Х"))
        self.label_6.setText(_translate("MainWindow", "Довiрковий iнтервал дисперсii грубий метод"))
        self.label_7.setText(_translate("MainWindow", "Довiрковий iнтервал дисперсii точний метод"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_4), _translate("MainWindow", "Дов. iнт. д. Х"))
        self.label_21.setText(_translate("MainWindow", "Гiстограма"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_5), _translate("MainWindow", "Iнт. ряд У"))
        self.label_19.setText(_translate("MainWindow", "Точковi оцiнки та розмах"))
        self.label_9.setText(_translate("MainWindow", "Точковi оцiнки в залежностi вiд експерименту"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_6), _translate("MainWindow", "Точ. оц. У"))
        self.label_20.setText(_translate("MainWindow", "Довiрковий iнтервал матиматичного очiкування грубий метод"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_7), _translate("MainWindow", "Дов. iнт. м. о. У"))
        self.label_23.setText(_translate("MainWindow", "Довiрковий iнтервал дисперсii грубий метод"))
        self.label_24.setText(_translate("MainWindow", "Довiрковий iнтервал дисперсii точний метод"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_8), _translate("MainWindow", "Дов. iнт. д. У"))
        self.label_10.setText(_translate("MainWindow", "Перевiрка на вiдповiднiсть до рiвномiрного закону розподiлу"))
        self.label_12.setText(_translate("MainWindow", "Крiтерiй Пiрсона"))
        self.label_13.setText(_translate("MainWindow", "Крiтерiй Колмогорова"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_9), _translate("MainWindow", "Стат. гiпотеза X"))
        self.label_11.setText(_translate("MainWindow", "Перевiрка на вiдповiднiсть до нормального закону розподiлу"))
        self.label_14.setText(_translate("MainWindow", "Крiтерiй Колмогорова"))
        self.label_15.setText(_translate("MainWindow", "Крiтерiй Пiрсона"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_11), _translate("MainWindow", "Стат. гiпотеза Y"))
        self.label_16.setText(_translate("MainWindow", "Кореляцiйний аналiз"))
        self.label_17.setText(_translate("MainWindow", "Регресiйний аналiз"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_10), _translate("MainWindow", "Кор. аналiз"))




class PresenterMain(QMainWindow, Ui_MainWindow):
    
    def __init__(self):
        super(PresenterMain, self).__init__()  
        self.setupUi(self)
        self.obj = Stat()
        self.creatButton.pressed.connect(self.obj.creatF)
        self.readButton.pressed.connect(self.read)
        self.analysisButton.pressed.connect(self.analysis)
        self.spinBox.valueChanged.connect(self.newBarX)
        self.spinBox_3.valueChanged.connect(self.newBarY)
        self.spinBox_2.valueChanged.connect(self.newTabPX)
        self.spinBox_5.valueChanged.connect(self.newTabPY)
        
    def newBarY(self):
        n = int(self.spinBox_3.value()/100 - 1)
        k = 0
        if n == 0:
            k = 8
        elif n == 1:
            k = 9
        elif 2<=n<=4:
            k = 10
        else:
            k = 11
        self.canvas2.axes.clear()
        y = range(len(self.obj.intervalY[n][0:k]))
        self.canvas2.axes.bar(y,self.obj.hArrY[n][0:k], align = 'edge')
        self.canvas2.axes.set_xticks(y)
        self.canvas2.axes.set_xticklabels(self.obj.intervalY[n][0:k])
        self.canvas2.draw()
        
        
    def newBarX(self):
        n = int(self.spinBox.value()/100 - 1)
        k = 0
        if n == 0:
            k = 8
        elif n == 1:
            k = 9
        elif 2<=n<=4:
            k = 10
        else:
            k = 11
        self.canvas.axes.clear()
        x = range(len(self.obj.intervalX[n][0:k]))
        self.canvas.axes.bar(x,self.obj.hArrX[n][0:k], align = 'edge')
        self.canvas.axes.set_xticks(x)
        self.canvas.axes.set_xticklabels(self.obj.intervalX[n][0:k])
        self.canvas.draw()
        
        
    def newTabPX(self):
        n = int(self.spinBox_2.value()/100 - 1)
        self.model17 = pandasModel(self.obj.TPX[n])
        self.tablePX.setModel(self.model17)
        
        
    def newTabPY(self):
        n = int(self.spinBox_5.value()/100 - 1)
        self.model18 = pandasModel(self.obj.TPY[n])
        self.tablePY.setModel(self.model18)
    
    
    def read(self):
        self.obj.readF()
    
    def analysis(self):
        self.obj.analiz()
        model1 = pandasModel(self.obj.groupX)
        self.tableInterval.setModel(model1)
        model2 = pandasModel(self.obj.groupY)
        self.tableInterval_2.setModel(model2)
        x = range(len(self.obj.intervalX[0][0:8]))
        self.canvas.axes.bar(x,self.obj.hArrX[0][:8], align = 'edge')
        self.canvas.axes.set_xticks(x)
        self.canvas.axes.set_xticklabels(self.obj.intervalX[0][0:8])
        self.canvas.draw()
        y = range(len(self.obj.intervalY[0][0:8]))
        self.canvas2.axes.bar(y,self.obj.hArrY[0][:8], align = 'edge')
        self.canvas2.axes.set_xticks(y)
        self.canvas2.axes.set_xticklabels(self.obj.intervalY[0][0:8])
        self.canvas2.draw()
        model3 = pandasModel(self.obj.pointsX)
        self.tablePoint.setModel(model3)
        model4 = pandasModel(self.obj.pointsY)
        self.tablePoint_3.setModel(model4)
        
        model5 = pandasModel(self.obj.conf_intMGX)
        self.tableMG.setModel(model5)
        model6 = pandasModel(self.obj.conf_intMTX)
        self.tableMT.setModel(model6)
        model7 = pandasModel(self.obj.conf_intDGX)
        self.tableDG.setModel(model7)
        model8 = pandasModel(self.obj.conf_intDTX)
        self.tableDT.setModel(model8)
        
        model9 = pandasModel(self.obj.conf_intMGY)
        self.tableMG_3.setModel(model9)
        model10 = pandasModel(self.obj.conf_intMTY)
        self.tableMT_3.setModel(model10)
        model11 = pandasModel(self.obj.conf_intDGY)
        self.tableDG_3.setModel(model11)
        model12 = pandasModel(self.obj.conf_intDTY)
        self.tableDT_3.setModel(model12)
        
        model13 = pandasModel(self.obj.DMX)
        self.tableMDX.setModel(model13)
        model14 = pandasModel(self.obj.DMY)
        self.tableMDY.setModel(model14)
        
        x3 = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
        
        self.canvas3.axes.plot(x3, self.obj.GX9m, marker = 'o', color = 'seagreen', label = '0,9')
        self.canvas3.axes.plot(x3, self.obj.GX9p, marker = 'o', color = 'seagreen')
        self.canvas3.axes.plot(x3, self.obj.GX95m, marker = 's', color = 'blue', label = '0,95')
        self.canvas3.axes.plot(x3, self.obj.GX95p, marker = 's', color = 'blue')
        self.canvas3.axes.plot(x3, self.obj.GX99m, marker = '^', color = 'orange', label = '0,99')
        self.canvas3.axes.plot(x3, self.obj.GX99p, marker = '^', color = 'orange')
        self.canvas3.axes.plot(x3, self.obj.Mx,  marker = 'X', color = 'red')
        self.canvas3.axes.legend(bbox_to_anchor=(0.8, 1), loc='upper left', borderaxespad=0.)
        
        self.canvas4.axes.plot(x3, self.obj.TX9m, marker = 'o', color = 'seagreen', label = '0,9')
        self.canvas4.axes.plot(x3, self.obj.TX9p, marker = 'o', color = 'seagreen')
        self.canvas4.axes.plot(x3, self.obj.TX95m, marker = 's', color = 'blue', label = '0,95')
        self.canvas4.axes.plot(x3, self.obj.TX95p, marker = 's', color = 'blue')
        self.canvas4.axes.plot(x3, self.obj.TX99m, marker = '^', color = 'orange', label = '0,99')
        self.canvas4.axes.plot(x3, self.obj.TX99p, marker = '^', color = 'orange')
        self.canvas4.axes.plot(x3, self.obj.Mx,  marker = 'X', color = 'red')
        self.canvas4.axes.legend(bbox_to_anchor=(0.8, 1), loc='upper left', borderaxespad=0.)
        
        self.canvas5.axes.plot(x3, self.obj.dGX9m, marker = 'o', color = 'seagreen', label = '0,9')
        self.canvas5.axes.plot(x3, self.obj.dGX9p, marker = 'o', color = 'seagreen')
        self.canvas5.axes.plot(x3, self.obj.dGX95m, marker = 's', color = 'blue', label = '0,95')
        self.canvas5.axes.plot(x3, self.obj.dGX95p, marker = 's', color = 'blue')
        self.canvas5.axes.plot(x3, self.obj.dGX99m, marker = '^', color = 'orange', label = '0,99')
        self.canvas5.axes.plot(x3, self.obj.dGX99p, marker = '^', color = 'orange')
        self.canvas5.axes.plot(x3, self.obj.Dx,  marker = 'X', color = 'red')
        self.canvas5.axes.legend(bbox_to_anchor=(0.8, 1), loc='upper left', borderaxespad=0.)
        
        self.canvas6.axes.plot(x3, self.obj.dTX9m, marker = 'o', color = 'seagreen', label = '0,9')
        self.canvas6.axes.plot(x3, self.obj.dTX9p, marker = 'o', color = 'seagreen')
        self.canvas6.axes.plot(x3, self.obj.dTX95m, marker = 's', color = 'blue', label = '0,95')
        self.canvas6.axes.plot(x3, self.obj.dTX95p, marker = 's', color = 'blue')
        self.canvas6.axes.plot(x3, self.obj.dTX99m, marker = '^', color = 'orange', label = '0,99')
        self.canvas6.axes.plot(x3, self.obj.dTX99p, marker = '^', color = 'orange')
        self.canvas6.axes.plot(x3, self.obj.Dx,  marker = 'X', color = 'red')
        self.canvas6.axes.legend(bbox_to_anchor=(0.8, 1), loc='upper left', borderaxespad=0.)
        
        
        self.canvas7.axes.plot(x3, self.obj.GY9m, marker = 'o', color = 'seagreen', label = '0,9')
        self.canvas7.axes.plot(x3, self.obj.GY9p, marker = 'o', color = 'seagreen')
        self.canvas7.axes.plot(x3, self.obj.GY95m, marker = 's', color = 'blue', label = '0,95')
        self.canvas7.axes.plot(x3, self.obj.GY95p, marker = 's', color = 'blue')
        self.canvas7.axes.plot(x3, self.obj.GY99m, marker = '^', color = 'orange', label = '0,99')
        self.canvas7.axes.plot(x3, self.obj.GY99p, marker = '^', color = 'orange')
        self.canvas7.axes.plot(x3, self.obj.My,  marker = 'X', color = 'red')
        self.canvas7.axes.legend(bbox_to_anchor=(0.8, 1), loc='upper left', borderaxespad=0.)
        
        self.canvas8.axes.plot(x3, self.obj.TY9m, marker = 'o', color = 'seagreen', label = '0,9')
        self.canvas8.axes.plot(x3, self.obj.TY9p, marker = 'o', color = 'seagreen')
        self.canvas8.axes.plot(x3, self.obj.TY95m, marker = 's', color = 'blue', label = '0,95')
        self.canvas8.axes.plot(x3, self.obj.TY95p, marker = 's', color = 'blue')
        self.canvas8.axes.plot(x3, self.obj.TY99m, marker = '^', color = 'orange', label = '0,99')
        self.canvas8.axes.plot(x3, self.obj.TY99p, marker = '^', color = 'orange')
        self.canvas8.axes.plot(x3, self.obj.My,  marker = 'X', color = 'red')
        self.canvas8.axes.legend(bbox_to_anchor=(0.8, 1), loc='upper left', borderaxespad=0.)
        
        self.canvas9.axes.plot(x3, self.obj.dGY9m, marker = 'o', color = 'seagreen', label = '0,9')
        self.canvas9.axes.plot(x3, self.obj.dGY9p, marker = 'o', color = 'seagreen')
        self.canvas9.axes.plot(x3, self.obj.dGY95m, marker = 's', color = 'blue', label = '0,95')
        self.canvas9.axes.plot(x3, self.obj.dGY95p, marker = 's', color = 'blue')
        self.canvas9.axes.plot(x3, self.obj.dGY99m, marker = '^', color = 'orange', label = '0,99')
        self.canvas9.axes.plot(x3, self.obj.dGY99p, marker = '^', color = 'orange')
        self.canvas9.axes.plot(x3, self.obj.Dy,  marker = 'X', color = 'red')
        self.canvas9.axes.legend(bbox_to_anchor=(0.8, 1), loc='upper left', borderaxespad=0.)
        
        self.canvas10.axes.plot(x3, self.obj.dTY9m, marker = 'o', color = 'seagreen', label = '0,9')
        self.canvas10.axes.plot(x3, self.obj.dTY9p, marker = 'o', color = 'seagreen')
        self.canvas10.axes.plot(x3, self.obj.dTY95m, marker = 's', color = 'blue', label = '0,95')
        self.canvas10.axes.plot(x3, self.obj.dTY95p, marker = 's', color = 'blue')
        self.canvas10.axes.plot(x3, self.obj.dTY99m, marker = '^', color = 'orange', label = '0,99')
        self.canvas10.axes.plot(x3, self.obj.dTY99p, marker = '^', color = 'orange')
        self.canvas10.axes.plot(x3, self.obj.Dy,  marker = 'X', color = 'red')
        self.canvas10.axes.legend(bbox_to_anchor=(0.8, 1), loc='upper left', borderaxespad=0.)
        
        
        model15 = pandasModel(self.obj.ZR)
        self.tableZR.setModel(model15)
        
        model16 = pandasModel(self.obj.ZN)
        self.tableZN.setModel(model16)
        
        self.model17 = pandasModel(self.obj.TPX[0])
        self.tablePX.setModel(self.model17)
        self.model18 = pandasModel(self.obj.TPY[0])
        self.tablePY.setModel(self.model18)

def main():
    app = QApplication(sys.argv)
    #appctxt = ApplicationContext()
    window = PresenterMain()
    window.show()
    sys.exit(app.exec_())
    #sys.exit(appctxt.app.exec_())

if __name__ == "__main__":
    main()        
        
        
       
        



            
        
def test():
    N = 1000
    st = Stat()
    X = st.initX(N)
    s = 0
    for i in X:
        s = s+i
    s = s/N
    #print(X)
    print(s)
    X1 = st.variationRange(X)
    #print(X1)
    R, inter, hArr, midl = st.intervalSeries(X1, N)
    print(R)
    print(inter)
    print(hArr)
    print(midl)
    st.barChart(hArr, midl)
    m1, m2, D, D2 = st.point(X1, N, midl, hArr)
    print(m1)
    print(m2)
    print(D)
    print(D2)
    fn1, fn2, fn3 = st.intervalM1(m1, D, N)
    print(fn1)
    print(fn2)
    print(fn3)
    ft1, ft2, ft3 = st.intervalM2(m1, D, N)
    print(ft1)
    print(ft2)
    print(ft3)

            
    
    
    
        
