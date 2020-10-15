#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as LA
import scipy.signal as ss
import operator

# Functions
def array_response_vector(array,theta):
    N = array.shape
    v = np.exp(1j*2*np.pi*array*np.sin(theta))
    return v/np.sqrt(N)

def music(CovMat,L,N,array,Angles):
    # CovMat is the signal covariance matrix, L is the number of sources, N is the number of antennas
    # array holds the positions of antenna elements
    # Angles are the grid of directions in the azimuth angular domain
    _,V = LA.eig(CovMat)
    Qn  = V[:,L:N]
    numAngles = Angles.size
    pspectrum = np.zeros(numAngles)
    for i in range(numAngles):
        av = array_response_vector(array,Angles[i])
        pspectrum[i] = 1/LA.norm((Qn.conj().transpose()@av))
    psindB    = np.log10(10*pspectrum/pspectrum.min())
    DoAsMUSIC,_= ss.find_peaks(psindB,height=1.35, distance=1.5)
    return DoAsMUSIC,pspectrum

def esprit(CovMat,L,N):
    # CovMat is the signal covariance matrix, L is the number of sources, N is the number of antennas
    _,U = LA.eig(CovMat)
    S = U[:,0:L]
    Phi = LA.pinv(S[0:N-1]) @ S[1:N] # the original array is divided into two subarrays [0,1,...,N-2] and [1,2,...,N-1]
    eigs,_ = LA.eig(Phi)
    DoAsESPRIT = np.arcsin(np.angle(eigs)/np.pi)
    return DoAsESPRIT
#=============================================================

np.random.seed(6)

lamda = 1 # wavelength
kappa = np.pi/lamda # wave number
L = 5  # number of sources
N = 32  # number of ULA elements 
snr = 10 # signal to noise ratio


array = np.linspace(0,(N-1)/2,N)

plt.figure()
plt.subplot(221)
plt.plot(array,np.zeros(N),'^')
plt.title('Uniform Linear Array')
plt.legend(['Antenna'])


Thetas = np.pi*(np.random.rand(L)-1/2)   # random source directions
Alphas = np.random.randn(L) + np.random.randn(L)*1j # random source powers
Alphas = np.sqrt(1/2)*Alphas
#print(Thetas)
#print(Alphas)

h = np.zeros(N)
for i in range(L):
    h = h + Alphas[i]*array_response_vector(array,Thetas[i])

Angles = np.linspace(-np.pi/2,np.pi/2,360)
numAngles = Angles.size

hv = np.zeros(numAngles)
for j in range(numAngles):
    av = array_response_vector(array,Angles[j])
    hv[j] = np.abs(np.inner(h,av.conj()))

powers = np.zeros(L)
for j in range(L):
    av = array_response_vector(array,Thetas[j])
    powers[j] = np.abs(np.inner(h,av.conj()))

plt.subplot(222)
plt.plot(Angles,hv)
plt.plot(Thetas,powers,'*')
plt.title('Correlation')
plt.legend(['Correlation power','Actual DoAs'])
numrealization = 100
H = np.zeros((N,numrealization)) + 1j*np.zeros((N,numrealization))

for iter in range(numrealization):
    htmp = np.zeros(N)
    for i in range(L):
        pha = np.exp(1j*2*np.pi*np.random.rand(1))
        htmp = htmp + pha*Alphas[i]*array_response_vector(array,Thetas[i])
    H[:,iter] = htmp + np.sqrt(0.5/snr)*(np.random.randn(N)+np.random.randn(N)*1j)
CovMat = H@H.conj().transpose()

# MUSIC algorithm
DoAsMUSIC, psindB = music(CovMat,L,N,array,Angles)


plt.subplot(223)
plt.plot(Angles,psindB)
plt.plot(Angles[DoAsMUSIC],psindB[DoAsMUSIC],'x')
plt.title('MUSIC')
plt.legend(['pseudo spectrum','Estimated DoAs'])

# ESPRIT algorithm
DoAsESPRIT = esprit(CovMat,L,N)
plt.subplot(224)
plt.plot(Thetas,np.zeros(L),'*')
plt.plot(DoAsESPRIT,np.zeros(L),'x')
plt.title('ESPRIT')
plt.legend(['Actual DoAs','Estimated DoAs'])

print('Actual DoAs:',np.sort(Thetas),'\n')
print('MUSIC DoAs:',np.sort(Angles[DoAsMUSIC]),'\n')
print('ESPRIT DoAs:',np.sort(DoAsESPRIT),'\n')

plt.show()


# In[ ]:


################################PATA##################


# In[2]:


import numpy as np


# In[6]:


#setup de sincronização
dis_mics = 0.2127
dt = 1/fs
vsom = 346.3
 
def lag_xcorr(signal1,signal2):
    xcorr = np.correlate(signal1, signal2, "full")
    return int(np.argmax(xcorr) - len(xcorr)/2)

def sincroniza_sinal(lag,signal1,signal2):
    k1 = signal2[abs(lag):]
    k2 = signal1[:len(k1)]
    return k1,k2

def lag_GCCphat(signal1,signal2):
    N=2*len(signal1)
    X1aux=scipy.fft(signal1,N)
    X2aux=scipy.fft(signal2,N)
    r12=np.real(scipy.fft.fftshift(scipy.fft.ifft(X1aux*np.conj(X2aux)/(1e-6+abs(X1aux*X2aux)))))
    return int(np.argmax(r12) - len(r12)/2)

#a principio só com a primeira parte de sinc
samples_sinc_interval = 1000


#######lag de defasagem######
lag = lag_xcorr(signal1[:samples_sinc_interval],signal2[:samples_sinc_interval])

signal_real1 = signal1[samples_sinc_interval:2*samples_sinc_interval]
signal_real2 = signal2[samples_sinc_interval:2*samples_sinc_interval]

signal_real1, signal_real2 = sincroniza_sinal(lag, signal_real2, signal_real1)


# In[207]:


import matplotlib.pyplot as plt
import scipy
import numpy as np

def lag_xcorr(signal1,signal2):
    xcorr = np.correlate(signal1, signal2, "full")
    return int(np.argmax(xcorr) - len(xcorr)/2)

def sincroniza_sinal(lag,signal1,signal2):
    k1 = signal2[abs(lag):]
    k2 = signal1[:len(k1)]
    return k1,k2

def lag_GCCphat(signal1,signal2):
    N=2*len(signal1)
    X1aux=scipy.fft(signal1,N)
    X2aux=scipy.fft(signal2,N)
    r12=np.real(scipy.fft.fftshift(scipy.fft.ifft(X1aux*np.conj(X2aux)/(1e-6+abs(X1aux*X2aux)))))
    return int(np.argmax(r12) - len(r12)/2), r12


# In[208]:


x = np.linspace(0, 2*np.pi, 10000)
sen = np.sin(x)
cos = np.cos(x)
plt.plot(x,sen)
plt.plot(x,cos)


# In[209]:


lag = lag_xcorr(sen,cos)


# In[210]:


sen,cos = sincroniza_sinal(lag,cos,sen)


# In[211]:


plt.plot(x[:len(sen)],sen)
plt.plot(x[:len(sen)],cos)


# In[203]:


x = np.linspace(0, 2*np.pi, 10000)
sen = np.sin(x)
cos = np.cos(x)
plt.plot(x,sen)
plt.plot(x,cos)


# In[204]:


lag, r12 = lag_GCCphat(sen,cos)


# In[205]:


sen,cos = sincroniza_sinal(lag,cos,sen)


# In[206]:


plt.plot(x[:len(sen)],sen)
plt.plot(x[:len(sen)],cos)


# In[ ]:




