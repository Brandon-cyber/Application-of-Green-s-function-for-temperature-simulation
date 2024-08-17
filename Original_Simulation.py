import numpy as np
from scipy import fftpack
import matplotlib.pyplot as plt

# see eq. 18 in https://journals.aps.org/prb/abstract/10.1103/PhysRevB.90.214306
def fdtr1d_gray(ww,qx,delta,vx,tau,CC):
    rr = 1.0/(vx*tau*qx)*np.arctan((vx*tau*qx)/(1.0+1.j*ww*tau))
    ret = 2.0*(1/delta)/((1.0/delta)**2+qx**2)*(rr)/(1.0-rr)/(CC/tau)
    return ret

def fdtr1d_gray_surf(ww,qx,delta,vx,tau,CC):
    rr = 1.0/(vx*tau*qx)*np.arctan((vx*tau*qx)/(1.0+1.j*ww*tau))
    ret = (rr)/(1.0-rr)/(CC/tau)
    return ret

def fdtr2d_gray(ww,qx,qy,qz,delta,R,vx,tau,CC):
    q = np.sqrt(qx**2+qy**2+qz**2)
    rr = 1.0/(vx*tau*q)*np.arctan((vx*tau*q)/(1.0+1.j*ww*tau))
    ret = R**3*2.0*(1/delta)/((1.0/delta)**2+qz**2)*np.exp(-R**2*(qx**2+qy**2)/(4*np.pi))*(rr)/(1.0-rr)/(CC/tau)
    return ret

# see eq. 13 in https://journals.aps.org/prb/abstract/10.1103/PhysRevB.90.214306
def fdtr1d_gray_ballistic(ww,qx,delta,vx,tau,CC):
    rr = 1.0/(vx*tau*qx)*np.arctan((vx*qx)/(1.j*ww))
    ret = 2.0*(1/delta)/((1.0/delta)**2+qx**2)*(rr)/(CC/tau)
    return ret

def fdtr1d_gray_ballistic_surf(ww,qx,delta,vx,tau,CC):
    rr = 1.0/(vx*tau*qx)*np.arctan((vx*qx)/(1.j*ww))
    ret = (rr)/(CC/tau)
    return ret

# Check convergence with respect to spatial frequency fourier transform variable
qx = np.linspace(-1e8,1e8,100)
qy = np.linspace(-1e8,1e8,100)
qz = np.linspace(-1e8,1e8,100)

#gray values from germanium from https://www.science.org/doi/10.1126/sciadv.abg4677
mfp = 4./3.*np.sqrt(2)*4000.0e-9
CC = 1.6e6
vx = 6733./(2*np.sqrt(2))
tau = mfp/(2.*vx)
delta = 15.0e-9
R=3.0e-6

#temporal frequency fourier transform variable
wlist = np.logspace(3.0,15.0,80)

#initialize array
deltaT = np.zeros((np.shape(wlist)),dtype=np.complex128)
deltaT_surf = np.zeros((np.shape(wlist)),dtype=np.complex128)
deltaT_ballistic = np.zeros((np.shape(wlist)),dtype=np.complex128)

for i,wi in enumerate(wlist):
    dT = 0.0+0.0j
    dT_b = 0.0+0.0j
    dT_s = 0.0+0.0j
    for ix,qxi in enumerate(qx):
        for iy,qyi in enumerate(qy):
            for iz,qzi in enumerate(qz):
                dT = dT + fdtr2d_gray(wi,qxi,qyi,qzi,delta,R,vx,tau,CC)
    #dT_s = np.sum(fdtr1d_gray_surf(wi,q,delta,vx,tau,CC))
    #dT_b = np.sum(fdtr1d_gray_ballistic_surf(wi,q,delta,vx,tau,CC))

    #deltaT[i] = np.angle(dT)
    print(dT.shape)
    deltaT[i] = dT
    #deltaT_surf[i] = dT_s
    #deltaT_ballistic[i] = dT_b
    print(deltaT[i])

#plot phase
plt.semilogx(wlist/(2*np.pi),np.angle(deltaT)*180/np.pi)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Phase lag (Deg)')
plt.savefig('fdtr2d_gray.pdf')
plt.close()

#plot amplitude
plt.loglog(wlist/(2*np.pi),np.abs(deltaT))
plt.savefig('fdtr2d_gray_amp.pdf')
plt.close()