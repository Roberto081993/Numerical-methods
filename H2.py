#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 10:09:38 2020

@author: roberto
"""

import numpy as np
from numpy import linalg as LA
from scipy import special
import matplotlib.pyplot as plt

alpha       = (0.168856,0.623913,3.42525) 
N1,N2,N3    = 0.444635,0.535328,0.154329 
r           = (0,0,1.4)
Ra,Rb       = (0,0,0),(0,0,1.4)
Zc          = 1


def g(alpha,Ra):
    c = np.power((2*alpha/np.pi),0.75)
    f = c*np.exp(-alpha*np.power(LA.norm(Ra),2))
    return f,alpha,Ra

def Phi(N1,N2,N3,g1,g2,g3):
    phi1 = (N1,g1[1],g1[2],N1*g1[0])
    phi2 = (N2,g2[1],g2[2],N2*g2[0])
    phi3 = (N3,g3[1],g3[2],N3*g3[0])
    return phi1,phi2,phi3
    
def Psi(Phi1,Phi2,S_12):
    phi1_i = 0
    phi2_i = 0
    a_11,a_12,a_13 = Phi1[0][1],Phi1[1][1],Phi1[2][1]
    a_21,a_22,a_23 = Phi2[0][1],Phi2[1][1],Phi2[2][1]
    N_11,N_12,N_13 = Phi1[0][0],Phi1[1][0],Phi1[2][0]
    N_21,N_22,N_23 = Phi2[0][0],Phi2[1][0],Phi2[2][0]
    Ra             = Phi1[0][2]
    Rb             = Phi2[0][2]
    a1             = [a_11,a_12,a_13]
    a2             = [a_21,a_22,a_23]
    N1             = [N_11,N_12,N_13]
    N2             = [N_21,N_22,N_23]
    cp             = np.power(0.5*(1+S_12),-0.5)
    cm             = np.power(0.5*(1-S_12),-0.5)
    for i in range(len(phi1)):
        phi1_i += Phi1[i][3]
        phi2_i += Phi2[i][3]
    Psi_p  = cp*(phi1_i+phi2_i)
    Psi_m  = cm*(phi1_i-phi2_i)
    return Psi_p,Psi_m,Phi1,Phi2

def S(phi1,phi2):
    s = 0
    for i in range(len(phi1)):
        for j in range(len(phi2)):
            a   = phi1[i][1]
            b   = phi2[j][1]
            Ra  = phi1[i][2]
            Rb  = phi2[j][2]
            Rab = LA.norm(np.subtract(Ra,Rb))
            Ni  = phi1[i][0]
            Nj  = phi2[j][0]
            ci  = np.power((2*phi1[i][1]/np.pi),0.75)
            cj  = np.power((2*phi2[j][1]/np.pi),0.75)
            s  += ci*cj*Ni*Nj*np.power(np.pi/(a+b),1.5)*np.exp(-(a*b/(a+b))*np.power(Rab,2))
    return s

def T(phi1,phi2):
    s    = 0
    for i in range(len(phi1)):
        for j in range(len(phi2)):
            a   = phi1[i][1]
            b   = phi2[j][1]
            Ni  = phi1[i][0]
            Nj  = phi2[j][0]
            ci  = np.power((2*a/np.pi),0.75)
            cj  = np.power((2*b/np.pi),0.75)
            Ra  = phi1[i][2]
            Rb  = phi2[j][2]
            Rab = LA.norm(np.subtract(Ra,Rb))
            s  += ci*cj*Ni*Nj*(a*b/(a+b))*(3-2*(a*b/(a+b))*np.power(Rab,2))*np.power(np.pi/(a+b),1.5)*np.exp(-(a*b/(a+b))*np.power(Rab,2))
    return s
    
def F(t):
    f = 0.5*np.power((np.pi/t),0.5)*special.erf(np.power(t,0.5))
    if t == 0.0:
        return 1.0
    else:
        return f

def V_noyaux(phi1,phi2,f,Zc):
    v = 0
    v1 = 0
    for i in range(len(phi1)):
        for j in range(len(phi2)):
            Ra  = phi1[i][2]
            Rb  = phi2[j][2]
            Rab = LA.norm(np.subtract(Ra,Rb))
            a   = phi1[i][1]
            b   = phi2[j][1]
            aRa = tuple([a*i for i in Ra])
            bRb = tuple([b*j for j in Rb])
            Rp  = np.add(aRa,bRb)/(a+b)
            Rpa = LA.norm(np.subtract(Rp,Ra))
            Rpb = LA.norm(np.subtract(Rp,Rb))
            Raa = LA.norm(np.subtract(Ra,Ra))
            Ni  = phi1[i][0]
            Nj  = phi2[j][0]
            ci  = np.power((2*a/np.pi),0.75)
            cj  = np.power((2*b/np.pi),0.75)
            if phi1 == phi2:
                Rpa = LA.norm(Ra)
            v  += -Ni*Nj*ci*cj*(2*np.pi/(a+b))*Zc*np.exp(-(a*b/(a+b))*np.power(Rab,2))*f((a+b)*np.power(Rpa,2))
            v1 += -Ni*Nj*ci*cj*(2*np.pi/(a+b))*Zc*np.exp(-(a*b/(a+b))*np.power(Rab,2))*f((a+b)*np.power(LA.norm(Ra),2))
    return v

def S_ABCD(phi1,phi2,phi3,phi4,f):
    s = 0
    for i in range(len(phi1)):
        for j in range(len(phi2)):
            for k in range(len(phi3)):
                for l in range(len(phi4)):
                    Ra,Rb   = phi1[i][2],phi2[j][2]
                    Rc,Rd   = phi3[k][2],phi4[l][2]
                    Rab,Rcd = LA.norm(np.subtract(Ra,Rb)),LA.norm(np.subtract(Rc,Rd))
                    a,b,c,d = phi1[i][1],phi2[j][1],phi3[k][1],phi4[l][1]
                    aRa,bRb = tuple([a*i for i in Ra]),tuple([b*j for j in Rb])
                    cRc,dRd = tuple([c*k for k in Rc]),tuple([d*l for l in Rd])   
                    Rp,Rq   = np.add(aRa,bRb)/(a+b),np.add(cRc,dRd)/(c+d)
                    Rpq     = LA.norm(np.subtract(Rp,Rq))
                    Ni,Nj   = phi1[i][0],phi2[j][0]
                    Nk,Nl   = phi3[k][0],phi4[l][0] 
                    ci,cj   = np.power((2*a/np.pi),0.75),np.power((2*b/np.pi),0.75)
                    ck,cl   = np.power((2*c/np.pi),0.75),np.power((2*d/np.pi),0.75)
                    C       = Ni*Nj*Nk*Nl*ci*cj*ck*cl
                    s      += C*(2*np.power(np.pi,2.5)/((a+b)*(c+d)*np.power(a+b+c+d,0.5))) * \
                    np.exp(-(a*b/(a+b))*np.power(Rab,2)-(c*d/(c+d))*np.power(Rcd,2))*\
                    f((((a+b)*(c+d))/(a+b+c+d))*np.power(Rpq,2))
    return s

def Mat(a,b,c,d):
    M = np.matrix([[a,b],[c,d]])
    return M

E = []
r = []
R = 6

    
g1     = g(alpha[0],Ra)
g2     = g(alpha[1],Ra)
g3     = g(alpha[2],Ra)
    
g4     = g(alpha[0],Rb)
g5     = g(alpha[1],Rb)
g6     = g(alpha[2],Rb)
    
phi1   = Phi(N1,N2,N3,g1,g2,g3)
phi2   = Phi(N1,N2,N3,g4,g5,g6)
    
T_11   = T(phi1,phi1)
T_12   = T(phi1,phi2)
T_22   = T(phi2,phi2)
    
S_11   = S(phi1,phi1)
S_12   = S(phi1,phi2)
S_22   = S(phi2,phi2)
    
V_11   = V_noyaux(phi1,phi1,F,Zc)
V_12   = V_noyaux(phi1,phi2,F,Zc)
V_22   = V_noyaux(phi2,phi2,F,Zc)
    
T_     = Mat(T_11,T_12,T_12,T_22)
V_1    = Mat(V_11,V_12,V_12,V_22)
V_2    = Mat(V_22,V_12,V_12,V_11)
S_     = Mat(S_11,S_12,S_12,S_22) 
H      = T_ + V_1 + V_2
    
H_11   = T_11+V_11+V_22
H_12   = T_12+2*V_12
H_22   = T_22+V_22+V_11

# Energie de H2+
#================================    
e_n1   = (H[0,0]+H[0,1])/(1+S_12)
e_n2   = (H[0,0]-H[0,1])/(1-S_12) 
#================================

S_AAAA = S_ABCD(phi1,phi1,phi1,phi1,F)
S_AAAB = S_ABCD(phi1,phi1,phi1,phi2,F)
S_AABB = S_ABCD(phi1,phi1,phi2,phi2,F)
S_ABAB = S_ABCD(phi1,phi2,phi1,phi2,F)
S_BBAB = S_ABCD(phi2,phi1,phi2,phi2,F)
S_BBBB = S_ABCD(phi2,phi2,phi2,phi2,F)
S_ABBA = S_ABCD(phi1,phi2,phi2,phi1,F)
    
Psi1   = Psi(phi1,phi2,S_12)[0]
Psi2   = Psi(phi1,phi2,S_12)[1]
    
cp = np.power(1/(2*(1+S_12)),0.5)
cm = np.power(1/(2*(1-S_12)),0.5)
    
F_11   = H[0,0]+(1/(1+S_12))*(0.5*S_AAAA+S_AABB+S_AAAB-0.5*S_ABAB) 
F_12   = H[0,1]+(1/(1+S_12))*(-0.5*S_AABB+S_AAAB+1.5*S_ABAB)

# Energies orbitalaire de H2
#============================    
e1     = (F_11+F_12)/(1+S_12)
e2     = (F_11-F_12)/(1-S_12)  
#============================
    
h_11 = (H_11+2*H_12+H_22)*cp**2
h_22 = (H_11-2*H_12+H_22)*cm**2
    
J_11 = (S_AAAA+S_BBBB+2*S_AABB+4*S_ABAB+4*S_AAAB+4*S_BBAB)*cp**4
J_22 = (S_AAAA+S_BBBB+2*S_AABB+4*S_ABAB-4*S_AAAB-4*S_BBAB)*cm**4
J_12 = (S_AAAA+S_BBBB+2*S_AABB-4*S_ABAB)*(cp**2)*(cm**2)
J1122 = (S_AAAA+S_BBBB+2*S_AABB-2*S_ABAB-2*S_ABBA)*(cp**2)*(cm**2)
    
K_11 = J_11
K_22 = J_22
K_12 = (S_AAAA+S_BBBB-2*S_AABB)*(cp**2)*(cm**2)
    
eps1 = h_11+J_11 
eps2 = h_22+2*J_12-K_12 
    
Rab = LA.norm(np.subtract(Ra,Rb))
# Energie électronique de H2
#================================
E_0 = 2*h_11+J_11
E_2 = 2*h_22+J_22
H2 = Mat(E_0,K_12,K_12,E_2)
eigenValues, eigenVectors = np.linalg.eig(H2)
#================================
E_tot = E_0+1/1.4
E_diss = E_tot-2*(T_11+V_11)
E_H2p = h_11
EI_exact = eigenValues[0]-E_H2p
E.append(E_diss)
r.append(R)

# Energie orbitalaire de H2-
#================================
e2 = h_22+2*J_12-K_12
#================================
EI = -(h_11+J_11)
EA = -h_22-2*J_12+K_12