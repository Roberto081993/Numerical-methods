#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 15:18:58 2020

@author: roberto
"""

import numpy as np

no    = 1
nv    = 1
noo   = int(no*(no-1)/2)
nvv   = int(nv*(nv-1)/2)
S_noo = int((no*(no+1))/2)
S_nvv = int((nv*(nv+1))/2)
T_noo = int(no*(no-1)/2)
T_nvv = int(nv*(nv-1)/2)
nBas  = no + nv
eps   = []
Int   = np.zeros((nBas,nBas,nBas,nBas))

with open("ERI_H2.dat","r") as f:
    data = f.readlines()
    for line in data:
        line      = line.split()
        l         = [float(a) for a in line]
        p         = int(l[0]) - 1
        i         = int(l[1]) - 1
        c         = int(l[2]) - 1
        d         = int(l[3]) - 1
        Int[p][i][c][d] = l[4]
        
with open("epsH2.dat","r") as f:
    data = f.readlines()
    for line in data:
        line    = line.split()
        l       = [float(a) for a in line]
        i       = int(l[0]) - 1
        eps.append(l[1])        

def I(N):
    M = np.zeros((N,N))
    for i in range(N):
        M[i,i] = -1
    return M

def W(noo,nvv):
    Ip       =   np.eye(nvv)
    Im       =   I(noo)
    Z1       =   np.zeros((nvv,noo))
    Z2       =   np.zeros((noo,nvv))
    Metrique =   np.block([[Ip,Z1],[Z2,Im]])
    return Metrique
    
def M(A,B,Bt,C):
    R  = np.block([[A,B],[Bt,C]])
    R2 = np.asmatrix(R)
    return R2

def Tri2(M):
    """
    Tri les valeurs propres et vecteurs propres en ordre decroissant de la
    gauche vers la droite, (les valeurs propres positives sont a gauche,
    les negatives a droite)
    
    [[ 4.,  0.,  0.,  0.],
     [ 0.,  0.,  0.,  0.],
     [ 0.,  0., -1.,  0.],
     [ 0.,  0.,  0., -3.]]
    """
    eigenValues, eigenVectors = np.linalg.eig(M)
    eigenValues               = np.real(eigenValues)
    eigenVectors              = np.real(eigenVectors)
    eigenValues_triees        = np.sort(eigenValues)[::-1]
    eigenVectors_tries        = eigenVectors[:,eigenValues.argsort()[::-1]]
    omega                     = np.asmatrix(np.diag(eigenValues_triees))
    p                         = eigenVectors_tries
    return omega,p

def Puissance(M,n):
    """
    Calcul la puissance d'une matrice dans la base ou la matrice est diagonale,
    !! On suppose que l'inverse s'ecrit comme la transposee !!
    """
    OMEGA_n   = np.power(np.diag(Tri2(M)[0]),n)
    Z         = Tri2(M)[1]
    M_OMEGA_n = np.zeros((len(Z),len(Z)))
    np.fill_diagonal(M_OMEGA_n,OMEGA_n)
    V1        = np.dot(M_OMEGA_n, np.transpose(Z))
    A_n       = np.dot(Z,V1)
    return A_n

def delta(a,b):
    if a == b:
        return 1
    else:
        return 0
    
# GENERATION DES MATRICES A B C
#------------------------------------------------------------------------------                                                  
A_s        = np.zeros((S_nvv,S_nvv))              
B_s        = np.zeros((S_nvv,S_noo))
B_ts       = np.transpose(B_s)                    
C_s        = np.zeros((S_noo,S_noo))       

eo         = eps[0:no]
ev         = eps[no:no+nv]
e_homo     = eo[-1]
e_lumo     = ev[0]
nu         = (e_homo+e_lumo)*0.5


            
# Matrices A B C pour le SINGULET

# matrice B
ab = 0
for b in range(no,nBas):
    for a in range(b,nBas):
        ij = 0
        for j in range(no):
            for i in range(j,no):
                d1, d2 = delta(a,b), delta(i,j)
                B_s[ab,ij] = (1/np.sqrt((1+d1)*(1+d2)))*(Int[a,b,i,j]+Int[a,b,j,i])
                ij += 1
        ab += 1
      
# matrice A
ab = 0
for b in range(0,nv):
    for a in range(b,nv):
        cd = 0
        for d in range(0,nv):
            for c in range(d,nv):
                d1,d2,d3,d4 = delta(a,c), delta(b,d), delta(a,b),delta(c,d)
                A_s[ab,cd]  = d1*d2*(ev[a]+ev[b]-2*nu)+(1/np.sqrt((1+d3)*(1+d4)))*(Int[a+no,b+no,c+no,d+no]+Int[a+no,b+no,d+no,c+no])
                cd += 1
        ab += 1
       
# matrice C
ij = 0
for j in range(no):
    for i in range(j,no):
        kl = 0
        for l in range(no):
            for k in range(l,no):
                d1, d2, d3 ,d4 = delta(i,k),delta(j,l),delta(i,j),delta(k,l)
                C_s[ij,kl] = -(d1*d2*eo[i]+eo[j]-2*nu)+(1/np.sqrt((1+d3)*(1+d4)))*(Int[i,j,k,l]+Int[i,j,l,k])
                kl += +1
        ij += +1

Ms         = M(A_s,B_s,-B_ts,-C_s)
W_s        = W(S_noo,S_nvv)
Z_s        = Tri2(Ms)[1]    

Chi1_s     = Z_s[:,0:S_nvv]
Chi2_s     = Z_s[:,S_nvv:S_nvv+S_noo] 

# ORTHOGONALISATION DES VECTEURS PROPRES
#------------------------------------------------------------------------------
N1_s         = np.dot(W_s,Chi1_s)
N1_s         = np.dot(np.transpose(Chi1_s),N1_s)

N2_s         = np.dot(W_s,Chi2_s)
N2_s         = np.dot(np.transpose(Chi2_s),N2_s)

X1_s         = Puissance(N1_s,-1/2)
X2_s         = Puissance(-N2_s,-1/2)

Chi1_X1_s    = np.dot(Chi1_s,X1_s)
n1_s         = np.dot(W_s,Chi1_X1_s) 
N1_s         = np.dot(np.transpose(Chi1_X1_s),n1_s)
ZX1_s        = Chi1_X1_s[0:S_nvv,0:S_nvv]
ZY1_s        = Chi1_X1_s[S_nvv:S_nvv+S_noo,0:S_nvv]

Chi2_X2_s    = np.dot(Chi2_s,X2_s)
n2_s         = np.dot(W_s,Chi2_X2_s) 
N2_s         = np.dot(np.transpose(Chi2_X2_s),n2_s)
ZX2_s        = Chi2_X2_s[0:S_nvv,0:S_noo]
ZY2_s        = Chi2_X2_s[S_nvv:S_nvv+S_noo,0:S_noo]
#------------------------------------------------------------------------------
# CALCUL DE LA SELF-ENERGY
#------------------------------------------------------------------------------        

Chi_1_s    = np.zeros((nBas,no,S_nvv))
Chi_2_s    = np.zeros((nBas,nv,S_noo))
SigC_s     = np.zeros((nBas,nBas))
dSigC_s    = np.zeros((nBas,nBas))
Z_ss       = np.zeros(nBas) 

Chi_1_ds   = np.zeros((nBas,no,S_nvv))
Chi_2_ds   = np.zeros((nBas,nv,S_noo))
SigC_ds    = np.zeros((nBas,nBas))
dSigC_ds   = np.zeros((nBas,nBas))
Z_sd       = np.zeros(nBas)

omega_S1   = np.diag(Tri2(Ms)[0])[0:S_nvv]          
omega_S2   = np.diag(Tri2(Ms)[0])[S_nvv:S_nvv+S_noo]

Ec1        =  sum(omega_S1) - np.trace(A_s)
Ec2        = -sum(omega_S2) - np.trace(C_s)

# Tableaux de Chi_1 et Chi_2       
for m in range(S_nvv):
    for i in range(no):
        for p in range(nBas): 
            cd = 0
            for c in range(no,nBas):
                for d in range(c,nBas):
                    Chi_1_ds[p,i,m] += np.dot(Int[p,i,c,d],ZX1_s[cd,m])
                    cd += 1
                    
            kl = 0
            for k in range(no):
                for l in range(k,no):
                    Chi_1_ds[p,i,m] += np.dot(Int[p,i,k,l],ZY1_s[kl,m])
                    kl += 1

for m in range(S_noo):
    for a in range(nv): 
        for p in range(nBas):
            cd = 0
            for c in range(no,nBas):
                for d in range(c,nBas):
                    Chi_2_ds[p,a,m] += np.dot(Int[p,no+a,c,d],ZX2_s[cd,m])

                    cd += 1
             
            kl = 0       
            for k in range(no):
                for l in range(k,no):
                    Chi_2_ds[p,a,m] += np.dot(Int[p,no+a,k,l],ZY2_s[kl,m])
                    kl += 1        
            
for p in range(nBas):
    for q in range(nBas):
        for m in range(S_nvv):
            for i in range(no):
                tmp_ds         = np.dot(Chi_1_ds[p,i,m],Chi_1_ds[q,i,m])/(eps[p]+eo[i]-omega_S1[m].item())
                SigC_ds[p,q]  += tmp_ds
                dSigC_ds[p,q] -= tmp_ds/(eps[p]+eo[i]-omega_S1[m].item())
                    
        for m in range(S_noo):
            for a in range(nv):
                tmp_ds         = np.dot(Chi_2_ds[p,a,m],Chi_2_ds[q,a,m])/(eps[p]+ev[a]-omega_S2[m].item())
                SigC_ds[p,q]  += tmp_ds
                dSigC_ds[p,q] -= tmp_ds/(eps[p]+ev[a]-omega_S2[m].item())
    
e_gt = [] 
for p in range(nBas):
    Z_sd[p] = 1/(1-dSigC_ds[p,p])                   
    c       = eps[p]+Z_sd[p]*SigC_ds[p,p]
    e_gt.append(c)  

e_gto = e_gt[0:no]
e_gtv = e_gt[no:no+nv]

         