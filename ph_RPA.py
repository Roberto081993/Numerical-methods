#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 16:04:27 2020

@author: roberto
"""

import numpy as np
from scipy.linalg import fractional_matrix_power
#from Singulet_standalone import T_s
import matplotlib.pyplot as plt

no    = 1
nv    = 1
S_noo = int((no*(no+1))/2)
S_nvv = int((nv*(nv+1))/2)
nBas  = no + nv
Int   = np.zeros((nBas,nBas,nBas,nBas))
eps   = []

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
    
def M(A,B,Bt,D):
    R  =  np.block([[A,B],[Bt,D]])
    R2 =  np.asmatrix(R)
    return R2

def Tri(M):
    """
    Tri les valeurs propres et vecteurs propres en ordre croissant de la
    gauche vers la droite, (les valeurs propres negatives sont a gauche,
    les positives a droite)
    
    [[-3.,  0.,  0.,  0.],
     [ 0., -1.,  0.,  0.],
     [ 0.,  0.,  0.,  0.],
     [ 0.,  0.,  0.,  4.]]
    """
    eigenValues, eigenVectors = np.linalg.eig(M)
    eigenValues               = np.real(eigenValues)
    eigenVectors              = np.real(eigenVectors)
    eigenValues_triees        = np.sort(eigenValues)
    eigenVectors_tries        = eigenVectors[:,eigenValues.argsort()]
    omega                     = np.asmatrix(np.diag(eigenValues_triees))
    p                         = eigenVectors_tries
    return omega,p


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
 
def Singulet(no,nv,ev,eo):
    A    = np.zeros((no*nv,no*nv))
    B    = np.zeros((no*nv,no*nv)) 
    # Matrice A
    ia = 0
    for i in range(no):
        for a in range(nv):
            jb = 0
            for j in range(no):
                for b in range(nv):
                    d1,d2    = delta(i,j), delta(a,b)
                    A[ia,jb] = d1*d2*(ev[a]-eo[i])+2*Int[i,j,a+no,b+no]
                    jb += 1
            ia += 1 
            
    # Matrice B
    ia = 0
    for i in range(no):
        for a in range(no,nBas):
            jb = 0
            for j in range(no):
                for b in range(no,nBas):
                    B[ia,jb] = 2*Int[i,j,a,b]
                    jb += 1
            ia += 1
    return A,B

def Triplet(no,nv,ev,eo):
    A    = np.zeros((no*nv,no*nv))
    B    = np.zeros((no*nv,no*nv)) 
    # Matrice A
    ia = 0
    for i in range(no):
        for a in range(nv):
            jb = 0
            for j in range(no):
                for b in range(nv):
                    d1,d2    = delta(i,j), delta(a,b)
                    A[ia,jb] = d1*d2*(ev[a]-eo[i])
                    jb += 1
            ia += 1 
            
    # Matrice B
    ia = 0
    for i in range(no):
        for a in range(no,nBas):
            jb = 0
            for j in range(no):
                for b in range(no,nBas):
                    B[ia,jb] = 0 
                    jb += 1
            ia += 1
    return A,B

def C(A,B,no,nv):        
    c        = np.dot((A+B),fractional_matrix_power(A-B,0.5))
    c        = np.dot(fractional_matrix_power(A-B,0.5),c)

    Zs       = Tri(c)[1]
    omega2_s = Tri(c)[0]
    omega_s  = fractional_matrix_power(omega2_s,0.5)

    Xs       = (np.dot(fractional_matrix_power(omega_s,-0.5),fractional_matrix_power(A-B,0.5))+np.dot(fractional_matrix_power(omega_s,0.5),fractional_matrix_power(A-B,-0.5)))
    Xs       = 0.5*np.dot(Xs,Zs)
    Ys       = (np.dot(fractional_matrix_power(omega_s,-0.5),fractional_matrix_power(A-B,0.5))-np.dot(fractional_matrix_power(omega_s,0.5),fractional_matrix_power(A-B,-0.5)))
    Ys       = 0.5*np.dot(Ys,Zs)  
    return omega_s,Xs,Ys,c

def SigGW_C(no,nv,nBas,eps,eo,ev,omega_s,w,Nw):
    omegas = [0.0 + 0.05 * i for i in range(Nw)]
    eta    = 0.0
    SigC   = np.zeros((nBas,nBas))
    dSigC  = np.zeros((nBas,nBas))
    Z      = np.zeros(nBas)
    
    for omega in omegas:
        for p in range(nBas):
            for q in range(nBas):
                for m in range(no*nv):
                    for i in range(no):
                        tmp         = w[p,i,m]*w[q,i,m]*(1/(omega+eps[p]-eo[i]+np.diag(omega_s)[m])-1.j*eta)
                        SigC[p,q]  += tmp
                        dSigC[p,q] -= tmp/(omega+eps[p]-eo[i]+np.diag(omega_s)[m]-1.j*eta)
                    for a in range(nv):
                        tmp         = w[p,a+no,m]*w[q,a+no,m]*(1/(omega+eps[p]-ev[a]-np.diag(omega_s)[m])+1.j*eta)
                        SigC[p,q]  += tmp
                        dSigC[p,q] -= tmp/(omega+eps[p]-ev[a]-np.diag(omega_s)[m]+1.j*eta)
    e_qp = [] 
    for p in range(nBas):
        Z[p]    = 1/(1-2*dSigC[p,p])
        c       = eps[p]+Z[p]*2*SigC[p,p]
        e_qp.append(c)
    return np.diag(2*SigC),e_qp,Z             

def w(no,nv,nBas,Xs,Ys):
    w = np.zeros((nBas,nBas,no*nv))
    for m in range(no*nv):
        for p in range(nBas):
            for q in range(nBas):
                    ia   = 0
                    for i in range(no):
                        for a in range(nv):
                            w[p,q,m] += Int[p,i,q,a+no]*(Xs[ia,m]+Ys[ia,m])
                            ia += 1    
    return w

def SigGT_C(noo,nvv,no,nv,nBas,eps,eo,ev,h_2p,Nw):
    omegas = [0.0 + 0.05 * i for i in range(Nw)]
    eta    = 0.0
    m         = W(no*nv,no*nv)
    Z_s        = Tri2(h_2p)[1]    
    
    Chi1_s     = Z_s[:,0:no*nv]
    Chi2_s     = Z_s[:, no*nv : no*nv + no*nv] 
    
    # ORTHOGONALISATION DES VECTEURS PROPRES
    #------------------------------------------------------------------------------
    N1_s         = np.dot(m,Chi1_s)
    N1_s         = np.dot(np.transpose(Chi1_s),N1_s)
    
    N2_s         = np.dot(m,Chi2_s)
    N2_s         = np.dot(np.transpose(Chi2_s),N2_s)
    
    X1_s         = Puissance(N1_s,-1/2)
    X2_s         = Puissance(-N2_s,-1/2)
    
    Chi1_X1_s    = np.dot(Chi1_s,X1_s)
    n1_s         = np.dot(m,Chi1_X1_s) 
    N1_s         = np.dot(np.transpose(Chi1_X1_s),n1_s)
    #ZX1_s        = Chi1_X1_s[0 : no*nv, 0 : no*nv]
    #ZY1_s        = Chi1_X1_s[no*nv : no*nv+no*nv, 0 : no*nv]
    ZX1_s        = Chi1_s[0 : no*nv, 0 : no*nv]
    ZY1_s        = Chi1_s[no*nv : no*nv+no*nv, 0 : no*nv]
    
    Chi2_X2_s    = np.dot(Chi2_s,X2_s)
    n2_s         = np.dot(m,Chi2_X2_s) 
    N2_s         = np.dot(np.transpose(Chi2_X2_s),n2_s)
    #ZX2_s        = Chi2_X2_s[0 : no*nv, 0 : no*nv]
    #ZY2_s        = Chi2_X2_s[no*nv : no*nv+no*nv, 0 : no*nv]
    ZX2_s        = Chi2_s[0 : no*nv, 0 : no*nv]
    ZY2_s        = Chi2_s[no*nv : no*nv+no*nv, 0 : no*nv]
    
    #------------------------------------------------------------------------------
    # CALCUL DE LA SELF-ENERGY
    #------------------------------------------------------------------------------        
    

    Chi_1_s   = np.zeros((nBas,no*nv,no*nv))
    Chi_2_s   = np.zeros((nBas,no*nv,no*nv))
    SigC_s    = np.zeros((nBas,nBas))
    dSigC_s   = np.zeros((nBas,nBas))
    Z         = np.zeros(nBas)

    omega_s1   = np.diag(Tri2(h_2p)[0])[0:no*nv]          
    omega_s2   = np.diag(Tri2(h_2p)[0])[no*nv : no*nv+no*nv]

    """
    # Tableaux de Chi_1 et Chi_2 pour la self-énergie      
    for m in range(no*nv):
        for i in range(no):
            for p in range(nBas): 
                cd = 0
                for c in range(no,nBas):
                    for d in range(c,nBas):
                        Chi_1_s[p,i,m] += np.dot(Int[p,i,c,d],ZX1_s[cd,m])
                        cd += 1
                        
                kl = 0
                for k in range(no):
                    for l in range(k,no):
                        Chi_1_s[p,i,m] += np.dot(Int[p,i,k,l],ZY1_s[kl,m])
                        kl += 1
    
    for m in range(no*nv):
        for a in range(nv): 
            for p in range(nBas):
                cd = 0
                for c in range(no,nBas):
                    for d in range(c,nBas):
                        Chi_2_s[p,a,m] += np.dot(Int[p,no+a,c,d],ZX2_s[cd,m])
                        cd += 1
                 
                kl = 0       
                for k in range(no):
                    for l in range(k,no):
                        Chi_2_s[p,a,m] += np.dot(Int[p,no+a,k,l],ZY2_s[kl,m])
                        kl += 1        
                
    for p in range(nBas):
        for q in range(nBas):
            for m in range(no*nv):
                for i in range(no):
                    tmp         = np.dot(Chi_1_s[p,i,m],Chi_1_s[q,i,m])/(eps[p]+eo[i]-omega_s1[m].item())
                    SigC_s[p,q]  += tmp
                    dSigC_s[p,q] -= tmp/(eps[p]+eo[i]-omega_s1[m].item())
                        
            for m in range(no*nv):
                for a in range(nv):
                    tmp           = np.dot(Chi_2_s[p,a,m],Chi_2_s[q,a,m])/(eps[p]+ev[a]-omega_s2[m].item())
                    SigC_s[p,q]  += tmp
                    dSigC_s[p,q] -= tmp/(eps[p]+ev[a]-omega_s2[m].item())
                    
    # Tableau de T_ijkl 
    
    T = np.zeros((nv*no,no*nv))

    ij = 0
    for i in range(nv):
        for j in range(no):
            kl = 0
            for k in range(no):
                for l in range(nv):                
                    for m in range(nvv):            
                        tmp         = np.dot(Chi_1_s[i,j,m],Chi_1_s[k,l,m])/(eo[i]-omega_s1[m].item())
                        T[ij,kl]  += tmp

                        tmp        = np.dot(Chi_2_s[k,l,m],Chi_2_s[i,j,m])/(ev[a]-omega_s2[m].item())
                        T[ij,kl]  -= tmp
     """   
    SigC       = np.diag(SigC_s)
    dSigC      = np.diag(dSigC_s)
    e_gt       = [] 
    for p in range(nBas):
        Z[p] = 1/(1-dSigC_s[p,p])                   
        c      = eps[p]+Z[p]*SigC_s[p,p]
        e_gt.append(c)  
        
    e_gto      = e_gt[0:no]
    e_gtv      = e_gt[no:no+nv]
    return e_gto,e_gtv,Z,SigC,dSigC

      
def Coulomb(no,nv,w,omega_s,Nw,t):
    W_ijab = np.zeros((no*nv,no*nv))
    W_ibaj = np.zeros((no*nv,no*nv))
    omegas = [0.0 + 0.05 * i for i in range(Nw)]
    eta    = 0.0
    for omega in omegas:
        S  = np.zeros((no,no,nv,nv))
        ia = 0
        for i in range(no):
            for a in range(nv):
                jb = 0
                for j in range(no):
                    for b in range(nv):  
                        for m in range(no*nv):
                            S[i,j,a,b] += w[i,j,m]*w[a+no,b+no,m]*(1/(omega-np.diag(omega_s)[m]+1.j*eta)-1/(omega+np.diag(omega_s)[m]-1.j*eta))
                        W_ijab[ia,jb]  += t*Int[i,a+no,j,b+no] + 2*S[i,j,a,b]
                        jb += 1
                ia += 1
    
    for omega in omegas:
        S  = np.zeros((no,nv,nv,no))
        ib = 0
        for i in range(no):
            for b in range(nv):
                aj = 0
                for a in range(nv):
                    for j in range(no):
                        for m in range(no*nv):
                            S[i,b,a,j] += w[i,b+no,m]*w[a+no,j,m]*(1/(omega-np.diag(omega_s)[m]+1.j*eta)-1/(omega+np.diag(omega_s)[m]-1.j*eta))
                        W_ibaj[ib,aj]  += t*Int[i,a+no,b+no,j] + 2*S[i,b,a,j]
                        aj += 1
                ib += 1 

    return W_ijab,W_ibaj

def BSE(W_ijab,W_ibaj,ev,eo,no,nv,nBas,t,nc):
    # Matrice A
    A    = np.zeros((no*nv,no*nv))
    B    = np.zeros((no*nv,no*nv))
    ia = 0
    for i in range(no):
        for a in range(nv):
            jb = 0
            for j in range(no):
                for b in range(nv):
                    d1,d2    = delta(i,j), delta(a,b)
                    A[ia,jb] = d1*d2*(ev[a]-eo[i])+2*t*Int[i,j,a+no,b+no]-W_ijab[ia,jb] 
                    jb += 1
            ia += 1 
            
    # Matrice B
    ia = 0
    for i in range(no):
        for a in range(no,nBas):
            jb = 0
            for j in range(no):
                for b in range(no,nBas):
                    B[ia,jb] = nc*(2*t*Int[i,j,a,b]-W_ibaj[ia,jb])
                    jb += 1
            ia += 1
    return A,B

def BSE_T(T_ijab,ev,eo,no,nv,nBas,t,nc):
    # Matrice A
    A    = np.zeros((no*nv,no*nv))
    B    = np.zeros((no*nv,no*nv))
    ia = 0
    for i in range(no):
        for a in range(nv):
            jb = 0
            for j in range(no):
                for b in range(nv):
                    d1,d2    = delta(i,j), delta(a,b)
                    A[ia,jb] = d1*d2*(ev[a]-eo[i])+2*t*Int[i,j,a+no,b+no]- T_ijab[ia,jb] 
                    jb += 1
            ia += 1 
            
    # Matrice B
    ia = 0
    for i in range(no):
        for a in range(no,nBas):
            jb = 0
            for j in range(no):
                for b in range(no,nBas):
                    B[ia,jb] = nc*(2*t*Int[i,j,a,b])#-T_ibaj[ia,jb])
                    jb += 1
            ia += 1
    return A,B

SigC   = np.zeros((nBas,nBas))
dSigC  = np.zeros((nBas,nBas)) 
eo     = eps[0:no]
ev     = eps[no:no+nv]
e_homo = eo[-1]
e_lumo = ev[0]
nu     = (e_homo+e_lumo)*0.5


A         = Singulet(no,nv,ev,eo)[0]
B         = np.zeros((no*nv,no*nv))
Bt        = np.transpose(B)
At        = np.transpose(A)
#B        = Singulet(no,nv,ev,eo)[1]

"""
A_T = BSE_T(T_s,ev,eo,no,nv,nBas,t = 1,nc = 0)[0]
B_T = BSE_T(T_s,ev,eo,no,nv,nBas,t = 1,nc = 0)[1]

H_2p      = M(A_T, B_T, B_T, A_T)

T = SigGT_C(no,nv,nBas,eps,eo,ev,H_2p,Nw=0)[5]

omega_T_s = C(A_T,B_T,no,nv)[0]
"""
# REDUCTION DE LA TAILLE DU SYSTEME
#------------------------------------------------------------------------------
As      = Singulet(no,nv,ev,eo)[0]
Bs      = Singulet(no,nv,ev,eo)[1]
omega_s = C(As,Bs,no,nv)[0]
Xs      = C(As,Bs,no,nv)[1]
Ys      = C(As,Bs,no,nv)[2]

At      = Triplet(no,nv,ev,eo)[0]
Bt      = Triplet(no,nv,ev,eo)[1]
omega_t = C(At,Bt,no,nv)[0]
Xt      = C(At,Bt,no,nv)[1]
Yt      = C(At,Bt,no,nv)[2]

# CALCUL DU POTENTIEL DE COULOMB
#------------------------------------------------------------------------------

s = 0
t = 0
for i in range(no):
    for a in range(nv):
        s += ev[a]-eo[i]+2*Int[i,i,a+no,a+no]
        t += ev[a]-eo[i]

Ec_s = -0.5*sum(np.diag(omega_s)) + 0.5*s
#Ec_t = -0.5*sum(np.diag(omega_t)) + 0.5*t

# CALCUL DE SIGC_W  
w_s      = w(no,nv,nBas,Xs,Ys)
     
SigC_gw_s   = SigGW_C(no,nv,nBas,eps,eo,ev,omega_s,w_s,Nw=1)[0]             

e_gw_s      = SigGW_C(no,nv,nBas,eps,eo,ev,omega_s,w_s,Nw=1)[1]
e_gwo_s     = e_gw_s[0:no]
e_gwv_s     = e_gw_s[no:no+nv]



# CALCUL BETHE-SALPETER
#------------------------------------------------------------------------------
W_ijab_s    = Coulomb(no,nv,w_s,omega_s,Nw=1,t=1)[0]
W_ibaj_s    = Coulomb(no,nv,w_s,omega_s,Nw=1,t=1)[1]


A_s_bse     = BSE(W_ijab_s,W_ibaj_s,e_gwv_s,e_gwo_s,no,nv,nBas,t=1,nc=1)[0]
B_s_bse     = BSE(W_ijab_s,W_ibaj_s,e_gwv_s,e_gwo_s,no,nv,nBas,t=1,nc=1)[1]
omega_s_bse = C(A_s_bse,B_s_bse,no,nv)[0]
Xs_bse      = C(A_s_bse,B_s_bse,no,nv)[1]
Ys_bse      = C(A_s_bse,B_s_bse,no,nv)[2]

