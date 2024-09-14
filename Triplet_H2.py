47
*#!/usr/bin/env python3
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
T_noo = int(no*(no-1)/2)
T_nvv = int(nv*(nv-1)/2)
nBas  = no + nv


def main(eps,Int):

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
    A_t        = np.zeros((T_nvv,T_nvv))              
    B_t        = np.zeros((T_nvv,T_noo))
    B_tt       = np.transpose(B_t)                    
    C_t        = np.zeros((T_noo,T_noo))       
    
    eo         = eps[0:no]
    ev         = eps[no:no+nv]
    e_homo     = eo[-1]
    e_lumo     = ev[0]
    nu         = (e_homo+e_lumo)*0.5
    

    # Matrices A B C pour le SINGULET
    
    # matrice B
    ab = 0
    for b in range(no,nBas-1):
        for a in range(b+1,nBas):
            ij = 0
            for j in range(no-1):
                for i in range(j+1,no):
                    B_t[ab,ij] = (Int[a,b,i,j]-Int[a,b,j,i])
                    ij += 1
            ab += 1
          
    # matrice A
    ab = 0
    for b in range(0,nv-1):
        for a in range(b+1,nv):
            cd = 0
            for d in range(0,nv-1):
                for c in range(d+1,nv):
                    d1,d2 = delta(a,c), delta(b,d),
                    A_t[ab,cd]  = d1*d2*(ev[a]+ev[b]-2*nu)+(Int[a+no,b+no,c+no,d+no]-Int[a+no,b+no,d+no,c+no])
                    cd += 1
            ab += 1
           
    # matrice C
    ij = 0
    for j in range(no-1):
        for i in range(j+1,no):
            kl = 0
            for l in range(no-1):
                for k in range(l+1,no):
                    d1, d2 = delta(i,k),delta(j,l)
                    C_t[ij,kl] = -(d1*d2*eo[i]+eo[j]-2*nu)+(Int[i,j,k,l]-Int[i,j,l,k])
                    kl += +1
            ij += +1
    
    Mt         = M(A_t,B_t,-B_tt,-C_t)
    W_t        = W(T_noo,T_nvv)
    Z_t        = Tri2(Mt)[1]    
    
    Chi1_t     = Z_t[:,0:T_nvv]
    Chi2_t     = Z_t[:,T_nvv:T_nvv+T_noo] 
    
    # ORTHOGONALISATION DES VECTEURS PROPRES
    #------------------------------------------------------------------------------
    N1_t         = np.dot(W_t,Chi1_t)
    N1_t         = np.dot(np.transpose(Chi1_t),N1_t)
    
    N2_t         = np.dot(W_t,Chi2_t)
    N2_t         = np.dot(np.transpose(Chi2_t),N2_t)
    
    X1_t         = Puissance(N1_t,-1/2)
    X2_t         = Puissance(-N2_t,-1/2)
    
    Chi1_X1_t    = np.dot(Chi1_t,X1_t)
    n1_t         = np.dot(W_t,Chi1_X1_t) 
    N1_t         = np.dot(np.transpose(Chi1_X1_t),n1_t)
    ZX1_t        = Chi1_X1_t[0:T_nvv,0:T_nvv]
    ZY1_t        = Chi1_X1_t[T_nvv:T_nvv+T_noo,0:T_nvv]
    
    Chi2_X2_t    = np.dot(Chi2_t,X2_t)
    n2_t         = np.dot(W_t,Chi2_X2_t) 
    N2_t         = np.dot(np.transpose(Chi2_X2_t),n2_t)
    ZX2_t        = Chi2_X2_t[0:T_nvv,0:T_noo]
    ZY2_t        = Chi2_X2_t[T_nvv:T_nvv+T_noo,0:T_noo]
    #------------------------------------------------------------------------------
    # CALCUL DE LA SELF-ENERGY
    #------------------------------------------------------------------------------        
    

    Chi_1_t   = np.zeros((nBas,no,T_nvv))
    Chi_2_t   = np.zeros((nBas,nv,T_noo))
    SigC_t    = np.zeros((nBas,nBas))
    dSigC_t   = np.zeros((nBas,nBas))
    Z_t       = np.zeros(nBas)
    
    omega_T1   = np.diag(Tri2(Mt)[0])[0:T_nvv]          
    omega_T2   = np.diag(Tri2(Mt)[0])[T_nvv:T_nvv+T_noo]
    
    Ec1        =  sum(omega_T1) - np.trace(A_t)
    Ec2        = -sum(omega_T2) - np.trace(C_t)
    
    sigc   = []
    dsigc  = []
    
    # Tableaux de Chi_1 et Chi_2       
    for m in range(T_nvv):
        for i in range(no):
            for p in range(nBas): 
                cd = 0
                for c in range(no,nBas-1):
                    for d in range(c+1,nBas):
                        Chi_1_t[p,i,m] += np.dot(Int[p,i,c,d],ZX1_t[cd,m])
                        cd += 1
                        
                kl = 0
                for k in range(no-1):
                    for l in range(k+1,no):
                        Chi_1_t[p,i,m] += np.dot(Int[p,i,k,l],ZY1_t[kl,m])
                        kl += 1
    
    for m in range(T_noo):
        for a in range(nv): 
            for p in range(nBas):
                cd = 0
                for c in range(no,nBas-1):
                    for d in range(c+1,nBas):
                        Chi_2_t[p,a,m] += np.dot(Int[p,no+a,c,d],ZX2_t[cd,m])
                        cd += 1
                 
                kl = 0       
                for k in range(no-1):
                    for l in range(k+1,no):
                        Chi_2_t[p,a,m] += np.dot(Int[p,no+a,k,l],ZY2_t[kl,m])
                        kl += 1        
                
    for p in range(nBas):
        for q in range(nBas):
            for m in range(T_nvv):
                for i in range(no):
                    tmp_t         = np.dot(Chi_1_t[p,i,m],Chi_1_t[q,i,m])/(eps[p]+eo[i]-omega_T1[m].item())
                    SigC_t[p,q]  += tmp_t
                    dSigC_t[p,q] -= tmp_t/(eps[p]+eo[i]-omega_T1[m].item())
                        
            for m in range(T_noo):
                for a in range(nv):
                    tmp_t         = np.dot(Chi_2_t[p,a,m],Chi_2_t[q,a,m])/(eps[p]+ev[a]-omega_T2[m].item())
                    SigC_t[p,q]  += tmp_t
                    dSigC_t[p,q] -= tmp_t/(eps[p]+ev[a]-omega_T2[m].item())
        
    SigC       = np.diag(SigC_t)
    dSigC      = np.diag(dSigC_t)
    e_gt       = [] 
    for p in range(nBas):
        Z_t[p] = 1/(1-dSigC_t[p,p])                   
        c      = eps[p]+Z_t[p]*SigC_t[p,p]
        e_gt.append(c)  
        
    e_gto      = e_gt[0:no]
    e_gtv      = e_gt[no:no+nv]
    return e_gto,e_gtv,Z_t,SigC,dSigC,e_gt



       