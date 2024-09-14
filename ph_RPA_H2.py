#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 16:04:27 2020

@author: roberto
"""

import numpy as np
from scipy.linalg import fractional_matrix_power
import matplotlib.pyplot as plt

no   = 1
nv   = 1
nBas = no + nv

def main(eps,Int):
      
    eo     = eps[0:no]
    ev     = eps[no:no+nv]
    e_homo = eo[-1]
    e_lumo = ev[0]
    nu     = (e_homo+e_lumo)*0.5
    
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
        Z         = Tri(M)[1]
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
     
    def Singulet(no,nv,ev,eo,Int):
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
                        A[ia,jb] = d1*d2*(ev[a]-eo[i])+2*Int[i,b+no,a+no,j]
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
    
    def Triplet(no,nv,ev,eo,Int):
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
        return omega_s,Xs,Ys,omega2_s
    
    def Sig_C(no,nv,nBas,eps,eo,ev,omega_s,w_mn,w):
        omegas = [0.0 + 0.0 * i for i in range(w)]
        eta    = 0.0
        SigC   = np.zeros((nBas,nBas))
        dSigC  = np.zeros((nBas,nBas))
        Z      = np.zeros(nBas)
        
        for omega in omegas:
            for p in range(nBas):
                for q in range(nBas):
                    for m in range(no*nv):
                        for i in range(no):
                            tmp         = 2*(w_mn[p,i,m]*w_mn[q,i,m])/(omega+eps[p]-eo[i]+np.diag(omega_s)[m])#-1.j*eta
                            SigC[p,q]  += tmp
                            dSigC[p,q] -= tmp/(omega+eps[p]-eo[i]+np.diag(omega_s)[m])#-1.j*eta
                        for a in range(nv):
                            tmp         = 2*(w_mn[p,a+no,m]*w_mn[q,a+no,m])/(omega+eps[p]-ev[a]-np.diag(omega_s)[m])#+1.j*eta
                            SigC[p,q]  += tmp
                            dSigC[p,q] -= tmp/(omega+eps[p]-ev[a]-np.diag(omega_s)[m])#+1.j*eta

        e_qp = [] 
        for p in range(nBas):
            Z[p]    = 1/(1-dSigC[p,p])
            c       = eps[p]+Z[p]*SigC[p,p]
            e_qp.append(c)
            
        sigc = np.diag(SigC)
        dsigc = np.diag(dSigC)
        return e_qp,sigc,dsigc,Z          
    
    def wmn(no,nv,nBas,Xs,Ys,Int):
        w_mn = np.zeros((nBas,nBas,no*nv))
        for m in range(no*nv):
            for p in range(nBas):
                for q in range(nBas):
                        ia   = 0
                        for i in range(no):
                            for a in range(nv):
                                w_mn[p,q,m] += Int[p,i,q,a+no]*(Xs[ia,m]+Ys[ia,m])
                                ia += 1    
        return w_mn
          
    def Coulomb(no,nv,w_mn,omega_s,Int,w,t):
        W_ijab = np.zeros((no*nv,no*nv))
        W_ibaj = np.zeros((no*nv,no*nv))
        omegas = [0.0 + 0.05 * i for i in range(w)]
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
                                S[i,j,a,b] += w_mn[i,j,m]*w_mn[a+no,b+no,m]*(1/(omega-np.diag(omega_s)[m]+1.j*eta)-1/(omega+np.diag(omega_s)[m]-1.j*eta))
                            W_ijab[ia,jb]  += t*Int[i,b+no,j,a+no] + 2*S[i,j,a,b]
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
                                S[i,b,a,j] += w_mn[i,b+no,m]*w_mn[a+no,j,m]*(1/(omega-np.diag(omega_s)[m]+1.j*eta)-1/(omega+np.diag(omega_s)[m]-1.j*eta))
                            W_ibaj[ib,aj]  += t*Int[i,j,b+no,a+no] + 2*S[i,b,a,j]
                            aj += 1
                    ib += 1 
    
        return W_ijab,W_ibaj
    
    def BSE(Xi_ijab,Xi_ibaj,ev,eo,no,nv,nBas,t):
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
                        A[ia,jb] = d1*d2*(ev[a]-eo[i])+2*t*Int[i,b+no,a+no,j]-Xi_ijab[ia,jb] 
                        jb += 1
                ia += 1 
                
        # Matrice B
        ia = 0
        for i in range(no):
            for a in range(no,nBas):
                jb = 0
                for j in range(no):
                    for b in range(no,nBas):
                        B[ia,jb] = 2*t*Int[i,j,a,b]-Xi_ibaj[ia,jb]
                        jb += 1
                ia += 1
        return A,B
    
    
    
    # REDUCTION DE LA TAILLE DU SYSTEME
    #------------------------------------------------------------------------------
    As      = Singulet(no,nv,ev,eo,Int)[0]
    Bs      = Singulet(no,nv,ev,eo,Int)[1]
    omega_s = C(As,Bs,no,nv)[0]
    Xs      = C(As,Bs,no,nv)[1]
    Ys      = C(As,Bs,no,nv)[2]
    
    
    At      = Triplet(no,nv,ev,eo,Int)[0]
    Bt      = Triplet(no,nv,ev,eo,Int)[1]
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
    Ec_t = -0.5*sum(np.diag(omega_t)) + 0.5*t
    
    Ec_s = -Ec_s*27.2
    
    
    # CALCUL DE SIGC_W  
    w_mn_s      = wmn(no,nv,nBas,Xs,Ys,Int)
    w_mn_t      = wmn(no,nv,nBas,Xt,Yt,Int) 
         
    SigC_gw_s   = Sig_C(no,nv,nBas,eps,eo,ev,omega_s,w_mn_s,w=1)[0]             
    SigC_gw_t   = Sig_C(no,nv,nBas,eps,eo,ev,omega_t,w_mn_t,w=1)[0]
    
    
    e_gw_s      = Sig_C(no,nv,nBas,eps,eo,ev,omega_s,w_mn_s,w=1)[0]
    e_gwo_s     = e_gw_s[0:no]
    e_gwv_s     = e_gw_s[no:no+nv]

    e_gw_t      = Sig_C(no,nv,nBas,eps,eo,ev,omega_t,w_mn_s,w=1)[0]
    e_gwo_t     = e_gw_t[0:no]
    e_gwv_t     = e_gw_t[no:no+nv]
    
    Z_s         = Sig_C(no,nv,nBas,eps,eo,ev,omega_s,w_mn_s,w=1)[3]



     
    # CALCUL BETHE-SALPETER
    #--------------------------------------------------------------------------
    W_ijab_s       = Coulomb(no,nv,w_mn_s,omega_s,Int,w=1,t=1)[0]
    W_ibaj_s       = Coulomb(no,nv,w_mn_s,omega_s,Int,w=1,t=1)[1]
 
    W_ijab_t       = Coulomb(no,nv,w_mn_s,omega_t,Int,w=1,t=1)[0]
    W_ibaj_t       = Coulomb(no,nv,w_mn_s,omega_t,Int,w=1,t=1)[1]

    # SINGULET
    #--------------------------------------------------------------------------
    A_s_bse_gw     = BSE(W_ijab_s,W_ibaj_s,e_gwv_s,e_gwo_s,no,nv,nBas,t=1)[0]
    B_s_bse_gw     = BSE(W_ijab_s,W_ibaj_s,e_gwv_s,e_gwo_s,no,nv,nBas,t=1)[1]
    omega_s_bse_gw = C(A_s_bse_gw,B_s_bse_gw,no,nv)[0]

    #--------------------------------------------------------------------------




    # TRIPLET
    #--------------------------------------------------------------------------
    A_t_bse_gw     = BSE(W_ijab_s,W_ibaj_s,e_gwv_t,e_gwo_t,no,nv,nBas,t=1)[0]
    B_t_bse_gw     = BSE(W_ijab_s,W_ibaj_s,e_gwv_t,e_gwo_t,no,nv,nBas,t=1)[1]
    omega_t_bse_gw = C(A_t_bse_gw,B_t_bse_gw,no,nv)[0]

    #--------------------------------------------------------------------------


    return e_gwo_s,e_gwv_s,e_gwo_t,e_gwv_t,Ec_s,Z_s,omega_s_bse_gw,omega_t_bse_gw,C(A_s_bse_gw,B_s_bse_gw,no,nv)
    