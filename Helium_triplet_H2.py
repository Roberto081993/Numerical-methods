#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 12:11:11 2020

@author: roberto
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 15:04:36 2020

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
    
    
    def Matrice(As,Bs,Bst,Cs,At,Bt,Btt,Ct):
        Z1 = np.zeros((S_nvv,T_nvv))
        Z2 = np.zeros((S_nvv,T_noo))
        Z3 = np.zeros((T_nvv,S_nvv))
        Z4 = np.zeros((T_nvv,S_noo))
        Z5 = np.zeros((S_noo,T_nvv))
        Z6 = np.zeros((S_noo,T_noo))
        Z7 = np.zeros((T_noo,T_nvv))
        Z8 = np.zeros((T_noo,S_noo))
        R  = np.block([[As,Z1,Bs,Z2],[Z3,At,Z4,Bt],[Bst,Z5,Cs,Z6],[Z7,Btt,Z8,Ct]])
        R2 = np.asmatrix(R)
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
        
    
    def Signature(Mat,W):
        lp    = []
        le    = []
        Z     = Tri2(Mat)[1]      # Vecteurs propres
        # Calcul des signatures et des valeurs propres
        for k in range(len(Z)):
            p1             = np.dot(W,Z[:,k])
            SGN            = np.dot(np.transpose(Z[:,k]),p1)
            p2             = np.dot(Mat,Z[:,k])
            e_k            = np.dot(np.transpose(Z[:,k]),p2)
            lp.append(SGN)
            le.append(e_k)
        p3  = np.dot(Mat,Z)
        O   = np.dot(np.transpose(Z),p3)
        WZ  = np.dot(W,Z)
        ZWZ = np.dot(np.transpose(Z),WZ)
        # Tri des valeurs propres positives dans ep, negatives dans en
        en = []
        ep = []
        for i in range(len(le)):
            if le[i] < 0:
                en.append(le[i])
            else:
                ep.append(le[i])
        return O, Z, ZWZ, lp, le, ep, en
    
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
    
    def Puissance_non_tries(M,n):
        """
        Calcul la puissance d'une matrice dans la base ou la matrice est diagonale,
        !! On suppose que l'inverse s'ecrit comme la transposee !!
        """
        eigenValues, eigenVectors = np.linalg.eig(M)
        eigenValues               = np.real(eigenValues)
        eigenValues               = np.power(np.diag(np.asmatrix(np.diag(eigenValues))),n)
        eigenVectors              = np.real(eigenVectors)
        M_OMEGA_n                 = np.zeros((len(eigenVectors),len(eigenVectors)))
        np.fill_diagonal(M_OMEGA_n,eigenValues)
        V1    = np.dot(M_OMEGA_n,np.transpose(eigenVectors))
        A_n   = np.dot(eigenVectors,V1)
        return A_n
    
    def delta(a,b):
        if a == b:
            return 1
        else:
            return 0 
        
    def Delta(a,b,c,d):
        if a == b and b == c and c == d:
            return 1
        else:
            return 0 
    
    
    # GENERATION DES MATRICES A B C
    #------------------------------------------------------------------------------
    
    A_aat      = np.zeros((T_nvv,T_nvv))              # T
    B_aat      = np.zeros((T_nvv,T_noo))              # R
    B_aatt     = np.transpose(B_aat)                  # I
    C_aat      = np.zeros((T_noo,T_noo))              # P  # alpha alpha
                                                      # L
                                                      # E
                                                      # T
     
    A_abt      = np.zeros((T_nvv,T_nvv))              # T
    B_abt      = np.zeros((T_nvv,T_noo))              # R
    B_abtt     = np.transpose(B_abt)                  # I
    C_abt      = np.zeros((T_noo,T_noo))              # P  # alpha beta
                                                      # L
                                                      # E
                                                      # T
                     
    ERI        = np.zeros((nBas,nBas,nBas,nBas))
    sERI       = np.zeros((nBas2,nBas2,nBas2,nBas2))            
    eo         = eps[0:no]
    ev         = eps[no:no+nv]
    e_homo     = eo[-1]
    e_lumo     = ev[0]
    nu         = (e_homo+e_lumo)*0.5
    
    # Tableau des intégrales biélectroniques
    for p in range(nBas):
        for q in range(nBas):
            for r in range(nBas):
                for s in range(nBas):
                    ERI[p,q,r,s]  = Int[p,q,r,s] - Int[p,q,s,r]
    
    for p in range(nBas2):
        for q in range(nBas2):
            for r in range(nBas2):
                for s in range(nBas2):
                    sERI[p,q,r,s] = delta(p%2,r%2)*delta(q%2,s%2)*sInt[(p+1)//2,(q+1)//2,(r+1)//2,(s+1)//2]
    
    
    # Matrices A B C pour le TRIPLET
    
    # matrice B
    ab = 0
    for b in range(no,nBas-1):
        for a in range(b+1,nBas):
            ij = 0
            for j in range(no-1):
                for i in range(j+1,no):
                    B_aat[ab,ij] = ERI[a,b,i,j]
                    ij += 1
            ab += 1
            
    # matrice A
    ab = 0
    for b in range(0,nv-1):
        for a in range(b+1,nv):
            cd = 0
            for d in range(0,nv-1):                          
                for c in range(d+1,nv):
                    d1,d2        = delta(a,c), delta(b,d)
                    d3,d4        = delta(a,b),delta(c,d)
                    A_aat[ab,cd] = d1*d2*(ev[a]+ev[b]-2*nu) + Int[a+no,b+no,c+no,d+no]-Int[a+no,b+no,d+no,c+no]  
                    cd += 1
            ab += 1
           
    # matrice C
    ij = 0
    for j in range(no-1):
        for i in range(j+1,no):
            kl = 0
            for l in range(no-1):
                for k in range(l+1,no):
                    C_aat[ij,kl] = -(eo[i]+eo[j]-2*nu)*delta(k,i)*delta(j,l)+ERI[i,j,k,l]
                    kl += 1
            ij += 1
    
    
    #******************************************************************************
    A_aa = []
    for b in range(0,nv-1):
        for a in range(b+1,nv):
            for d in range(0,nv-1):
                for c in range(d+1,nv):
                    d1,d2       = delta(a,c), delta(b,d)
                    A_aa.append((d1*d2*(ev[a]+ev[b]-2*nu) + Int[a+no,b+no,c+no,d+no]-Int[a+no,b+no,d+no,c+no]))
                          
    l_aa = len(A_aa)
    A_aa = np.asmatrix(A_aa)    
    A_aa = A_aa.reshape(int(np.sqrt(l_aa)),int(np.sqrt(l_aa)))
    
    A_ab = []
    for b in range(0,nv-1):
        for a in range(b+1,nv):
            for d in range(0,nv-1):
                for c in range(d+1,nv):
                    d1,d2       = delta(a,c), delta(b,d)
                    A_ab.append((d1*d2*(ev[a]+ev[b]-2*nu) + Int[a+no,b+no,c+no,d+no]))
    l_ab = len(A_ab)
    A_ab = np.asmatrix(A_ab)
    A_ab = A_ab.reshape(int(np.sqrt(l_ab)),int(np.sqrt(l_ab)))
    
    
    
    C_aa = []
    for j in range(no-1):
        for i in range(j+1,no):
            for l in range(no-1):
                for k in range(l+1,no):
                    d1,d2       = delta(i,k), delta(j,l)
                    C_aa.append(-(eo[i]+eo[j]-2*nu)*d1*d2+Int[i,j,k,l]-Int[i,j,l,k])
                             
    C_ab = []
    for j in range(no-1):
        for i in range(j+1,no):
            for l in range(no-1):
                for k in range(l+1,no):
                    d1,d2       = delta(i,k), delta(j,l)
                    C_ab.append(-(eo[i]+eo[j]-2*nu)*d1*d2+Int[i,j,k,l])
    
    
    c_aa = len(C_aa)
    c_ab = len(C_ab)
    C_aa = np.asmatrix(C_aa)    
    C_ab = np.asmatrix(C_ab)
    C_aa = C_aa.reshape(int(np.sqrt(c_aa)),int(np.sqrt(c_aa)))
    C_ab = C_ab.reshape(int(np.sqrt(c_ab)),int(np.sqrt(c_ab)))
    
    
    zc_aaab  = np.zeros((len(C_aa),len(C_ab)))
    zc_aaabt = np.transpose(zc_aaab)
    zc_aaaa  = np.zeros((len(C_aa),len(C_aa)))
    
    B_aa = []
    for b in range(no,nBas-1):
        for a in range(b+1,nBas):
            for j in range(no-1):
                for i in range(j+1,no):
                    B_aa.append(Int[a,b,i,j]-Int[a,b,j,i])
                             
    B_ab = []
    for b in range(no,nBas-1):
        for a in range(b+1,nBas):
            for j in range(no-1):
                for i in range(j+1,no):
                    B_ab.append(Int[a,b,i,j])
    
    b_aa = len(B_aa)
    b_ab = len(B_ab)
    B_aa = np.asmatrix(B_aa)    
    B_ab = np.asmatrix(B_ab)
    B_aa = B_aa.reshape(int(len(A_aa)),int(len(C_aa)))
    B_ab = B_ab.reshape(int(len(A_ab)),int(len(C_ab)))
    
    
    
    Maat       = M(A_aat,B_aat,-B_aatt,-C_aat)
    Z_aat      = Tri2(Maat)[1]
    W_aat      = W(T_noo,T_nvv)
    
    Mabt       = M(A_abt,B_abt,-B_abtt,-C_abt)
    Z_abt      = Tri2(Mabt)[1]
    W_abt      = W(T_noo,T_nvv)
    
    
    Chi1_aat   = Z_aat[:,0:T_nvv]
    Chi2_aat   = Z_aat[:,T_nvv:T_nvv+T_noo] 
    Chi1_abt   = Z_abt[:,0:T_nvv]
    Chi2_abt   = Z_abt[:,T_nvv:T_nvv+T_noo] 


    # ORTHOGONALISATION DES VECTEURS PROPRES
    #------------------------------------------------------------------------------
    N1_aat       = np.dot(W_aat,Chi1_aat)
    N1_aat       = np.dot(np.transpose(Chi1_aat),N1_aat)
    
    N2_aat       = np.dot(W_aat,Chi2_aat)
    N2_aat       = np.dot(np.transpose(Chi2_aat),N2_aat)
    
    X1_aat       = Puissance(N1_aat,-1/2)
    X2_aat       = Puissance(-N2_aat,-1/2)
    
    Chi1_X1_aat  = np.dot(Chi1_aat,X1_aat)
    n1_aat       = np.dot(W_aat,Chi1_X1_aat) 
    N1_aat       = np.dot(np.transpose(Chi1_X1_aat),n1_aat)
    ZX1_aat      = Chi1_X1_aat[0:T_nvv,0:T_nvv]
    ZY1_aat      = Chi1_X1_aat[T_nvv:T_nvv+T_noo,0:T_nvv]
    
    Chi2_X2_aat  = np.dot(Chi2_aat,X2_aat)
    n2_aat       = np.dot(W_aat,Chi2_X2_aat) 
    N2_aat       = np.dot(np.transpose(Chi2_X2_aat),n2_aat)
    ZX2_aat      = Chi2_X2_aat[0:T_nvv,0:T_noo]
    ZY2_aat      = Chi2_X2_aat[T_nvv:T_nvv+T_noo,0:T_noo]
    #------------------------------------------------------------------------------
    N1_abt       = np.dot(W_abt,Chi1_abt)
    N1_abt       = np.dot(np.transpose(Chi1_abt),N1_abt)
    
    N2_abt       = np.dot(W_abt,Chi2_abt)
    N2_abt       = np.dot(np.transpose(Chi2_abt),N2_abt)
    
    X1_abt       = Puissance(N1_abt,-1/2)
    X2_abt       = Puissance(-N2_abt,-1/2)
    
    Chi1_X1_abt  = np.dot(Chi1_abt,X1_abt)
    n1_abt       = np.dot(W_abt,Chi1_X1_abt) 
    N1_abt       = np.dot(np.transpose(Chi1_X1_abt),n1_abt)
    ZX1_abt      = Chi1_X1_abt[0:T_nvv,0:T_nvv]
    ZY1_abt      = Chi1_X1_abt[T_nvv:T_nvv+T_noo,0:T_nvv]
    
    Chi2_X2_abt  = np.dot(Chi2_abt,X2_abt)
    n2_abt       = np.dot(W_abt,Chi2_X2_abt) 
    N2_abt       = np.dot(np.transpose(Chi2_X2_abt),n2_abt)
    ZX2_abt      = Chi2_X2_abt[0:T_nvv,0:T_noo]
    ZY2_abt      = Chi2_X2_abt[T_nvv:T_nvv+T_noo,0:T_noo]
    
    #------------------------------------------------------------------------------
    # CALCUL DE LA SELF-ENERGY
    #------------------------------------------------------------------------------        
    
    Chi_1_aat  = np.zeros((nBas,no,T_nvv))
    Chi_2_aat  = np.zeros((nBas,nv,T_noo))
    SigC_aat   = np.zeros((nBas,nBas))
    dSigC_aat  = np.zeros((nBas,nBas))
    Z_aa       = np.zeros(nBas)
    
    Chi_1_abt  = np.zeros((nBas,no,T_nvv))
    Chi_2_abt  = np.zeros((nBas,nv,T_noo))
    SigC_abt   = np.zeros((nBas,nBas))
    dSigC_abt  = np.zeros((nBas,nBas))
    Z_ab       = np.zeros(nBas)
    
    Chi_1_abdt = np.zeros((nBas,no,T_nvv))
    Chi_2_abdt = np.zeros((nBas,nv,T_noo))
    SigC_abdt  = np.zeros((nBas,nBas))
    dSigC_abdt = np.zeros((nBas,nBas))
    Z_abd      = np.zeros(nBas)
    
    
    
    omega_Taa1 = np.diag(Tri2(Maat)[0])[0:T_nvv]
    omega_Taa2 = np.diag(Tri2(Maat)[0])[T_nvv:T_nvv+T_noo]
    omega_Tab1 = np.diag(Tri2(Mabt)[0])[0:T_nvv]
    omega_Tab2 = np.diag(Tri2(Mabt)[0])[T_nvv:T_nvv+T_noo]
    
    
    # Tableaux de Chi_1 et Chi_2 du TRIPLET       
    for m in range(T_nvv):
        for i in range(no):
            for p in range(nBas): 
                cd = 0
                for c in range(no,nBas-1):
                    for d in range(c+1,nBas):
                        Chi_1_aat[p,i,m]  += np.dot(ERI[p,i,c,d],ZX1_aat[cd,m])
                        Chi_1_abdt[p,i,m] += np.dot(Int[p,i,c,d],ZX1_abt[cd,m])   
                        cd += 1
                        
                kl = 0
                for k in range(no-1):
                    for l in range(k+1,no):
                        Chi_1_aat[p,i,m]  += np.dot(ERI[p,i,k,l],ZY1_aat[kl,m])
                        Chi_1_abdt[p,i,m] += np.dot(Int[p,i,k,l],ZY1_abt[kl,m])
                        kl += 1
    
    for m in range(T_noo):
        for a in range(nv): 
            for p in range(nBas):
                cd = 0
                for c in range(no,nBas-1):
                    for d in range(c+1,nBas):
                        Chi_2_aat[p,a,m]  += np.dot(ERI[p,no+a,c,d],ZX2_aat[cd,m])
                        Chi_2_abdt[p,a,m] += np.dot(Int[p,no+a,c,d],ZX2_abt[cd,m]) 
                        cd += 1
                 
                kl = 0       
                for k in range(no-1):
                    for l in range(k+1,no):
                        Chi_2_aat[p,a,m] += np.dot(ERI[p,no+a,k,l],ZY2_aat[kl,m])
                        Chi_2_abdt[p,a,m] += np.dot(Int[p,no+a,k,l],ZY2_abt[kl,m])
                        kl += 1
                        
                
    for p in range(nBas):
        for q in range(nBas):
            for m in range(T_nvv):
                for i in range(no):
                    tmp_aat          = np.dot(Chi_1_aat[p,i,m],Chi_1_aat[q,i,m])/(eps[p]+eo[i]-omega_Taa1[m].item())
                    SigC_aat[p,q]   += tmp_aat
                    dSigC_aat[p,q]  -= tmp_aat/(eps[p]+eo[i]-omega_Taa1[m].item())
                    
                    tmp_abdt         = np.dot(Chi_1_abdt[p,i,m],Chi_1_abdt[q,i,m])/(eps[p]+eo[i]-omega_Tab1[m].item())
                    SigC_abdt[p,q]  += tmp_abdt
                    dSigC_abdt[p,q] -= tmp_abdt/(eps[p]+eo[i]-omega_Tab1[m].item())
                    
            for m in range(T_noo):
                for a in range(nv):
                    tmp_aat          = np.dot(Chi_2_aat[p,a,m],Chi_2_aat[q,a,m])/(eps[p]+ev[a]-omega_Taa2[m].item())
                    SigC_aat[p,q]   += tmp_aat
                    dSigC_aat[p,q]  -= tmp_aat/(eps[p]+ev[a]-omega_Taa2[m].item())
                    
                    tmp_abt          = np.dot(Chi_2_abt[p,a,m],Chi_2_abt[q,a,m])/(eps[p]+ev[a]-omega_Tab2[m].item())
                    SigC_abt[p,q]   += tmp_abt
                    dSigC_abt[p,q]  -= tmp_abt/(eps[p]+ev[a]-omega_Tab2[m].item())
                    
                    tmp_abdt         = np.dot(Chi_2_abdt[p,a,m],Chi_2_abdt[q,a,m])/(eps[p]+ev[a]-omega_Tab2[m].item())
                    SigC_abdt[p,q]  += tmp_abdt
                    dSigC_abdt[p,q] -= tmp_abdt/(eps[p]+ev[a]-omega_Tab2[m].item())
                    
                    
    for p in range(nBas):
        Z_aa[p]  = 1/(1 - dSigC_aat[p,p])
        Z_abd[p] = 1/(1 - dSigC_abt[p,p]) 
