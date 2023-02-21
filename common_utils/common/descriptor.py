from numpy import mat
import numpy as np
import copy
from numpy.linalg import norm
# difine by our group
from MathFunction import norm_d


class NormIndex:
    # =============================================================================
    #  The calculation of norm index from the mtatices combined by DS(step/distance matrices) and PQ(property matrices)
    #  NI is the norm indexes have been calculated
    #  M1~4 are the combination method
    # =============================================================================
    def NormIndex(DS, PQ):
        NI = mat([])
        for i in DS.values():
            for j in PQ.values():
                M1 = np.multiply(i, (j.dot(j.T)))
                M2 = np.multiply(i, abs(j - j.T))
                M3 = i.dot(j.dot(j.T))
                M4 = i.dot(abs(j - j.T))
                NI = np.c_[NI, mat(
                    [norm_d(M1, 1), norm_d(M1, 2), norm_d(M1, 3), norm(M1, ord='fro'), norm(M1, ord=2), norm(M1, ord=1),
                     norm_d(M2, 1), norm_d(M2, 2), norm_d(M2, 3), norm(M2, ord='fro'), norm(M2, ord=2), norm(M2, ord=1),
                     norm_d(M3, 1), norm_d(M3, 2), norm_d(M3, 3), norm(M3, ord='fro'), norm(M3, ord=2), norm(M3, ord=1),
                     norm_d(M4, 1), norm_d(M4, 2), norm_d(M4, 3), norm(M4, ord='fro'), norm(M4, ord=2),
                     norm(M4, ord=1)])]
        return NI

    # =============================================================================
    # The name of norm index for the mtatices combined by DS(step/distance matrices) and PQ(property matrices)
    # DSType is the type of DS(step/distance matrices)
    # PQType is the type of PQ(property matrices)
    # name_NI is the name of norm indexes have been calculated
    # =============================================================================
    def NameNormIndex(DS, DSType, PQ, PQType):
        name_NI = mat([])
        for i in DS.keys():
            for j in PQ.keys():
                M1 = f'{DSType[0]}{i}{DSType[1]}.*({PQType[0]}{j}*{PQType[0]}{j}.T)'
                M2 = f'{DSType[0]}{i}{DSType[1]}.*|{PQType[0]}{j}-{PQType[0]}{j}.T|'
                M3 = f'{DSType[0]}{i}{DSType[1]}*({PQType[0]}{j}*{PQType[0]}{j}.T)'
                M4 = f'{DSType[0]}{i}{DSType[1]}*|{PQType[0]}{j}-{PQType[0]}{j}.T|'
                name_NI = np.c_[name_NI, mat(
                    [f'norm_d({M1}, 1)', f'norm_d({M1}, 2)', f'norm_d({M1}, 3)', f'norm({M1}, fro)', f'norm({M1}, 2)',
                     f'norm({M1}, 1)',
                     f'norm_d({M2}, 1)', f'norm_d({M2}, 2)', f'norm_d({M2}, 3)', f'norm({M2}, fro)', f'norm({M2}, 2)',
                     f'norm({M2}, 1)',
                     f'norm_d({M3}, 1)', f'norm_d({M3}, 2)', f'norm_d({M3}, 3)', f'norm({M3}, fro)', f'norm({M3}, 2)',
                     f'norm({M3}, 1)',
                     f'norm_d({M4}, 1)', f'norm_d({M4}, 2)', f'norm_d({M4}, 3)', f'norm({M4}, fro)', f'norm({M4}, 2)',
                     f'norm({M4}, 1)'])]
        return name_NI


class GroupPosition:
    # =============================================================================
    # The defination of group (Group)
    # Lbon_str the bond in liner str
    # Ladj is adjacent relationship of atoms
    # Ladj_H is adjacent relationship of hydrogen atoms with all atoms
    # iCon_nH is the index of connection atom in the Hydrogen-supressed structure
    # =============================================================================
    def group(Latom, Ladj, Ladj_H, Lbon_str):
        Latom_H = copy.deepcopy(Latom)
        L = len(Latom_H)
        Con = mat(np.zeros((L, 1)))
        Group = copy.deepcopy(Latom)
        Left, Right = [], []
        for i in range(0, L):
            Left.append('(')
            Right.append(')')
            L_H = len(Ladj_H[i])
            L_L = len(Ladj[i]) - L_H
            if L_L > 1:
                Con[i, 0] = 1
                Left[i] = '['
                Right[i] = ']'
            elif L_L == 1:
                if L_H == 1:
                    Latom_H[i] += 'H'
                elif L_H > 1:
                    Latom_H[i] += f'H{str(L_H)}'

        for i, k in enumerate(Ladj):
            Group_ = []
            for j, w in enumerate(k):
                Group_.append(f'{Left[w - 1]}{Lbon_str[i][j]}{Latom_H[w - 1]}{Right[w - 1]}')
            Group_ = np.sort(Group_)
            for j in Group_:
                Group[i] += j
        iend = np.where(Con == 0)[0]
        Group = np.delete(Group, iend, 0).T.tolist()

        inH = np.where(mat(Latom) != 'H')[1]
        Con_nH = copy.deepcopy(Con)[inH, :]
        iCon_nH = np.where(Con_nH == 1)[0]

        return Group, iCon_nH

    # =============================================================================
    # The position factor (PF) from DS(step/distance matrices) and PQ(property matrices)
    # The group from distance and quantum parameter
    # DSType is the type of DS
    # PQType is the type of PQ
    # iCon_nH is the index of connection atom in the Hydrogen-supressed structure
    # name is the name of position
    # =============================================================================
    def GroupPosition(DS, PQ, iCon_nH):
        k = 0
        for k0, i in DS.items():
            for k1, j in PQ.items():
                MM = np.multiply(i, (j.dot(j.T)))
                smm = np.sqrt(sum(abs(MM)))[0, iCon_nH]
                if k == 0:
                    GNPF = smm
                    k += 1
                else:
                    GNPF = np.r_[GNPF, smm]
        return GNPF.T

    # =============================================================================
    # The name of position factor (PF) from DS(step/distance matrices) and PQ(property matrices)
    # The group from distance and quantum parameter
    # DSType is the type of DS
    # PQType is the type of PQ
    # iCon_nH is the index of connection atom in the Hydrogen-supressed structure
    # name is the name of position
    # =============================================================================
    def NameGroupPosition(DS, DSType, PQ, PQType):
        name = []
        for k0 in DS.keys():
            name_=[]
            for k1 in PQ.keys():
                name_.append(f'sqrt(abs(sum({DSType[0]}{k0}{DSType[1]}.*({PQType[0]}{k1}*{PQType[0]}{k1}.T)))')
            name.append(name_)
        return name

class MolecularInformation:
    # iformtion from S
    def InformationFromStep(Latom, Mwei, SF_H):
        n_atom = len(Latom)
        iH = np.where(mat(Latom) == 'H')[1]
        nnH = len(iH)
        mw = np.sum(abs(Mwei), axis=0)[0, 0]
        st_m = np.max(SF_H)
        st_s = np.sum(SF_H)
        if_s = [n_atom, n_atom - nnH, mw, st_m, st_s]
        return mat(if_s)

    # iformtion from D
    def InformationFromDistance(DF_H):
        st_m = np.max(DF_H)
        st_s = np.sum(DF_H)
        if_d = [st_m, st_s]
        return mat(if_d)
