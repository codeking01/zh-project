from numpy import mat
from numpy import array
import numpy as np
import math
import copy
from numpy import power as p
# difine by our group
import AtomicProperty as AtomicProperty

ap = getattr(AtomicProperty, 'AtomicProperty')


class SIFromFile:
    # =============================================================================
    # Calculate the matrix and list from the initial structure information of hin
    # The structure information from hin
    # Latom is the atom
    # n_atom is the number of atom
    # Mbon is the matrix of chemical bond
    # Sa is adjacent matrix
    # Ladj is adjacent relationship of atoms
    # Ladj_H is adjacent relationship of hydrogen atoms with all atoms
    # Ladj_nH is adjacent relationship of none hydrogen atoms with all atoms
    # Lbon is the chemical bond
    # iH is the index of hydrogen
    # SF is full step matrix
    # aro is aromatics atoms
    # Ladj_nl is adjacent relationship of  atoms not in the end
    # lne is is the index of atom not in the end
    # 1.5(b_aro) is the bond in aromatics ring
    # cyc is the atom on ring
    # Ddis is atomic coordinates
    # cha is the atomic charge
    # =============================================================================
    def hin(ISI):
        si, Ladj, Lbon = SIFunction.StrFromHin(ISI)
        si = array(si)
        Latom = si[:, 3].tolist()
        n_atom = len(Latom)
        cha = list(map(float, si[:, 6]))
        Ddis = SIFunction.coordinate(si[:, 7:10])
        Sa, Sbon = SIFunction.AdjacentBond(n_atom, Ladj, Lbon)
        Ladj, Ladj_H, Ladj_nH, Lbon = SIFunction.AdjacentLists(n_atom, Latom, Sa, Sbon)
        iH = np.where(mat(Latom) == 'H')[1]
        SF = SIFunction.FullStepbyNoneH(n_atom, Sa, Ladj_nH, iH)
        aro = SIFunction.Aromatics(n_atom, Lbon, 1.5)
        Ladj_nl, lne = SIFunction.DeleteEndAtom(Ladj_nH, Latom)
        cyc = SIFunction.AtomOnRing(SF, Ladj_nl, lne, aro)
        SI = {'n_atom': n_atom, 'Sa': Sa, 'SF': SF, 'Lbon': Lbon, 'Sbon': Sbon, 'Latom': Latom, 'aro': aro,
              'Ladj': Ladj, 'Ladj_H': Ladj_H, 'Ladj_nH': Ladj_nH, 'cyc': cyc, 'cha': cha, 'Ddis': Ddis}
        return SI

    # =============================================================================
    # Calculate the matrix and list from the initial structure information of mol
    # The structure information from mol
    # Latom is the atom
    # n_atom is the number of atom
    # Mbon is the matrix of chemical bond
    # Sa is adjacent matrix
    # Ladj is adjacent relationship of atoms
    # Ladj_H is adjacent relationship of hydrogen atoms with all atoms
    # Ladj_nH is adjacent relationship of none hydrogen atoms with all atoms
    # Lbon is the chemical bond
    # iH is the index of hydrogen
    # SF is full step matrix
    # aro is aromatics atoms
    # Ladj_nl is adjacent relationship of  atoms not in the end
    # lne is is the index of atom not in the end
    # 1.5(b_aro) is the bond in aromatics ring
    # cyc is the atom on ring
    # Ddis is atomic coordinates
    # cha is the atomic charge
    # =============================================================================
    def mol(ISI):
        ISI = copy.deepcopy(ISI)
        si, n_atom, n_adj, n_atom_start = SIFunction.StrFromMol(ISI)
        Ddis = SIFunction.coordinate(si[:, 0:3])
        Latom = si[:, 3].tolist()
        Sa, Sbon = SIFunction.StepBondFromMol(n_atom, n_atom_start, n_adj, ISI)
        Ladj, Ladj_H, Ladj_nH, Lbon = SIFunction.AdjacentLists(n_atom, Latom, Sa, Sbon)
        iH = np.where(mat(Latom) == 'H')[1]
        SF = SIFunction.FullStepbyNoneH(n_atom, Sa, Ladj_nH, iH)
        aro = SIFunction.Aromatics(n_atom, Lbon, 1.5)
        Ladj_nl, lne = SIFunction.DeleteEndAtom(Ladj_nH, Latom)
        cyc = SIFunction.AtomOnRing(SF, Ladj_nl, lne, aro)
        SI = {'n_atom': n_atom, 'Sa': Sa, 'SF': SF, 'Lbon': Lbon, 'Sbon': Sbon, 'Latom': Latom, 'aro': aro,
              'Ladj': Ladj, 'Ladj_H': Ladj_H, 'Ladj_nH': Ladj_nH, 'cyc': cyc, 'Ddis': Ddis}
        return SI

    # =============================================================================
    # Calculate the matrix and list from the initial structure information of log
    # Qcn: the charge for natural population analysis (NPA)
    # Qncn: the number of core electrons for NPA
    # Qnvn: the number of valence electrons for NPA
    # QnRn: the number of Rydberg electrons for NPA
    # Qntn: the number of total electrons for NPA
    # Qec: the electrostatic potential (esp) charges
    # QMc: the Mulliken charges
    # Qep: electrostatic properties
    # Qecn: the energy of core electrons for NPA
    # Qevn: the energy of valence electrons for NPA
    # QeRn: the energy of Rydberg electrons for NPA
    # Qetn: the energy of total electrons for NPA
    # Qgc: the gross orbital population of core electrons
    # Qgv: the gross orbital population of valence electrons
    # QgR: the gross orbital population of Rydberg electrons
    # Qgt: the gross orbital population of total electrons
    # Qdis:the distance matrix
    # =============================================================================
    def log(ISI):
        ISI = copy.deepcopy(ISI)
        iq = SIFunction.QuantumIndex(ISI)
        s = ISI[iq['Total'] - 2]
        n_ato = SIFunction.ParameterFromStr(s, 1, int)
        s = ISI[iq['Alpha  occ. eigenvalues']]
        Homo = SIFunction.ParameterFromStr(s, -1, float)
        s = ISI[iq['Alpha  occ. eigenvalues'] + 1]
        Lumo = SIFunction.ParameterFromStr(s, 4, float)
        s = ISI[iq['Electronic spatial extent']]
        ese = SIFunction.ParameterFromStr(s, -1, float)
        s = ISI[iq['Dipole moment'] + 1]
        dm = SIFunction.ParameterFromStr(s, -1, float)
        hl = [n_ato, Homo, Lumo, ese, dm]

        # atomic coordinates
        s = ISI[iq['Standardorientation'] + 5:iq['Standardorientation'] + 5 + n_ato]
        Ddis = mat(SIFunction.ParameterFromStrs(s, (3,), float))
        # Mulliken charges
        s = ISI[iq['Mulliken charges'] + 2:iq['Mulliken charges'] + 2 + n_ato]
        Mc = mat(SIFunction.ParameterFromStrs(s, (2, 3), float))

        # Natural Population Analysis
        s = ISI[
            iq['Summary of Natural Population Analysis'] + 6:iq['Summary of Natural Population Analysis'] + 6 + n_ato]
        npa = mat(SIFunction.ParameterFromStrs(s, (2, 8), float))
        cn = npa[:, 0]
        ncn = npa[:, 1]
        nvn = npa[:, 2]
        nrn = npa[:, 3]
        ntn = npa[:, 4]

        # Gross orbital populations (ecvrt) and natural atomic orbital energy (gcvrt)
        iq_g = iq['Gross orbital populations']
        iq_e0 = iq['NATURAL POPULATIONS'] + 4
        iq_e1 = iq['Summary of Natural Population Analysis'] - 3
        ecvrt, gcvrt = SIFunction.EnegyGrossFromStrs(iq_g, iq_e0, iq_e1, n_ato, ISI)
        ecn = ecvrt['Cor']
        evn = ecvrt['Val']
        ern = ecvrt['Ryd']
        etn = ecvrt['Tot']
        gc = gcvrt['Cor']
        gv = gcvrt['Val']
        gR = gcvrt['Ryd']
        gt = gcvrt['Tot']

        SI = {'hl': hl, 'Ddis': Ddis, 'Mc': Mc, 'cn': cn, 'ncn': ncn, 'nvn': nvn, 'nrn': nrn, 'ntn': ntn,
              'ecn': ecn, 'evn': evn, 'ern': ern, 'etn': etn, 'gc': gc, 'gv': gv, 'gR': gR, 'gt': gt}
        return SI

    # =============================================================================
    # Calculate the matrix and list from the initial structure information of gjf
    # The structure information from gjf
    # Latom is the atom
    # n_atom is the number of atom
    # Sbon is the matrix of chemical bond
    # Sa is adjacent matrix
    # Ladj is adjacent relationship of atoms
    # Ladj_H is adjacent relationship of hydrogen atoms with all atoms
    # Ladj_nH is adjacent relationship of none hydrogen atoms with all atoms
    # Lbon is the chemical bond
    # iH is the index of hydrogen
    # SF is full step matrix
    # aro is aromatics atoms
    # Ladj_nl is adjacent relationship of  atoms not in the end
    # lne is is the index of atom not in the end
    # 1.5(b_aro) is the bond in aromatics ring
    # cyc is the atom on ring
    # =============================================================================
    def gjf(ISI):
        ISI = copy.deepcopy(ISI)
        Latom, n_atom = SIFunction.AtomFromGjf(ISI)
        Sbon, Sa = SIFunction.BondStepFromGjf(n_atom, ISI)
        Ladj, Ladj_H, Ladj_nH, Lbon = SIFunction.AdjacentLists(n_atom, Latom, Sa, Sbon)
        iH = np.where(mat(Latom) == 'H')[1]
        SF = SIFunction.FullStepbyNoneH(n_atom, Sa, Ladj_nH, iH)
        aro = SIFunction.Aromatics(n_atom, Lbon, 1.5)
        Ladj_nl, lne = SIFunction.DeleteEndAtom(Ladj_nH, Latom)
        cyc = SIFunction.AtomOnRing(SF, Ladj_nl, lne, aro)
        Lbon_str = SIFunction.BondNumber2Line(Lbon)
        Lbon_str = SIFunction.BondLine2Curve(Lbon_str, SF, Ladj, Ladj_nl, lne)

        SI = {'n_atom': n_atom, 'Sa': Sa, 'SF': SF, 'Lbon': Lbon, 'Lbon_str': Lbon_str, 'Sbon': Sbon, 'Latom': Latom,
              'aro': aro,
              'Ladj': Ladj, 'Ladj_H': Ladj_H, 'Ladj_nH': Ladj_nH, 'cyc': cyc}
        return SI


class SIFunction:
    # =============================================================================
    # The calculation of full step matrix (SF) by none hydrogen atoms
    # n_atom is the number of atom
    # Sa is adjacent matrix
    # Ladj_nH is adjacent relationship of none hydrogen atoms with all atoms
    # iH is the index of hydrogen
    # =============================================================================
    def FullStepbyNoneH(n_atom, Sa, Ladj_nH, iH):
        SF = copy.deepcopy(Sa)
        SF[iH, :] = 0
        SF[:, iH] = 0
        w_ms = np.where(SF == 1)
        w_ms_r = w_ms[0]
        w_ms_c = w_ms[1]
        for m in range(1, n_atom - 1):
            if len(w_ms_r) == 0:
                break
            w_ms_r_ = []
            w_ms_c_ = []
            for k, i in enumerate(w_ms_r):
                w_msi = Ladj_nH[w_ms_c[k]]
                for j in w_msi:
                    if SF[i, j - 1] == 0 and i != j - 1:
                        SF[i, j - 1] = m + 1
                        SF[j - 1, i] = m + 1
                        w_ms_r_.append(i)
                        w_ms_c_.append(j - 1)
                        w_ms_r_.append(j - 1)
                        w_ms_c_.append(i)
            w_ms_r = w_ms_r_
            w_ms_c = w_ms_c_
        j_H = []
        for i in iH:
            j_H.append(Ladj_nH[i][0] - 1)
        SF[iH, :] = SF[j_H, :] + 1
        SF[:, iH] = SF[:, j_H] + 1
        SF[iH, iH] = 0
        return SF

    # =============================================================================
    # The calculation of full step matrix (SF)
    # n_atom is the number of atom
    # Sa is adjacent matrix
    # Ladj is adjacent relationship of atoms
    # =============================================================================
    def FullStep(n_atom, Sa, Ladj):
        SF = copy.deepcopy(Sa)
        w_ms = np.where(SF == 1)
        w_ms_r = w_ms[0]
        w_ms_c = w_ms[1]
        for m in range(1, n_atom - 1):
            if len(w_ms_r) == 0:
                break
            w_ms_r_ = []
            w_ms_c_ = []
            for i in range(0, len(w_ms_r)):
                w_msi = Ladj[w_ms_c[i]]
                for j in w_msi:
                    if SF[w_ms_r[i], j - 1] == 0 and w_ms_r[i] != j - 1:
                        SF[w_ms_r[i], j - 1] = m + 1
                        SF[j - 1, w_ms_r[i]] = m + 1
                        w_ms_r_.append(w_ms_r[i])
                        w_ms_c_.append(j - 1)
                        w_ms_r_.append(j - 1)
                        w_ms_c_.append(w_ms_r[i])
            w_ms_r = w_ms_r_
            w_ms_c = w_ms_c_
        return SF

    # =============================================================================
    # n_atom is the number of atom
    # Sa is adjacent matrix
    # Ladj is adjacent relationship of atoms
    # =============================================================================
    def AdjacentStep(n_atom, Ladj):
        Sa = mat(np.zeros((n_atom, n_atom)))
        for i in range(0, n_atom):
            Sa[i, np.array(Ladj[i]) - 1] = 1
        return Sa

    # =============================================================================
    # n_atom is the number of atom
    # Sa is adjacent matrix
    # Ladj is adjacent relationship of atoms
    # Lbon is the chemical bond
    # =============================================================================
    def AdjacentBond(n_atom, Ladj, Lbon):
        Sa = mat(np.zeros((n_atom, n_atom)))
        Sbon = mat(np.zeros((n_atom, n_atom)))
        for i in range(0, n_atom):
            Sa[i, np.array(Ladj[i]) - 1] = 1
            Sbon[i, np.array(Ladj[i]) - 1] = mat(Lbon[i])
        return Sa, Sbon

    # =============================================================================
    # The calculation of full distance matrix (DF)
    # n_atom is the number of atom
    # Mdis is atomic coordinates
    # =============================================================================
    def FullDistance(n_atom, Mdis):
        Mdis = copy.deepcopy(Mdis)
        DF = mat(np.zeros((n_atom, n_atom)))
        for i in range(n_atom - 1):
            for j in range(i + 1, n_atom):
                DF[i, j] = np.linalg.norm(Mdis[i, :] - Mdis[j, :], ord=2)
                DF[j, i] = DF[i, j]
        return DF

    # =============================================================================
    # n_atom is the number of atom
    # Sa is adjacent matrix
    # Ladj is adjacent relationship of atoms
    # =============================================================================
    def AdjacentList(Sa):
        n_atom = np.shape(Sa)[0]
        Ladj = []
        for i in range(0, n_atom):
            w_ms = np.where(Sa[i, :] == 1)
            Ladj.append(list(w_ms[1] + 1))
        return Ladj

    # =============================================================================
    # The calculation of bond matrix (Sbon)
    # n_atom is the number of atom
    # Sa is adjacent matrix
    # Ladj is adjacent relationship of atoms
    # Lbon is the chemical bond
    # =============================================================================
    def BondStep4to1_5(n_atom, Ladj, Lbon):
        Sbon = mat(np.zeros((n_atom, n_atom), dtype=float))
        for i in range(0, n_atom):
            Sbona = np.array(Lbon[i], dtype=float)
            Sbona[Sbona == 4] = 1.5
            Sbon[i, np.array(Ladj[i]) - 1] = Sbona
        return Sbon

    # =============================================================================
    # The adjacent relationship of atoms
    # n_atom is the number of atom
    # Sa is adjacent matrix
    # Ladj is adjacent relationship of atoms
    # Ladj_nH is adjacent relationship of none hydrogen atoms with all atoms
    # Ladj_H is adjacent relationship of hydrogen atoms with all atoms
    # Sbon is the matrix of chemical bond
    # =============================================================================
    def BondStepFromGjf(n_atom, orimsi_all):
        Sa = mat(np.zeros((n_atom, n_atom)))
        Sbon = mat(np.zeros((n_atom, n_atom)))
        for i_L in orimsi_all:
            i_L = i_L.split()
            if len(i_L) > 2 and i_L[0].isdigit():
                i_L_n = list(map(float, i_L))
                i_L_n_i = np.hstack((i_L_n[0], i_L_n[1::2])) - 1
                i_L_n_i = i_L_n_i.astype(np.int)
                i_l0 = i_L_n_i[0]
                i_l1 = list(i_L_n_i[1::])
                i_l2 = i_L_n[2::2]
                for i in range(0, len(i_l1)):
                    i_s = min([i_l0, i_l1[i]])
                    i_g = max([i_l0, i_l1[i]])
                    Sa[i_s, i_g] = 1
                    Sbon[i_s, i_g] = i_l2[i]
        Sa = Sa + Sa.T
        Sbon = Sbon + Sbon.T
        return Sbon, Sa

    # =============================================================================
    # The adjacent relationship of atoms
    # n_atom is the number of atom
    # n_atom is the number of atom
    # n_adj is the number of adj
    # Sa is adjacent matrix
    # Sbon is the matrix of chemical bond
    # =============================================================================
    def StepBondFromMol(n_atom, n_atom_start, n_adj, ISI):
        ISI = copy.deepcopy(ISI)
        Sa = mat(np.zeros((n_atom, n_atom)))
        Sbon = mat(np.zeros((n_atom, n_atom)))
        for j in range(n_atom_start + n_atom, n_atom_start + n_atom + n_adj):
            i_s = min([int(ISI[j][0:3]) - 1, int(ISI[j][3:6]) - 1])
            i_g = max([int(ISI[j][0:3]) - 1, int(ISI[j][3:6]) - 1])
            Sa[i_s, i_g] = 1
            Sbon[i_s, i_g] = int(ISI[j][6:9])
        Sa = Sa + Sa.T
        Sbon = Sbon + Sbon.T

    # =============================================================================
    # The atom on ring (Pcyc)
    # SF is full step matrix
    # lne is is the index of atom not in the end
    # Paro is aromatics atoms
    # Ladj_nl is adjacent relationship of  atoms not in the end
    # =============================================================================
    def AtomOnRing(SF, Ladj_nl, lne, Paro):
        Cyc = copy.deepcopy(Paro)
        while [] in Ladj_nl:
            Ladj_nl.remove([])
        Ladj_nl_ = copy.deepcopy(Ladj_nl)
        lne_ = copy.deepcopy(lne)
        w_a = np.where(Paro == 1)[0]
        for j in w_a:
            i_j = lne_.index(j)
            lne_.remove(j)
            del Ladj_nl_[i_j]
        for i, k in enumerate(Ladj_nl_):
            i_l = lne_[i]
            if len(k) > 0:
                for j in lne:
                    wl = (np.where(SF[j, (mat(k) - 1).tolist()[0]] < SF[j, i_l] + 1)[1]).tolist()
                    if len(wl) > 1:
                        Cyc[i_l, 0] = 1
                        break
        return Cyc

    # =============================================================================
    # The adjacent relationship of atoms
    # n_atom is the number of atom
    # Sa is adjacent matrix
    # Ladj is adjacent relationship of atoms
    # Ladj_nH is adjacent relationship of none hydrogen atoms with all atoms
    # Ladj_H is adjacent relationship of hydrogen atoms with all atoms
    # Lbon is the chemical bond
    # =============================================================================
    def AdjacentLists(n_atom, Latom, Sa, Sbon):
        Ladj_H = []
        Ladj_nH = []
        Ladj = []
        Lbon = []
        for j in range(0, n_atom):
            Ladj_ = np.where(Sa[j, :] > 0)
            Ladj.append(list(1 + Ladj_[1]))
            Lbon.append(Sbon[j, Ladj_[1]].tolist()[0])
            Ladj__ = []
            Ladj_nH_ = []
            for i in Ladj_[1]:
                if Latom[i] == 'H':
                    Ladj__.append(i + 1)
                else:
                    Ladj_nH_.append(i + 1)
            Ladj_nH.append(Ladj_nH_)
            Ladj_H.append(Ladj__)
        return Ladj, Ladj_H, Ladj_nH, Lbon

    # =============================================================================
    # delete the atoms in thr end
    # Latom is the atom
    # Ladj_nl is adjacent relationship of  atoms not in the end
    # lne is is the index of atom not in the end
    # =============================================================================
    def DeleteEndAtom(Ladj_nH, Latom):
        Ladj_nl = copy.deepcopy(Ladj_nH)
        lne = []
        for i, j in enumerate(Ladj_nl):
            if Latom[i] == 'H':
                Ladj_nl[i] = []
            elif len(j) > 0:
                lne.append(i)
        isem = 1
        while isem > 0:
            isem = 0
            for i in lne:
                if len(Ladj_nl[i]) == 1:
                    isem = 1
                    ii_0 = Ladj_nl[i][0] - 1
                    ii_1 = i + 1
                    Ladj_nl[ii_0].remove(ii_1)
                    Ladj_nl[i] = []
                    lne.remove(i)
        return Ladj_nl, lne

    # =============================================================================
    #  add the properties of hydrogen to heavyatom
    #  Ladj_H is the adjacent matrix for hydrogen
    #  Q is a a dict for the quantum parameter marix
    #  n_atom is the number of atoms
    # =============================================================================
    def Hydrogen2HeavyAtom(Ladj_H, Q, n_atom):
        Q = copy.deepcopy(Q)
        for i in Q:
            for j in range(0, n_atom):
                if not Ladj_H[j]:
                    Q[i][j] = Q[i][j] + sum(Q[i][Ladj_H[j]])
        return Q

    # =============================================================================
    # The aromatics atoms (Paro)
    # n_atom is the number of atom
    # Lbon is the chemical bond
    # b_aro is the bond in aromatics ring
    # =============================================================================
    def Aromatics(n_atom, Lbon, b_aro):
        Paro = mat(np.zeros((n_atom, 1)))
        for k in range(n_atom):
            if len(np.where(mat(Lbon[k]) == b_aro)[1]) > 0:
                Paro[k, 0] = 1
        return Paro

    # =============================================================================
    # The aromatics atoms (Latom) from gjf
    # n_atom is the number of atom
    # =============================================================================
    def AtomFromGjf(orimsi_all):
        Latom = []
        n_atom = 0;
        for i_L in orimsi_all:
            if i_L[5:12] == '       ':
                n_atom = n_atom + 1
                i_L = i_L.replace('\n', '')
                i_L = i_L.split()
                i_at = i_L[0]
                Latom.append(i_at)
        return Latom, n_atom

    # =============================================================================
    # The index of quantum parameter
    # =============================================================================
    def QuantumIndex(orimsi):
        orimsi = copy.deepcopy(orimsi)
        i_n = 0
        i_q = {}
        for i_L in orimsi:
            if i_L.startswith(' Mulliken charges:'):
                i_q['Mulliken charges'] = i_n
            elif i_L.startswith('                         Standard orientation:'):
                i_q['Standardorientation'] = i_n
            elif i_L.startswith('          Condensed to atoms (all electrons):'):
                i_q['Condensed to atoms'] = i_n
            elif i_L.startswith('     Gross orbital populations:'):
                i_q['Gross orbital populations'] = i_n
            elif i_L.startswith(' NATURAL POPULATIONS:  Natural atomic orbital occupancies '):
                i_q['NATURAL POPULATIONS'] = i_n
            elif i_L.startswith(' Summary of Natural Population Analysis: '):
                i_q['Summary of Natural Population Analysis'] = i_n
            elif i_L.startswith(' Dipole orientation:'):
                i_q['Dipole orientation'] = i_n
            elif i_L.startswith('   * Total *'):
                i_q['Total'] = i_n
            elif i_L.startswith('       nuclear repulsion energy'):
                i_q['nuclear repulsion energy'] = i_n
            elif i_L.startswith(' Alpha  occ. eigenvalues'):
                i_q['Alpha  occ. eigenvalues'] = i_n
            elif i_L.startswith(' Electronic spatial extent (au):'):
                i_q['Electronic spatial extent'] = i_n
            elif i_L.startswith(' Dipole moment (field-independent basis, Debye):'):
                i_q['Dipole moment'] = i_n
            i_n = i_n + 1
        return i_q

    # =============================================================================
    # Extract one parameter from s
    # i is index of parameter in s
    # dtype is the type of data
    # =============================================================================
    def ParameterFromStr(s, i, dtype):
        s = copy.deepcopy(s)
        s = s.replace('\n', '')
        s = s.split()
        parameter = dtype(s[i])
        return parameter

    # =============================================================================
    # Extract parameters from s
    # i is index of parameter in s
    # dtype is the type of data
    # =============================================================================
    def ParameterFromStrs(s, i, dtype):
        ii = []
        if len(i) == 1:
            for k in s:
                k = k.replace('\n', '')
                k = k.split()
                ii.append(list(map(dtype, k[i[0]:])))
        elif len(i) == 2:
            for k in s:
                k = k.replace('\n', '')
                k = k.split()
                ii.append(list(map(dtype, k[i[0]:i[1]])))
        return ii

    # =============================================================================
    # Extract atomic enegy and gross orbital populations from orimsi
    # iq_g is index of gross orbital populations
    # iq_e0 and iq_e1 are the begining and end indexes of atomic enegy
    # dtype is the type of data
    # ecvrt is the atomic enegy of Cor, Val, Ryd and total obital
    # gcvrt is the gross orbital population of Cor, Val, Ryd and total obital
    # =============================================================================
    def EnegyGrossFromStrs(iq_g, iq_e0, iq_e1, n_ato, orimsi):
        orimsi = copy.deepcopy(orimsi)
        ecvrt = {'Cor': mat(np.zeros((n_ato, 1))), 'Val': mat(np.zeros((n_ato, 1))), 'Ryd': mat(np.zeros((n_ato, 1))),
                 'Tot': mat(np.zeros((n_ato, 1)))}
        gcvrt = {'Cor': mat(np.zeros((n_ato, 1))), 'Val': mat(np.zeros((n_ato, 1))), 'Ryd': mat(np.zeros((n_ato, 1))),
                 'Tot': mat(np.zeros((n_ato, 1)))}
        kk = 0
        kkk = 0
        for k in range(iq_e0, iq_e1):
            if orimsi[k].isspace():
                kk = kk + 1
            else:
                i_g = orimsi[iq_g + kkk + 2]
                i_g = i_g.replace('\n', '')
                i_g = i_g.split()
                kkk = kkk + 1
                i_ = orimsi[k]
                i_ = i_.replace('\n', '')
                i_ = i_.split()
                i_1 = int(i_[2]) - 1
                i_2 = float(i_[-1])
                ecvrt[i_[4][0:3]][i_1] += i_2
                ecvrt['Tot'][i_1] += i_2
                gcvrt[i_[4][0:3]][i_1] += float(i_g[-1])
                gcvrt['Tot'][i_1] += float(i_g[-1])
            if kk == n_ato: break
        return ecvrt, gcvrt

    # =============================================================================
    # The struture information (si) fron hin
    # Ladj is adjacent relationship of atoms
    # Lbon is the chemical bond
    # =============================================================================
    def StrFromHin(ISI):
        ISI = copy.deepcopy(ISI)
        si = []
        Ladj = []
        Lbon = []
        for i_L in ISI:
            if i_L.startswith('atom'):
                i_L = i_L.replace('\n', '')
                i_L = i_L.replace('s', '1')
                i_L = i_L.replace('d', '2')
                i_L = i_L.replace('t', '3')
                i_L = i_L.replace('a', '1.5')
                i_L = i_L.split()
                si.append(i_L[0:11])
                # Ladj.append(i_L[11:-1:2])
                Ladj.append(list(map(int, i_L[11:-1:2])))
                Lbon.append(list(map(float, i_L[12::2])))
        return si, Ladj, Lbon

    def coordinate(si):
        si = copy.deepcopy(si)
        dis = [];
        for i in range(0, 3):
            dis.append(list(map(float, si[:, i])))
        dis = mat(list(map(list, zip(*dis))))
        return dis

    # =============================================================================
    # The struture information (si) fron mol
    # n_atom is the number of atom
    # n_adj is the number of adj
    # =============================================================================
    def StrFromMol(ISI):
        ISI = copy.deepcopy(ISI)
        si = []
        n_atom_start = 4;
        n_atom = int(ISI[3][0:3])
        n_adj = int(ISI[3][3:6])
        for j in range(n_atom_start, n_atom_start + n_atom):
            si.append(ISI[j].split())
        si = np.array(si)
        return si, n_atom, n_adj, n_atom_start

    # =============================================================================
    # Change the bond from number to liner str
    # Lbon_str the bond in liner str
    # Lbon the bond in number
    # =============================================================================
    def BondNumber2Line(Lbon):
        Lbon_str = []
        num2lin = {1: '-', 2: '=', 3: '≡', 1.5: '∷'}
        for i, j in enumerate(Lbon):
            bon_str = []
            for k in j:
                bon_str.append(num2lin[k])
            Lbon_str.append(bon_str)
        return Lbon_str

    # =============================================================================
    # Change the liner str  to noneline str in the ring
    # Lbon_str the bond in liner str
    # Ladj is adjacent relationship of atoms
    # Ladj_nl is adjacent relationship of  atoms not in the end
    # lne is is the index of atom not in the end
    # SF is full step matrix
    # =============================================================================
    def BondLine2Curve(Lbon_str, SF, Ladj, Ladj_nl, lne):
        lin2nlin = {'-': '～', '=': '≈', '≡': '≋', '∷': '∷'}
        lin2nlin_keys = lin2nlin.keys()
        while [] in Ladj_nl:
            Ladj_nl.remove([])
        for i, k in enumerate(Ladj_nl):
            i_l = lne[i]
            if len(k) > 0:
                for j in lne:
                    wl = (np.where(SF[j, (mat(k) - 1).tolist()[0]] < SF[j, i_l] + 1)[1]).tolist()
                    if len(wl) > 1:
                        for k_wl in wl:
                            i_s = np.where(Ladj[i_l] == k[k_wl])[0][0]
                            if Lbon_str[i_l][i_s] in lin2nlin_keys:
                                Lbon_str[i_l][i_s] = lin2nlin[Lbon_str[i_l][i_s]]
        return Lbon_str


class PositionProperty:
    # =============================================================================
    # the properties of atoms are from http://environmentalchemistry.com/yogi/periodic/
    # wei is atomic weight
    # rad is atomic radius (Å)
    # noe is number of outermost electrons
    # nes is number of electrons shell
    # ion is Ionization energy(eV)
    # =============================================================================
    def properties(name_atom):
        ele = [];
        wei = [];
        rad = [];
        noe = [];
        nes = [];
        ion = [];
        for atom in name_atom:
            ele.append(ap['ele'][atom])
            wei.append(ap['wei'][atom])
            rad.append(ap['rad'][atom])
            noe.append(ap['noe'][atom])
            nes.append(ap['nes'][atom])
            ion.append(ap['ion'][atom])
        P = {'ele': mat(ele).T, 'wei': mat(wei).T, 'rad': mat(rad).T, 'noe': mat(noe).T, 'nes': mat(nes).T,
             'ion': mat(ion).T}
        return P

    # =============================================================================
    # The step matrices and property matrices
    # a is the adjacent step matix
    # b is the interphase step matix
    # c is the jump step matix
    # B is the adjacent and interphase step matix
    # C is the adjacent interphase and jump step matix
    # F is the full step matrix
    # bon is the chemical bond matrix
    # bra is the branch degree
    # cyc is the atom on ring
    # aro is the atom on aromatic ring
    # S is step marix
    # S_H is the step matrix of Hydrogen-suppressed structure
    # P is property marix
    # P_H is the propertyp matrix of Hydrogen-suppressed structure
    # =============================================================================
    def step(SI):
        SI = copy.deepcopy(SI)
        a = SI['Sa']
        F = SI['SF']
        aro = SI['aro']
        cyc = SI['cyc']
        cyc = cyc - aro
        bon = SI['Sbon']
        b = copy.deepcopy(F)
        b[F != 2] = 0
        c = copy.deepcopy(F)
        c[F != 3] = 0
        B = copy.deepcopy(F)
        B[F > 2] = 0
        C = copy.deepcopy(F)
        C[F > 3] = 0
        S = {'F': F, 'a': a, 'b': b, 'c': c, 'B': B, 'C': C, 'Faro': np.multiply(F, aro.dot(aro.T)),
             'Caro_': np.multiply(C, aro.T), 'Fcyc': np.multiply(F, cyc.dot(cyc.T)), 'Ccyc_': np.multiply(C, cyc.T),
             'acyc_': np.multiply(a, cyc.T), 'boncyc_': np.multiply(bon, cyc.T), 'bon': bon}
        return S

    # =============================================================================
    # Q is quantum parameter marix
    # =============================================================================
    def QuantumGauss(SI):
        Q = {'Mc': SI['Mc'], 'cn': SI['cn'], 'ncn': SI['ncn'], 'nvn': SI['nvn'], 'nrn': SI['nrn'], 'ntn': SI['ntn'],
             'ecn': SI['ecn'], 'evn': SI['evn'], 'ern': SI['ern'], 'etn': SI['etn'], 'gc': SI['gc'], 'gv': SI['gv'],
             'gR': SI['gR'], 'gt': SI['gt']}
        return Q

    # =============================================================================
    # The distance matrices
    # Da is the adjacent distance matix
    # Db is the interphase distance matix
    # Dc is the jump distance matix
    # DB is the adjacent and interphase distance matix
    # DC is the adjacent, interphase and jump distance matix
    # DF is the full distance matrix
    # D is distance marix
    # =============================================================================
    def DisatanceGauss(SI, S):
        S = copy.deepcopy(S)
        n_atom = SI['n_atom']
        a = S['a']
        b = S['b'] / 2
        c = S['c'] / 3
        F = SIFunction.FullDistance(n_atom, SI['Ddis'])
        a = np.multiply(F, a)
        b = np.multiply(F, b)
        c = np.multiply(F, c)
        B = a + b
        C = a + b + c
        D = {'F': F, 'a': a, 'b': b, 'c': c, 'B': B, 'C': C}
        return D


class PositionPropertyDerivative:
    def properties(P):
        Data = {'wei': P['wei'], 'ele': P['ele'], 'ion': P['ion'], 'rad': P['rad'],
                'noe': P['noe'], 'noe_nes': P['noe'] / P['nes'], 'aro': P['aro'], 'bra': P['bra']}
        DataType = ['P', '']
        return Data, DataType

    def PropertiesNoHydrogen(P_H):
        Data = {'wei': P_H['wei'], 'ele': P_H['ele'], 'ion': P_H['ion'], 'rad': P_H['rad'],
                'noe': P_H['noe'], 'noe_nes': P_H['noe'] / P_H['nes'], 'aro': P_H['aro'], 'bra': P_H['bra']}
        DataType = ['P_H', '']
        return Data, DataType

    def quantum(Q):
        Data = {'Mc': Q['Mc'], 'cn': Q['cn'], 'ncn': Q['ncn'], 'nvn': Q['nvn'], 'nrn': Q['nrn'],
                'ntn': Q['ntn'], 'ecn': Q['ecn'], 'evn': Q['evn'], 'ern': Q['ern'], 'etn': Q['etn'],
                'gc': Q['gc'], 'gv': Q['gv'], 'gR': Q['gR'], 'gt': Q['gt']}
        DataType = ['Q', '']
        return Data, DataType

    def QuantumHydrogenToHeavyAtom(Q_H):
        Data = {'Mc': Q_H['Mc'], 'cn': Q_H['cn'], 'ncn': Q_H['ncn'], 'nvn': Q_H['nvn'], 'nrn': Q_H['nrn'],
                'ntn': Q_H['ntn'], 'ecn': Q_H['ecn'], 'evn': Q_H['evn'], 'ern': Q_H['ern'], 'etn': Q_H['etn'],
                'gc': Q_H['gc'], 'gv': Q_H['gv'], 'gR': Q_H['gR'], 'gt': Q_H['gt']}
        DataType = ['Q_H', '']
        return Data, DataType

    def step(S):
        e = math.exp(1)
        Data = {'F': p(S['F'], e), 'a': p(S['a'], e)}
        DataType = ['S', '^e']
        return Data, DataType

    def StepNoHydrogen(S_H):
        e = math.exp(1)
        Data = {'F': p(S_H['F'], e), 'a': p(S_H['a'], e)}
        DataType = ['S_H', '^e']
        return Data, DataType

    def distance(D):
        e = math.exp(1)
        Data = {'F': p(D['F'], e), 'a': p(D['a'], e)}
        DataType = ['D', '^e']
        return Data, DataType

    def DistanceNoHydrogen(D_H):
        e = math.exp(1)
        Data = {'F': p(D_H['F'], e), 'a': p(D_H['a'], e)}
        DataType = ['D_H', '^e']
        return Data, DataType


"""
QSPR
class PositionPropertyDerivative:
      def properties(P):
          Data={'wei': P['wei'], 'ele': P['ele'], 'ion': P['ion'], 'rad': P['rad'],
                'noe': P['noe'], 'noe_nes': P['noe']/P['nes'], 'bra': P['bra']} 
          DataType=['P','']
          return Data,DataType
      def PropertiesNoHydrogen(P_H):
          Data={'wei': P_H['wei'], 'ele': P_H['ele'], 'ion': P_H['ion'], 'rad': P_H['rad'], 
                'noe': P_H['noe'], 'noe_nes': P_H['noe']/P_H['nes'], 'bra': P_H['bra']}
          DataType=['P_H','']
          return Data,DataType
      def quantum(Q):
          Data={'Mc': Q['Mc'],'cn': Q['cn'],'ncn': Q['ncn'],'nvn': Q['nvn'],'nrn': Q['nrn'],
                'ntn': Q['ntn'],'ecn': Q['ecn'],'evn': Q['evn'],'ern': Q['ern'],'etn': Q['etn'],
                'gc': Q['gc'],'gv': Q['gv'],'gR': Q['gR'],'gt': Q['gt']}
          DataType=['Q','']
          return Data,DataType
      def QuantumHydrogenToHeavyAtom(Q_H):
          Data={'Mc': Q_H['Mc'],'cn': Q_H['cn'],'ncn': Q_H['ncn'],'nvn': Q_H['nvn'],'nrn': Q_H['nrn'],
                'ntn': Q_H['ntn'],'ecn': Q_H['ecn'],'evn': Q_H['evn'],'ern': Q_H['ern'],'etn': Q_H['etn'],
                'gc': Q_H['gc'],'gv': Q_H['gv'],'gR': Q_H['gR'],'gt': Q_H['gt']}
          DataType=['Q_H','']
          return Data,DataType
      def step(S):
          e=math.exp(1)
          Data={'F': p(S['F'],e),'a': p(S['a'],e),'B': p(S['B'],e),'C': p(S['C'],e),'bon': p(S['bon'],e),'Caro_': p(S['Caro_'],e),'Ccyc_': p(S['Ccyc_'],e),'boncyc_': p(S['boncyc_'],e)} 
          DataType=['S','^e']
          return Data,DataType
      def StepNoHydrogen(S_H):
          e=math.exp(1)
          Data={'F': p(S_H['F'],e),'a': p(S_H['a'],e),'B': p(S_H['B'],e),'C': p(S_H['C'],e),'bon': p(S_H['bon'],e),'Caro_': p(S_H['Caro_'],e),'Ccyc_': p(S_H['Ccyc_'],e),'boncyc_': p(S_H['boncyc_'],e)} 
          DataType=['S_H','^e']
          return Data,DataType
      def distance(D):
          e=math.exp(1)
          Data={'F': p(D['F'],e),'a': p(D['a'],e),'B': p(D['B'],e),'C': p(D['C'],e)} 
          DataType=['D','^e']
          return Data,DataType
      def DistanceNoHydrogen(D_H):
          e=math.exp(1)
          Data={'F': p(D_H['F'],e),'a': p(D_H['a'],e),'B': p(D_H['B'],e),'C': p(D_H['C'],e)} 
          DataType=['D_H','^e']
          return Data,DataType
 """
