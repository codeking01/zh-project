import math
import numpy as np
p=np.power
e=math.exp(1)
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
    Data={'F': S['F'],'a': S['a'],'B': S['B'],'C': S['C'],'bon': S['bon'],'Caro_': S['Caro_'],'Ccyc_': S['Ccyc_'],'boncyc_': S['boncyc_']} 
    DataType=['S','']
    return Data,DataType
def StepNoHydrogen(S_H):
    Data={'F': S_H['F'],'a': S_H['a'],'B': S_H['B'],'C': S_H['C'],'bon': S_H['bon'],'Caro_': S_H['Caro_'],'Ccyc_': S_H['Ccyc_'],'boncyc_': S_H['boncyc_']} 
    DataType=['S_H','']
    return Data,DataType
def distance(D):
    Data={'F': D['F'],'a': D['a'],'B': D['B'],'C': D['C']} 
    DataType=['D','']
    return Data,DataType
def DistanceNoHydrogen(D_H):
    Data={'F': D_H['F'],'a': D_H['a'],'B': D_H['B'],'C': D_H['C']} 
    DataType=['D_H','']
    return Data,DataType
