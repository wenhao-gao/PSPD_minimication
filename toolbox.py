#!/bin/python
"""
ToolsBox, small functions that helps protein structural study
The __main__ part can generate score against rmsd plot
"""
from pyrosetta.rosetta.core.io.pdb import dump_pdb
from pyrosetta import *
from pyrosetta.rosetta import *
from pyrosetta.teaching import *
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import os
import numpy as np
#from movement.moves import *
#from pathos.multiprocessing import ProcessingPool as Pool
# import joblib
import time


def read_sequence(file):
    count = 1
    seq = ''
    with open(file, 'rt') as f:
        while True:
            line = f.readline()
            if not line:
                break
            if count == 1:
                temp = line.split('|')[0][1:]
                temp = temp.replace(':', '_')
                target = temp
                count += 1
                continue
            seq = seq + line.strip('\n')
            count += 1

    return target, seq


# print(read_sequence('Fastq/1LMR_A.fasta.txt'))

def pose2pdb(poses, scores, directory, name, num, show, pmm):
    temp = [(poses[i], scores[i]) for i in range(len(scores))]
    temp.sort(key=lambda x:x[1])
    for i in range(num):
        pdb_name = name + '_' + str(i + 1) + '.pdb'
        dump_pdb(
            temp[i][0], os.path.join(directory, pdb_name)
        )
        # show the lowest energy pose in PyMol
        if show:
            pmm.apply(temp[i][0])

    return None


def cheat(native, cent, num):
    pose = Pose()
    pose.assign(native)
    rms = []
    score = []

    if cent:
        to_centroid = SwitchResidueTypeSetMover("centroid")
        to_full_atom = SwitchResidueTypeSetMover("fa_standard")
        sfxn = pyrosetta.create_score_function("score3")
        rms.append(0)
        to_centroid.apply(native)
        score.append(sfxn(native))
        to_full_atom.apply(native)
        while True:
            generate_decoy_random(pose, degree=1)
            r = CA_rmsd(native, pose)
            if r < 1:
                rms.append(r)
                to_centroid.apply(pose)
                score.append(sfxn(pose))
                to_full_atom.apply(pose)
            pose.assign(native)
            if len(score) >= num:
                break
    else:
        sfxn = get_fa_scorefxn()
        rms.append(0)
        score.append(sfxn(native))
        while True:
            generate_decoy_random(pose, degree=1)
            r = CA_rmsd(native, pose)
            if r < 1:
                rms.append(r)
                score.append(sfxn(pose))
            pose.assign(native)
            if len(score) >= num:
                break

    return rms, score


def evaluate(native_pdb, rms, score, name):
    init()
    native = pose_from_pdb(native_pdb)

    c_rms, c_score = cheat(native, cent=True, num=10)

    plt.scatter(c_rms, c_score, label='near-native')
    plt.scatter(rms, score, label='decoy')
    plt.grid()
    plt.legend(loc='best')
    plt.xlabel('RMSD from PDB')
    plt.ylabel('Decoy Score')
    plt.savefig(name + '_low.png', quality=50)
    return None


def get_gradient(pose, sfxn=None):
    if sfxn is None:
        sfxn = get_fa_scorefxn()
    E = sfxn(pose)
    eps = 1
    numerical_derivs = []
    for i in range(len(pose.sequence())):
        temp_pose = pose.clone()
        temp_pose.set_phi(i + 1, pose.phi(i + 1) + eps)
        E_pert = sfxn(temp_pose)
        numerical_derivs.append((E_pert - E) / eps)

    for i in range(len(pose.sequence())):
        temp_pose = pose.clone()
        temp_pose.set_psi(i + 1, pose.psi(i + 1) + eps)
        E_pert = sfxn(temp_pose)
        numerical_derivs.append((E_pert - E) / eps)

    return np.array(numerical_derivs)


def get_phipsi(pose):
    phipsi = []
    for i in range(len(pose.sequence())):
        phipsi.append(pose.phi(i + 1))
    for i in range(len(pose.sequence())):
        phipsi.append(pose.psi(i + 1))

    return np.array(phipsi)


def update_phipsi(pose, phipsi):
    for i in range(len(pose.sequence())):
        pose.set_phi(i+1, phipsi[i])
    for i in range(len(pose.sequence())):
        pose.set_psi(i+1, phipsi[len(pose.sequence()) + i])
    return None


def get_energy(pose_, sfxn, phipsi):
    pose = pose_.clone()
    update_phipsi(pose, phipsi)
    return sfxn(pose)


def get_derivative(pose, sfxn, x, p, alpha):
    pert = 0.001
    alpha_pert = alpha + pert
    E = get_energy(pose, sfxn, x + alpha * p)
    E_pert = get_energy(pose, sfxn, x + alpha_pert * p)
    return (E_pert - E) / pert


if __name__ == '__main__':
    # Generate score vs RMSD plot
    target = '2REB_A'
    r = np.load('output/' + target + '/' + target + '_rmsd.npy')
    s = np.load('output/' + target + '/' + target + '_score.npy')
    pdb = 'native/' + target + '.pdb'
    evaluate(pdb, r, s, target)

    # Test the parallel version of functions
    # init()
    # pose = pose_from_pdb('helix.pdb')
    # sfxn = get_fa_scorefxn()
    #
    # LENGTH = 16
    #
    # t1 = time.time()
    # for i in range(LENGTH):
    #     dE = get_gradient(pose, sfxn)
    # t2 = time.time()
    # print('Time used:')
    # print(t2 - t1)
    #
    # pool = Pool()
    # # poses = [pose] * 4
    # # sfxns = [sfxn] * 4
    # # t1 = time.time()
    # # results = pool.map(get_gradient, poses, sfxns)
    # poses = [1] * 4
    # sfxns = [2] * 4
    # t1 = time.time()
    # results = pool.map(test_work, poses, sfxns)
    # pool.close()
    # print('Paralled:')
    # print("Took {}s".format(time.time() - t1))
    # print(results)



