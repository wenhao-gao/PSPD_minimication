from minimization.minimization_options import *
import argparse
import numpy as np


if __name__ == '__main__':
    # command parsing
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-i", "--pdb", help='Initial pdb file'
    )

    args = parser.parse_args()

    # initialize pyrosetta
    init()
    pose = pose_from_pdb(args.pdb)
    scorefxn = get_fa_scorefxn()
    starting_pose = pose.clone()
        
    method = 'adam'
    alpha = 10
    rho_1 = 0.99
    rho_2 = 0.99
    parameters = str(alpha) + str(rho_1) + str(rho_2)
    
    t1 = time.time()
    pose, energy, i = adam(pose, scorefxn, alpha_=alpha, tol=0.001, max_iter=10000, rho1=rho_1, rho2=rho_2)
    dt = time.time() - t1
    
    fig_name = args.pdb.split('.')[0] + '_' + method + '_' + parameters + '_min.png'
    plt.plot(list(range(len(energy))), energy)
    plt.grid()
    plt.savefig(fig_name, quality=50)

    pdb_name = args.pdb.split('.')[0] + '_' + method + '_' + parameters + '_min.pdb'
    dump_pdb(pose, pdb_name)

    energy_name = args.pdb.split('.')[0] + '_' + method + '_' + parameters + '_min.npy'
    np.save(energy_name, np.array(energy))

    file_name = args.pdb.split('.')[0] + '_' + method + '_' + parameters + '_min.txt'
    with open(file_name, 'wt') as f:
        print('The start energy is: %.3f' % scorefxn(starting_pose), file=f)
        print('Final energy is: %.3f' % energy[-1], file=f)
        print('Iteration round used: %i' % i, file=f)
        print('Total time used: %.3f seconds' % dt, file=f)
        print('Average time per iteration: %.8f seconds' % (float(dt)/float(i)), file=f)
