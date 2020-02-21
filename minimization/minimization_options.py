#!/bin/python
"""
The minimization methods zoo
The __main__ part can generate score against rmsd plot
"""
from toolbox import *
from pyrosetta import *
from rosetta.protocols.loops import get_fa_scorefxn
#import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import time


def vanilla(
        pose_,
        sfxn,
        tol=0.001,
        alpha=1,
        max_iter=10000
):
    """
    The vanilla gradient descent method

    :param pose: *Pose*
        The starting pose of the protein
    :param sfxn: *Score Function*
        The function that gives the score to optimize
    :param tol: *float*
        The tolerance for the gradient to end the iteration
    :param alpha: *float*
        The step size
    :param max_iter: *int*
        The maximum iteration turns

    :return:
    pose: *Pose*
        The conformation after minimization
    energy: *list float*
        The score function along the minimization process
    i: *int*
        The iteration turns used to reach minimum
    """
    pose = pose_.clone()
    converge = False
    energy = [sfxn(pose)]

    for i in range(max_iter):
        # print('Round %i' %i)
        # print('Current Energy is: %.1f' % energy[-1])
        x = get_phipsi(pose)
        de_dx = get_gradient(pose, sfxn)
        p = - de_dx
        step = alpha * p

        converge = all([abs(g) < tol for g in step])
        if converge:
            break

        if len(energy) >= 2:
            de = energy[-1] - energy[-2]
            if abs(de) <= tol:
                converge = True
                break

        new_x = x + step
        update_phipsi(pose, new_x)
        energy.append(sfxn(pose))

    max_iter_reached = i == max_iter - 1

    if converge:
        print("Gradient converged below %s!" % str(tol))
    elif max_iter_reached:
        print("Function has not converged yet.  Maxiter was reached.")
    else:
        print("This... should not print.")

    return pose, energy, i


def conjugate_gradient(
    pose_,
    sfxn,
    tol=0.001,
    alpha=0.1,
    max_iter=1000
):
    """
    The conjugate gradient descent method

    :param pose: *Pose*
        The starting pose of the protein
    :param sfxn: *Score Function*
        The function that gives the score to optimize
    :param tol: *float*
        The tolerance for the gradient to end the iteration
    :param alpha: *float*
        The step size
    :param max_iter: *int*
        The maximum iteration turns

    :return:
    pose: *Pose*
        The conformation after minimization
    energy: *list float*
        The score function along the minimization process
    i: *int*
        The iteration turns used to reach minimum
    """
    pose = pose_.clone()
    converge = False
    energy = []

    x = get_phipsi(pose)
    de_dx = get_gradient(pose, sfxn)
    p = - de_dx
    step = alpha * p
    new_x = x + step
    update_phipsi(pose, new_x)
    energy.append(sfxn(pose))

    for i in range(max_iter):
        # print('Round %i' %i)
        # print('Current Energy is: %.1f' % energy[-1])
        x = get_phipsi(pose)
        prev_de_dx = de_dx
        de_dx = get_gradient(pose, sfxn)
        beta = np.dot(de_dx, de_dx) / np.dot(prev_de_dx, prev_de_dx)
        p = - de_dx + beta * p
        step = alpha * p

        converge = all([abs(g) < tol for g in step])
        if converge:
            break

        if len(energy) >= 2:
            de = energy[-1] - energy[-2]
            if abs(de) <= tol:
                converge = True
                break

        new_x = x + step
        update_phipsi(pose, new_x)
        energy.append(sfxn(pose))

    max_iter_reached = i == max_iter - 1

    if converge:
        print("Gradient converged below %s!" % str(tol))
    elif max_iter_reached:
        print("Function has not converged yet.  Maxiter was reached.")
    else:
        print("This... should not print.")

    return pose, energy, i


def bfgs(
    pose_,
    sfxn,
    tol=0.001,
    alpha=1,
    max_iter=1000
):
    """
    The BFGS (Broyden-Fletcher-Goldfarb-Shanno) method
    A quasi newton method

    Update every step by following math:
        x_t+1 = x_t + alpha * p_t
        p_t = - B^{-1}_t * de_dx
        B^{-1}_{t+1} = B^{-1}_t
            + (1 + \frac{y_t^TB^{-1}_ty_t}{y_t^Ts_t})\frac{s_ts_t^T}{s_t^Ty_t}
            - \frac{s_ty_t^TB^{-1}_t + B^{-1}_ty_ts_t^T}{s_t^Ty_t}

        with notation:
            y_t = de_dx_{t+1} - de_dx_{t}
            s_t = x_{t+1} - x_{t}

    :param pose: *Pose*
        The starting pose of the protein
    :param sfxn: *Score Function*
        The function that gives the score to optimize
    :param tol: *float*
        The tolerance for the gradient to end the iteration
    :param alpha: *float*
        The step size
    :param max_iter: *int*
        The maximum iteration turns

    :return:
    pose: *Pose*
        The conformation after minimization
    energy: *list float*
        The score function along the minimization process
    i: *int*
        The iteration turns used to reach minimum
    """
    pose = pose_.clone()
    converge = False
    energy = []

    x = get_phipsi(pose)
    de_dx = get_gradient(pose, sfxn)
    p = - de_dx
    step = alpha * p
    new_x = x + step
    update_phipsi(pose, new_x)
    energy.append(sfxn(pose))
    b_inv = np.eye(len(x))

    for i in range(max_iter):
        # print('Round %i' %i)
        # print('Current Energy is: %.1f' % energy[-1])
        prev_x = x
        x = get_phipsi(pose)

        s = x - prev_x
        prev_de_dx = de_dx
        de_dx = get_gradient(pose, sfxn)
        y = de_dx - prev_de_dx
        sy = s.dot(y)

        term_a = s.reshape((-1, 1)).dot(s.reshape((1, -1)))
        term_a = ((sy + y.reshape((1, -1)).dot(b_inv.dot(y.reshape((-1, 1))))[0][0]) / (sy)**2) * term_a
        term_b = b_inv.dot(y.reshape((-1, 1))).dot(s.reshape((1, -1))) + \
                 s.reshape((-1, 1)).dot(y.reshape((1, -1)).dot(b_inv))
        term_b = term_b / sy
        b_inv = b_inv +term_a - term_b

        p = - b_inv.dot(de_dx)
        step = alpha * p

        converge = all([abs(g) < tol for g in step])
        if converge:
            break

        if len(energy) >= 2:
            de = energy[-1] - energy[-2]
            if abs(de) <= tol:
                converge = True
                break

        new_x = x + step
        update_phipsi(pose, new_x)
        energy.append(sfxn(pose))

    max_iter_reached = i == max_iter - 1

    if converge:
        print("Gradient converged below %s!" % str(tol))
    elif max_iter_reached:
        print("Function has not converged yet.  Maxiter was reached.")
    else:
        print("This... should not print.")

    return pose, energy, i


def adam(
    pose_,
    sfxn,
    tol=0.001,
    alpha_=0.1,
    max_iter=1000,
    eps=0.000001,
    rho1=0.9,
    rho2=0.999
):
    """
    The Adam(Adaptive Momentum Method)

    Update location with following math:
        x_t+1 = x_t - alpha_t * p_t
        p_{t+1} = (rho_1 * p_t + (1 - rho_1) * de_dx) / (1 - rho_1^t)
        alpha_{t+1} = alpha / (sqrt(p_2_t) + epsilon)
        p_2_{t+1} = (rho_2 * p_t + (1 - rho_2) * (de_dx)**2) / (1 - rho_2^t)

    :param pose: *Pose*
        The starting pose of the protein
    :param sfxn: *Score Function*
        The function that gives the score to optimize
    :param tol: *float*
        The tolerance for the gradient to end the iteration
    :param alpha: *float*
        The step size
    :param max_iter: *int*
        The maximum iteration turns

    :return:
    pose: *Pose*
        The conformation after minimization
    energy: *list float*
        The score function along the minimization process
    i: *int*
        The iteration turns used to reach minimum
    """
    pose = pose_.clone()
    converge = False
    energy = []
    energy.append(sfxn(pose))

    n_dof = 2 * len(pose.sequence())

    p = np.zeros(n_dof)
    p_2 = np.zeros(n_dof)
    rho_1 = rho1
    rho_2 = rho2

    for i_ in range(max_iter):
        # print('Round %i' %i_)
        # print('Current Energy is: %.1f' % energy[-1])
        i = i_ + 1
        x = get_phipsi(pose)
        de_dx = get_gradient(pose, sfxn)

        p = (rho_1 * p + (1 - rho_1) * de_dx) / (1 - rho_1 ** i)
        p_2 = (rho_2 * p_2 + (1 - rho_2) * de_dx ** 2) / (1 - rho_2 ** i)

        alpha = alpha_ / (np.sqrt(p_2) + eps)

        step = - alpha * p

        converge = all([abs(g) < tol for g in step])
        if converge:
            break

        if len(energy) >= 2:
            de = energy[-1] - energy[-2]
            if abs(de) <= tol:
                converge = True
                break

        new_x = x + step
        update_phipsi(pose, new_x)
        energy.append(sfxn(pose))

    max_iter_reached = i == max_iter - 1

    if converge:
        print("Gradient converged below %s!" % str(tol))
    elif max_iter_reached:
        print("Function has not converged yet.  Maxiter was reached.")
    else:
        print("This... should not print.")

    return pose, energy, i


def brent(pose, sfxn, x, p, alpha_min, alpha_max, tol=0.001):
    """
    The Brent method to set up the step length
    :param alpha_min:
    :param alpha_max:
    :param tol:
    :return: alpha
    """
    l = 0.618
    while True:
        d_alpha = alpha_max - alpha_min
        if d_alpha <= tol:
            alpha = alpha_min
            break
        else:
            alpha_1 = alpha_min + (1 - l) * d_alpha
            alpha_2 = alpha_min + l * d_alpha
            e_1 = get_energy(pose, sfxn, x + alpha_1 * p)
            e_2 = get_energy(pose, sfxn, x + alpha_2 * p)
            if e_1 <= e_2:
                alpha_max = alpha_2
            else:
                alpha_min = alpha_1

    return alpha


def pre_brent(pose, sfxn, x, p, step):
    e = sfxn(pose)
    alpha = 0
    while True:
        pre_e = e
        pre_alpha = alpha
        alpha += step
        e = get_energy(pose, sfxn, x + alpha * p)
        if e > pre_e:
            break

    return pre_alpha, alpha


def bfgs_brent(
    pose_,
    sfxn,
    tol=0.001,
    alpha_=1,
    max_iter=1000
):
    """
    The BFGS (Broyden-Fletcher-Goldfarb-Shanno) method
    A quasi newton method

    combined with Brent search method to get step size

    Update every step by following math:
        x_t+1 = x_t + alpha * p_t
        p_t = - B^{-1}_t * de_dx
        B^{-1}_{t+1} = B^{-1}_t
            + (1 + \frac{y_t^TB^{-1}_ty_t}{y_t^Ts_t})\frac{s_ts_t^T}{s_t^Ty_t}
            - \frac{s_ty_t^TB^{-1}_t + B^{-1}_ty_ts_t^T}{s_t^Ty_t}

        with notation:
            y_t = de_dx_{t+1} - de_dx_{t}
            s_t = x_{t+1} - x_{t}

    :param pose: *Pose*
        The starting pose of the protein
    :param sfxn: *Score Function*
        The function that gives the score to optimize
    :param tol: *float*
        The tolerance for the gradient to end the iteration
    :param alpha: *float*
        The step size of brent search
    :param max_iter: *int*
        The maximum iteration turns

    :return:
    pose: *Pose*
        The conformation after minimization
    energy: *list float*
        The score function along the minimization process
    i: *int*
        The iteration turns used to reach minimum
    """
    pose = pose_.clone()
    converge = False
    energy = []

    x = get_phipsi(pose)
    de_dx = get_gradient(pose, sfxn)
    p = - de_dx

    alpha_min, alpha_max = pre_brent(pose, sfxn, x, p, alpha_)
    alpha = brent(pose, sfxn, x, p, alpha_min, alpha_max)

    step = alpha * p
    new_x = x + step
    update_phipsi(pose, new_x)
    energy.append(sfxn(pose))
    b_inv = np.eye(len(x))

    for i in range(max_iter):
        # print('Round %i' %i)
        # print('Current Energy is: %.1f' % energy[-1])
        prev_x = x
        x = get_phipsi(pose)

        s = x - prev_x
        prev_de_dx = de_dx
        de_dx = get_gradient(pose, sfxn)
        y = de_dx - prev_de_dx
        sy = s.dot(y)

        term_a = s.reshape((-1, 1)).dot(s.reshape((1, -1)))
        term_a = ((sy + y.reshape((1, -1)).dot(b_inv.dot(y.reshape((-1, 1))))[0][0]) / (sy)**2) * term_a
        term_b = b_inv.dot(y.reshape((-1, 1))).dot(s.reshape((1, -1))) + \
                 s.reshape((-1, 1)).dot(y.reshape((1, -1)).dot(b_inv))
        term_b = term_b / sy
        b_inv = b_inv +term_a - term_b

        p = - b_inv.dot(de_dx)

        alpha_min, alpha_max = pre_brent(pose, sfxn, x, p, alpha_)
        alpha = brent(pose, sfxn, x, p, alpha_min, alpha_max)

        step = alpha * p

        converge = all([abs(g) < tol for g in step])
        if converge:
            break

        if len(energy) >= 2:
            de = energy[-1] - energy[-2]
            if abs(de) <= tol:
                converge = True
                break

        new_x = x + step
        update_phipsi(pose, new_x)
        energy.append(sfxn(pose))

    max_iter_reached = i == max_iter - 1

    if converge:
        print("Gradient converged below %s!" % str(tol))
    elif max_iter_reached:
        print("Function has not converged yet.  Maxiter was reached.")
    else:
        print("This... should not print.")

    return pose, energy, i


def vanilla_brent(
        pose_,
        sfxn,
        tol=0.001,
        alpha_=1,
        max_iter=10000
):
    """
    The vanilla gradient descent method

    :param pose: *Pose*
        The starting pose of the protein
    :param sfxn: *Score Function*
        The function that gives the score to optimize
    :param tol: *float*
        The tolerance for the gradient to end the iteration
    :param alpha: *float*
        The step size
    :param max_iter: *int*
        The maximum iteration turns

    :return:
    pose: *Pose*
        The conformation after minimization
    energy: *list float*
        The score function along the minimization process
    i: *int*
        The iteration turns used to reach minimum
    """
    pose = pose_.clone()
    converge = False
    energy = []

    for i in range(max_iter):
        x = get_phipsi(pose)
        de_dx = get_gradient(pose, sfxn)
        p = - de_dx

        alpha_min, alpha_max = pre_brent(pose, sfxn, x, p, alpha_)
        alpha = brent(pose, sfxn, x, p, alpha_min, alpha_max)

        step = alpha * p

        converge = all([abs(g) < tol for g in step])
        if converge:
            break

        if len(energy) >= 2:
            de = energy[-1] - energy[-2]
            if abs(de) <= tol:
                converge = True
                break

        new_x = x + step
        update_phipsi(pose, new_x)
        energy.append(sfxn(pose))

    max_iter_reached = i == max_iter - 1

    if converge:
        print("Gradient converged below %s!" % str(tol))
    elif max_iter_reached:
        print("Function has not converged yet.  Maxiter was reached.")
    else:
        print("This... should not print.")

    return pose, energy, i


def conjugate_gradient_brent(
    pose_,
    sfxn,
    tol=0.001,
    alpha_=0.1,
    max_iter=1000
):
    """
    The conjugate gradient descent method

    :param pose: *Pose*
        The starting pose of the protein
    :param sfxn: *Score Function*
        The function that gives the score to optimize
    :param tol: *float*
        The tolerance for the gradient to end the iteration
    :param alpha: *float*
        The step size
    :param max_iter: *int*
        The maximum iteration turns

    :return:
    pose: *Pose*
        The conformation after minimization
    energy: *list float*
        The score function along the minimization process
    i: *int*
        The iteration turns used to reach minimum
    """
    pose = pose_.clone()
    converge = False
    energy = []

    x = get_phipsi(pose)
    de_dx = get_gradient(pose, sfxn)
    p = - de_dx

    alpha_min, alpha_max = pre_brent(pose, sfxn, x, p, alpha_)
    alpha = brent(pose, sfxn, x, p, alpha_min, alpha_max)

    step = alpha * p
    new_x = x + step
    update_phipsi(pose, new_x)
    energy.append(sfxn(pose))

    for i in range(max_iter):
        # print('Round %i' %i)
        # print('Current Energy is: %.1f' % energy[-1])
        x = get_phipsi(pose)
        prev_de_dx = de_dx
        de_dx = get_gradient(pose, sfxn)
        beta = np.dot(de_dx, de_dx) / np.dot(prev_de_dx, prev_de_dx)
        p = - de_dx + beta * p

        alpha_min, alpha_max = pre_brent(pose, sfxn, x, p, alpha_)
        alpha = brent(pose, sfxn, x, p, alpha_min, alpha_max)

        step = alpha * p

        converge = all([abs(g) < tol for g in step])
        if converge:
            break

        if len(energy) >= 2:
            de = energy[-1] - energy[-2]
            if abs(de) <= tol:
                converge = True
                break

        new_x = x + step
        update_phipsi(pose, new_x)
        energy.append(sfxn(pose))

    max_iter_reached = i == max_iter - 1

    if converge:
        print("Gradient converged below %s!" % str(tol))
    elif max_iter_reached:
        print("Function has not converged yet.  Maxiter was reached.")
    else:
        print("This... should not print.")

    return pose, energy, i


def zoom(pose, sfxn, x, p, alpha_1, alpha_2, c1, c2, tol=0.01):
    e_0 = sfxn(pose)
    de_da_0 = get_derivative(pose, sfxn, x, p, 0)
    while True:
        alpha = alpha_1 + 0.5 * (alpha_2 - alpha_1)
        e = get_energy(pose, sfxn, x + alpha * p)
        if (e > e_0 + c1 * de_da_0 * alpha) or (e >= get_energy(pose, sfxn, x + alpha_1 * p)):
            alpha_2 = alpha
        else:
            de_da = get_derivative(pose, sfxn, x, p, alpha)
            if abs(de_da) <= - c2 * de_da_0:
                return alpha
            if de_da * (alpha_2 - alpha_1) >= 0:
                alpha_2 = alpha_1
            alpha_1 = alpha

        if abs(alpha_2 - alpha_1) <= tol:
            return alpha_1


def armijo(pose, sfxn, x, p, c1=0.0001, c2=0.9, alpha_max=2):
    """
    Armijo step size selection method
    The line search with wolfe condition

    P 59-60 in
    .. [nocedal2006a]  Nocedal, J. and Wright, S. (2006),
        Numerical Optimization, 2nd edition, Springer..

    :param pose:
    :param sfxn:
    :param x:
    :param p:
    :param c1:
    :param c2:
    :param alpha_max:
    :return:
    """
    alphas = [0]
    step = 0.01
    i = 1
    e_0 = get_energy(pose, sfxn, x)
    de_da_0 = get_derivative(pose, sfxn, x, p, 0)

    alpha = step
    alphas.append(alpha)
    energy = [0, e_0]

    while True:
        e = get_energy(pose, sfxn, x + alpha * p)
        energy.append(e)
        del energy[0]
        if (e > e_0 + c1 * de_da_0 * alpha) or (energy[-1] > energy[-2]):
            return zoom(pose, sfxn, x, p, alphas[0], alphas[1], c1, c2)

        de_da = get_derivative(pose, sfxn, x, p, alpha)
        if abs(de_da) <= - c2 * de_da_0:
            return alpha
        if de_da_0 >= 0:
            return zoom(pose, sfxn, x, p, alphas[1], alphas[0], c1, c2)

        alpha += step
        if alpha >= alpha_max:
            return alpha_max
        alphas.append(alpha)
        del alphas[0]

        i += 1


def bfgs_armijo(
    pose_,
    sfxn,
    tol=0.001,
    alpha_=1,
    max_iter=1000
):
    """
    The BFGS (Broyden-Fletcher-Goldfarb-Shanno) method
    A quasi newton method

    combined with Armoji method to get step size

    Update every step by following math:
        x_t+1 = x_t + alpha * p_t
        p_t = - B^{-1}_t * de_dx
        B^{-1}_{t+1} = B^{-1}_t
            + (1 + \frac{y_t^TB^{-1}_ty_t}{y_t^Ts_t})\frac{s_ts_t^T}{s_t^Ty_t}
            - \frac{s_ty_t^TB^{-1}_t + B^{-1}_ty_ts_t^T}{s_t^Ty_t}

        with notation:
            y_t = de_dx_{t+1} - de_dx_{t}
            s_t = x_{t+1} - x_{t}

    :param pose: *Pose*
        The starting pose of the protein
    :param sfxn: *Score Function*
        The function that gives the score to optimize
    :param tol: *float*
        The tolerance for the gradient to end the iteration
    :param alpha: *float*
        The step size of brent search
    :param max_iter: *int*
        The maximum iteration turns

    :return:
    pose: *Pose*
        The conformation after minimization
    energy: *list float*
        The score function along the minimization process
    i: *int*
        The iteration turns used to reach minimum
    """
    pose = pose_.clone()
    converge = False
    energy = []

    x = get_phipsi(pose)
    de_dx = get_gradient(pose, sfxn)
    p = - de_dx

    alpha = armijo(pose, sfxn, x, p, alpha_max=alpha_)

    step = alpha * p
    new_x = x + step
    update_phipsi(pose, new_x)
    energy.append(sfxn(pose))
    b_inv = np.eye(len(x))

    for i in range(max_iter):
        # print('Round %i' %i)
        # print('Current Energy is: %.1f' % energy[-1])
        prev_x = x
        x = get_phipsi(pose)

        s = x - prev_x
        prev_de_dx = de_dx
        de_dx = get_gradient(pose, sfxn)
        y = de_dx - prev_de_dx
        sy = s.dot(y)

        term_a = s.reshape((-1, 1)).dot(s.reshape((1, -1)))
        term_a = ((sy + y.reshape((1, -1)).dot(b_inv.dot(y.reshape((-1, 1))))[0][0]) / (sy)**2) * term_a
        term_b = b_inv.dot(y.reshape((-1, 1))).dot(s.reshape((1, -1))) + \
                 s.reshape((-1, 1)).dot(y.reshape((1, -1)).dot(b_inv))
        term_b = term_b / sy
        b_inv = b_inv +term_a - term_b

        p = - b_inv.dot(de_dx)

        alpha = armijo(pose, sfxn, x, p, alpha_max=alpha_)

        step = alpha * p

        converge = all([abs(g) < tol for g in step])
        if converge:
            break

        if len(energy) >= 2:
            de = energy[-1] - energy[-2]
            if abs(de) <= tol:
                converge = True
                break

        new_x = x + step
        update_phipsi(pose, new_x)
        energy.append(sfxn(pose))

    max_iter_reached = i == max_iter - 1

    if converge:
        print("Gradient converged below %s!" % str(tol))
    elif max_iter_reached:
        print("Function has not converged yet.  Maxiter was reached.")
    else:
        print("This... should not print.")

    return pose, energy, i


def vanilla_armijo(
        pose_,
        sfxn,
        tol=0.001,
        alpha_=1,
        max_iter=10000
):
    """
    The vanilla gradient descent method

    :param pose: *Pose*
        The starting pose of the protein
    :param sfxn: *Score Function*
        The function that gives the score to optimize
    :param tol: *float*
        The tolerance for the gradient to end the iteration
    :param alpha: *float*
        The step size
    :param max_iter: *int*
        The maximum iteration turns

    :return:
    pose: *Pose*
        The conformation after minimization
    energy: *list float*
        The score function along the minimization process
    i: *int*
        The iteration turns used to reach minimum
    """
    pose = pose_.clone()
    converge = False
    energy = []
    energy.append(sfxn(pose))

    for i in range(max_iter):
        # print('Round %i' % i)
        # print('Current Energy is: %.1f' % energy[-1])
        x = get_phipsi(pose)
        de_dx = get_gradient(pose, sfxn)
        p = - de_dx

        alpha = armijo(pose, sfxn, x, p, alpha_max=alpha_)

        step = alpha * p

        converge = all([abs(g) < tol for g in step])
        if converge:
            break

        if len(energy) >= 2:
            de = energy[-1] - energy[-2]
            if abs(de) <= tol:
                converge = True
                break

        new_x = x + step
        update_phipsi(pose, new_x)
        energy.append(sfxn(pose))

    max_iter_reached = i == max_iter - 1

    if converge:
        print("Gradient converged below %s!" % str(tol))
    elif max_iter_reached:
        print("Function has not converged yet.  Maxiter was reached.")
    else:
        print("This... should not print.")

    return pose, energy, i


def conjugate_gradient_armijo(
    pose_,
    sfxn,
    tol=0.001,
    alpha_=0.1,
    max_iter=1000
):
    """
    The conjugate gradient descent method

    :param pose: *Pose*
        The starting pose of the protein
    :param sfxn: *Score Function*
        The function that gives the score to optimize
    :param tol: *float*
        The tolerance for the gradient to end the iteration
    :param alpha: *float*
        The step size
    :param max_iter: *int*
        The maximum iteration turns

    :return:
    pose: *Pose*
        The conformation after minimization
    energy: *list float*
        The score function along the minimization process
    i: *int*
        The iteration turns used to reach minimum
    """
    pose = pose_.clone()
    converge = False
    energy = []

    x = get_phipsi(pose)
    de_dx = get_gradient(pose, sfxn)
    p = - de_dx

    alpha = armijo(pose, sfxn, x, p, alpha_max=alpha_)

    step = alpha * p
    new_x = x + step
    update_phipsi(pose, new_x)
    energy.append(sfxn(pose))

    for i in range(max_iter):
        # print('Round %i' %i)
        # print('Current Energy is: %.1f' % energy[-1])
        x = get_phipsi(pose)
        prev_de_dx = de_dx
        de_dx = get_gradient(pose, sfxn)
        beta = np.dot(de_dx, de_dx) / np.dot(prev_de_dx, prev_de_dx)
        p = - de_dx + beta * p

        alpha = armijo(pose, sfxn, x, p, alpha_max=alpha_)

        step = alpha * p

        converge = all([abs(g) < tol for g in step])
        if converge:
            break

        if len(energy) >= 2:
            de = energy[-1] - energy[-2]
            if abs(de) <= tol:
                converge = True
                break

        new_x = x + step
        update_phipsi(pose, new_x)
        energy.append(sfxn(pose))

    max_iter_reached = i == max_iter - 1

    if converge:
        print("Gradient converged below %s!" % str(tol))
    elif max_iter_reached:
        print("Function has not converged yet.  Maxiter was reached.")
    else:
        print("This... should not print.")

    return pose, energy, i


if __name__ == '__main__':
    init()
    # helix = pose_from_pdb('../helix.pdb')
    helix = pose_from_sequence('A' * 10)
    scorefxn = get_fa_scorefxn()

    print('The start energy is: %.3f' % scorefxn(helix))
    t1 = time.time()

    # pose, energy, i = vanilla_armijo(helix, scorefxn)
    pose, energy, i = conjugate_gradient_brent(helix, scorefxn)
    # pose, energy, i = bfgs_armijo(helix, scorefxn)
    # pose, energy, i = adam(helix, scorefxn)

    print('Iteration round used: %i' % i)
    print('Final energy is: %.3f' % energy[-1])
    print('Total time used: %.1f' % (time.time() - t1))

    plt.plot(list(range(len(energy))), energy)
    plt.grid()
    plt.savefig('test.png', quality=50)

    # print(brent(-3, 2, lambda x: x**2))
