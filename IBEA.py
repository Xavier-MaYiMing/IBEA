#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/9/19 10:14
# @Author  : Xavier Ma
# @Email   : xavier_mayiming@163.com
# @File    : IBEA.py
# @Statement : Indicator-based evolutionary algorithm
# @Reference : Zitzler E, Künzli S. Indicator-based selection in multiobjective search[C]//International Conference on Parallel Problem Solving from Nature. Berlin, Heidelberg: Springer Berlin Heidelberg, 2004: 832-842.
import numpy as np
import matplotlib.pyplot as plt


def cal_obj(pop, nobj):
    # DTLZ2
    g = np.sum((pop[:, nobj - 1:] - 0.5) ** 2, axis=1)
    temp1 = np.concatenate((np.ones((g.shape[0], 1)), np.cos(pop[:, : nobj - 1] * np.pi / 2)), axis=1)
    temp2 = np.concatenate((np.ones((g.shape[0], 1)), np.sin(pop[:, np.arange(nobj - 2, -1, -1)] * np.pi / 2)), axis=1)
    return np.tile((1 + g).reshape(g.shape[0], 1), (1, nobj)) * np.fliplr(np.cumprod(temp1, axis=1)) * temp2


def cal_fitness(objs, kappa):
    # calculate the fitness
    npop = objs.shape[0]
    objs = (objs - np.min(objs, axis=0)) / (np.max(objs, axis=0) - np.min(objs, axis=0))
    I = np.zeros((npop, npop))
    for i in range(npop):
        for j in range(npop):
            I[i, j] = np.max(objs[i] - objs[j])
    C = np.max(np.abs(I), axis=0)
    fitness = np.sum(-np.exp(-I / np.tile(C, (npop, 1)) / kappa), axis=0) + 1
    return fitness, I, C


def selection(pop, objs, kappa, k=2):
    # binary tournament selection
    (npop, nvar) = pop.shape
    fitness = cal_fitness(objs, kappa)[0]
    nm = npop if npop % 2 == 0 else npop + 1
    mating_pool = np.zeros((nm, nvar))
    for i in range(nm):
        [ind1, ind2] = np.random.choice(npop, k, replace=False)
        if fitness[ind1] > fitness[ind2]:
            mating_pool[i] = pop[ind1]
        else:
            mating_pool[i] = pop[ind2]
    return mating_pool


def crossover(mating_pool, lb, ub, eta_c):
    # simulated binary crossover (SBX)
    (noff, nvar) = mating_pool.shape
    nm = int(noff / 2)
    parent1 = mating_pool[:nm]
    parent2 = mating_pool[nm:]
    beta = np.zeros((nm, nvar))
    mu = np.random.random((nm, nvar))
    flag1 = mu <= 0.5
    flag2 = ~flag1
    beta[flag1] = (2 * mu[flag1]) ** (1 / (eta_c + 1))
    beta[flag2] = (2 - 2 * mu[flag2]) ** (-1 / (eta_c + 1))
    beta = beta * (-1) ** np.random.randint(0, 2, (nm, nvar))
    beta[np.random.random((nm, nvar)) < 0.5] = 1
    beta[np.tile(np.random.random((nm, 1)) > 1, (1, nvar))] = 1
    offspring1 = (parent1 + parent2) / 2 + beta * (parent1 - parent2) / 2
    offspring2 = (parent1 + parent2) / 2 - beta * (parent1 - parent2) / 2
    offspring = np.concatenate((offspring1, offspring2), axis=0)
    offspring = np.min((offspring, np.tile(ub, (noff, 1))), axis=0)
    offspring = np.max((offspring, np.tile(lb, (noff, 1))), axis=0)
    return offspring


def mutation(pop, lb, ub, eta_m):
    # polynomial mutation
    (npop, nvar) = pop.shape
    lb = np.tile(lb, (npop, 1))
    ub = np.tile(ub, (npop, 1))
    site = np.random.random((npop, nvar)) < 1 / nvar
    mu = np.random.random((npop, nvar))
    delta1 = (pop - lb) / (ub - lb)
    delta2 = (ub - pop) / (ub - lb)
    temp = np.logical_and(site, mu <= 0.5)
    pop[temp] += (ub[temp] - lb[temp]) * ((2 * mu[temp] + (1 - 2 * mu[temp]) * (1 - delta1[temp]) ** (eta_m + 1)) ** (1 / (eta_m + 1)) - 1)
    temp = np.logical_and(site, mu > 0.5)
    pop[temp] += (ub[temp] - lb[temp]) * (1 - (2 * (1 - mu[temp]) + 2 * (mu[temp] - 0.5) * (1 - delta2[temp]) ** (eta_m + 1)) ** (1 / (eta_m + 1)))
    pop = np.min((pop, ub), axis=0)
    pop = np.max((pop, lb), axis=0)
    return pop


def environmental_selection(pop, objs, num, kappa):
    # environmental selection
    Next = np.arange(pop.shape[0])
    fitness, I, C = cal_fitness(objs, kappa)
    while Next.shape[0] > num:
        worst = np.argmin(fitness[Next])
        fitness += np.exp(-I[Next[worst]] / C[Next[worst]] / kappa)
        Next = np.delete(Next, worst)
    return pop[Next], objs[Next]


def dominates(obj1, obj2):
    # determine whether obj1 dominates obj2
    sum_less = 0
    for i in range(len(obj1)):
        if obj1[i] > obj2[i]:
            return False
        elif obj1[i] != obj2[i]:
            sum_less += 1
    return sum_less > 0


def main(npop, iter, lb, ub, nobj=3, kappa=0.05, eta_c=20, eta_m=20):
    """
    The main function
    :param npop: population size
    :param iter: iteration number
    :param lb: lower bound
    :param ub: upper bound
    :param nobj: the dimension of objective space (default = 3)
    :param kappa: fitness scaling factor (default = 0.05)
    :param eta_c: spread factor distribution index (default = 20)
    :param eta_m: perturbance factor distribution index (default = 20)
    :return:
    """
    # Step 1. Initialization
    nvar = len(lb)  # the dimension of decision space
    pop = np.random.uniform(lb, ub, (npop, nvar))  # population
    objs = cal_obj(pop, nobj)  # objectives

    # Step 2. The main loop
    for t in range(iter):

        if (t + 1) % 20 == 0:
            print('Iteration: ' + str(t + 1) + ' completed.')

        # Step 2.1. Mating selection + crossover + mutation
        mating_pool = selection(pop, objs, kappa)
        off = crossover(mating_pool, lb, ub, eta_c)
        off = mutation(off, lb, ub, eta_m)
        off_objs = cal_obj(off, nobj)

        # Step 2.2. Environmental selection
        pop, objs = environmental_selection(np.concatenate((pop, off), axis=0), np.concatenate((objs, off_objs), axis=0), npop, kappa)

    # Step 3. Sort the results
    dom = np.full(npop, False)
    for i in range(npop - 1):
        for j in range(i, npop):
            if not dom[i] and dominates(objs[j], objs[i]):
                dom[i] = True
            if not dom[j] and dominates(objs[i], objs[j]):
                dom[j] = True
    pf = objs[~dom]
    ax = plt.figure().add_subplot(111, projection='3d')
    ax.view_init(45, 45)
    x = [o[0] for o in pf]
    y = [o[1] for o in pf]
    z = [o[2] for o in pf]
    ax.scatter(x, y, z, color='red')
    ax.set_xlabel('objective 1')
    ax.set_ylabel('objective 2')
    ax.set_zlabel('objective 3')
    plt.title('The Pareto front of DTLZ2')
    plt.savefig('Pareto front')
    plt.show()


if __name__ == '__main__':
    main(100, 150, np.array([0] * 7), np.array([1] * 7))
