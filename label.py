from networkx import adjacency_matrix, all_neighbors, karate_club_graph
from scipy.sparse import dia_matrix
from numpy import copy, full, zeros, array, set_printoptions
from numpy.linalg import norm
from random import choice, random, sample

# import networkx as nx
# import numpy as np
# import sys
from sys import maxsize
import random as rnd

from pprint import pprint


def simulateInfection(G, src, model, lamda, runtime, threshold):
    N = G.number_of_nodes()
    nodes = list(G.nodes())
    infected_nodes = {src}

    for i in range(runtime):
        temp_infected = infected_nodes.copy()

        for node in infected_nodes:
            for neighbour in all_neighbors(G, node):
                if random() < lamda:
                    temp_infected.add(neighbour)

        infected_nodes = temp_infected

        if len(infected_nodes) > threshold:
            break

    y = full(N, -1)
    for infNode in infected_nodes:
        y[nodes.index(infNode)] = 1

    return y


def simulatePartialInfection(G, src, model, lamda, runtime, threshold, sampling):
    nodes = list(G.nodes())
    infected_nodes = {src}

    for i in range(runtime):
        temp_infected = infected_nodes.copy()

        for node in infected_nodes:
            for neighbour in all_neighbors(G, node):
                if random() < lamda:
                    temp_infected.add(neighbour)

        infected_nodes = temp_infected

        if len(infected_nodes) > threshold:
            break

    snapshot = sample(nodes, int(sampling * len(nodes)))
    known_dict = {
        'infected': [],
        'safe': []
    }

    for node in snapshot:
        if node in infected_nodes:
            known_dict['infected'].append(node)
        else:
            known_dict['safe'].append(node)

    return known_dict


# Page 4
def labelRankingScore(G, y, alpha):
    N = G.number_of_nodes()

    W = adjacency_matrix(G)
    diag_elem = W.sum(axis=1).A1 ** (-0.5)

    inv_sqrt_D = dia_matrix((diag_elem, [0]), shape=W.get_shape())
    S = inv_sqrt_D @ W @ inv_sqrt_D

    f = copy(y)

    while True:
        f_ = alpha * S @ f + (1 - alpha) * y

        if norm(f - f_) < 0.001 * N: # Convergence Criteria
            break

        f = f_

    return f


def BLRSI(G, y, a):
    f = labelRankingScore(G, y, a)
    s = max(enumerate(f), key=(lambda x: x[1]))

    return s

# Page 9

# knownDict = {
#   infected: [nodes],
#   safe: [nodes]
# }

def resetF(F, known_dict):
    for node in known_dict['safe']:
        F[node][0] = 1
        F[node][1] = 0

    for node in known_dict['infected']:
        F[node][0] = 0
        F[node][1] = 1

def GFGH(G, known_dict):

    N = G.number_of_nodes()
    W = adjacency_matrix(G)

    diag_elem = 1 / W.sum(axis=1).A1
    inv_D = dia_matrix((diag_elem, [0]), shape=W.get_shape())

    P = inv_D @ W
    F = zeros((N, 2))

    resetF(F, known_dict)
    prev_diff = 0

    while True:
        F_ = P @ F
        curr_diff = sum(sum(abs(F - F_))) # Convergence Criteria

        resetF(F, known_dict)

        if curr_diff < 0.0001 * N:
            break

        F = F_

    O = array([1 if f[1] > f[0] else -1 for f in F])
    return O

# Page 10


def LGC(G, known_dict, alpha):

    N = G.number_of_nodes()
    W = adjacency_matrix(G)

    diag_elem = W.sum(axis=1).A1 ** (-0.5)
    inv_sqrt_D = dia_matrix((diag_elem, [0]), shape=W.get_shape())

    S = inv_sqrt_D @ W @ inv_sqrt_D
    F = zeros((N, 2))

    resetF(F, known_dict)
    Y = copy(F)

    while True:
        F_ = alpha * S @ F + (1 - alpha) * Y

        if sum(sum(abs(F - F_))) < 0.0001 * N:
            break

        F = F_

    O = array([1 if f[1] > f[0] else -1 for f in F])
    return O


# Ultimate Function
def TSSI_GFGH(G, Y, a):
    O = GFGH(G, Y)
    # label vector y from O
    s = BLRSI(G, y, a)
    return s


def TSSI_LGC(F, Y, a1, a2):
    O = LGC(G, Y, a1)
    # label vector y from O
    s = BLRSI(G, y, a2)
    return s


def main():
    set_printoptions(threshold=maxsize)
    # propModels = Enum('SI', 'SIR')

    karate = karate_club_graph()
    N = karate.number_of_nodes()

    src = choice(list(karate.nodes()))

    print(f'src: {src}')
    
    y = simulateInfection(karate, src, model='SI', lamda=0.3,
                          runtime=1000, threshold=N * 0.3)

    f1 = labelRankingScore(karate, y, 0.2)
    src1 = sorted(enumerate(f1), key=lambda x: x[1], reverse=True)

    pprint(src1)

    known_dict = simulatePartialInfection(
        karate, src, model='SI', lamda=0.3, runtime=1000, threshold=N * 0.3, sampling=0.75)

    O1 = GFGH(karate, known_dict)
    # pprint(O1)
    O2 = LGC(karate, known_dict, 0.3)

    f2 = labelRankingScore(karate, O1, 0.3)
    src2 = sorted(enumerate(f2), key=lambda x: x[1], reverse=True)

    f3 = labelRankingScore(karate, O2, 0.3)
    src3 = sorted(enumerate(f3), key=lambda x: x[1], reverse=True)

    pprint(src2)
    pprint(src2)


if __name__ == '__main__':
    main()
