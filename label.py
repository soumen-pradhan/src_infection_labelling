from networkx.linalg.graphmatrix import adjacency_matrix
from scipy.sparse import dia_matrix
from numpy import copy, full
from numpy.linalg import norm
from random import choice, random

import networkx as nx
import numpy as np
import sys

from pprint import pprint


def simulateInfection(G, src, model, lamda, runtime, threshold):
    nodes = list(G.nodes())
    infected_nodes = {src}

    for i in range(runtime):
        temp_infected = infected_nodes.copy()

        for node in infected_nodes:
            for neighbour in nx.all_neighbors(G, node):
                if random() < lamda:
                    temp_infected.add(neighbour)

        infected_nodes = temp_infected

        if len(infected_nodes) > threshold:
            break

    y = np.full(G.number_of_nodes(), -1)
    for infNode in infected_nodes:
        y[nodes.index(infNode)] = 1

    return y


# Page 4
def labelRankingScore(G, y, alpha):
    W = adjacency_matrix(G)
    diag_elem = W.sum(axis=1).A1 ** (-0.5)

    inv_sqrt_D = dia_matrix((diag_elem, [0]), shape=W.get_shape())
    S = inv_sqrt_D @ W @ inv_sqrt_D

    f = copy(y)

    while True:
        f_ = alpha * S @ f + (1 - alpha) * y

        if norm(f - f_) < 0.001 * G.number_of_nodes():
            break

        f = f_

    return f


def BLRSI(G, y, a):
    f = labelRankingScore(G, y, a)
    s = max(enumerate(f), key=(lambda x: x[1]))

    return s

# Page 9


def GFGH(G, y):
    W = adjacency_matrix(G)
    diag_elem = W.sum(axis=1).A1

    D = dia_matrix((diag_elem, [0]), shape=W.get_shape())

    P = (D.A ** -1) @ W
    # F

    while true:  # F is NOT convergent
        pass
        # F = P * F

        # F_l = Y_l

    # O = max(F)
    # return O

# Page 10


def LGC(G, Y, a):
    W = adjacency_matrix(G)
    diag_elem = W.sum(axis=1).A1

    D = dia_matrix((diag_elem, [0]), shape=W.get_shape())
    inv_sqrt_D = D.A ** (1 / 2)

    S = inv_sqrt_D @ W @ inv_sqrt_D
    # F

    while true:  # F is NOT convergent
        pass
        # F = a * (S * F) + (1-a) * Y

    # O = max(F)
    # return O


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
    np.set_printoptions(threshold=sys.maxsize)
    # propModels = Enum('SI', 'SIR')

    karate = nx.karate_club_graph()

    src = choice(list(karate.nodes()))
    print(f'src: {src}')
    y = simulateInfection(karate, src, model='SI', lamda=0.3,
                          runtime=1000, threshold=karate.number_of_nodes() * 0.3)

    f = labelRankingScore(karate, y, 0.2)
    s = sorted(enumerate(f), key=lambda x: x[1], reverse=True)

    pprint(list(enumerate(y)))
    pprint(list(enumerate(f)))


if __name__ == '__main__':
    main()
