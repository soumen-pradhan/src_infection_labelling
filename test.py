import numpy as np
import networkx as nx
import scipy as sp
from scipy import stats
from scipy.linalg import fractional_matrix_power as fmp
import random

import sys
from pprint import pprint


def main():
    np.set_printoptions(threshold=sys.maxsize)


    # # networkx.classes.graph.Graph
    karate = nx.generators.social.karate_club_graph()

    N = karate.number_of_nodes()
    W = nx.adjacency_matrix(karate)

    # D = np.zeros((N, N))
    # for i in range(0, N):
    #     D[i][i] = (sum(W[i])) ** (-0.5)
    # S = np.dot(np.dot(D, W), D)
    
    diag_elem = 1 / W.sum(axis=1).A1
    D = sp.sparse.dia_matrix((diag_elem, [0]), shape=W.get_shape())

    P = D @ W

    f = np.zeros((N, 2))
    
    

    print(f)

    # A, _ = g2m.graph2matrix(G, weighted=False)
    # Asum = A.sum(axis=1)
    # P = A / Asum

    # adj = nx.adjacency_matrix(karate).A
    # print(adj)
    # lap = nx.laplacian_matrix(karate, weight=None).A
    # print(lap)
    # y = np.random.choice([-1, 1], karate.number_of_nodes())
    # f = np.copy(y)

    # print(y)

    # # scipy.sparse.csr.csr_matrix
    # adj_mat = nx.linalg.graphmatrix.adjacency_matrix(karate)

    # # rvs = stats.poisson(25, loc=10).rvs
    # # mat = sp.sparse.random(8, 8, density=0.3, format='csr',
    # #                        random_state=42, data_rvs=rvs)
    # # print(f'{mat.A}\n')
    # diagonals = adj_mat.sum(axis=1).A1

    # diag_mat = sp.sparse.dia_matrix(
    #     (diagonals, [0]), shape=adj_mat.get_shape())

    # inv_sqrt_diag = diag_mat.A ** (1/2)
    # S = inv_sqrt_diag @ adj_mat @ inv_sqrt_diag
    # a = 0.5
    # for i in range(5):
    #     f = [a * sum(S[node, j] * f[node] for j in adj_dict.keys()) + (1-a)*y[node]
    #          for node, adj_dict in karate.adjacency()]

    #     print(max(enumerate(f), key=(lambda x: x[1])))

    # # print(x[0])
    # # print(S)
    # # print(f'\n{S[0, 0]}')

    # # print(adj_mat.todense())

    # # diag_sum = adj_mat.sum(axis=0) # correct
    # # pprint(diag_sum)
    # # pprint(diag_sum.reshape(-1))

    # # D = spdiags(diag_sum, 0, 34, 34)
    # # print(D)

    # G = nx.generators.social.karate_club_graph()

    # nodes = list(G.nodes())

    # x = [nodes[random.randint(0, len(G.nodes()) - 1)]]
    # print([nodes[random.randint(0, len(G.nodes()) - 1)]])


if __name__ == '__main__':
    main()
