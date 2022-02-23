from nx.linalg.graphmatrix import adjacency_matrix
from scipy.sparse import dia_matrix
from numpy import copy


# Page 4
def labelRankingScore(G, y, a):
    W = adjacency_matrix(G)
    diag_elem = W.sum(axis=1).A1

    D = dia_matrix((diag_elem, [0]), shape=W.get_shape())
    inv_sqrt_D = D.A ** (1 / 2)

    S = inv_sqrt_D @ W @ inv_sqrt_D
    f = copy(y)

    while true:  # run till f is NOT convergent
        f = [a * sum(S[node, j] * f[node] for j in adj_dict.keys()) +
             (1-a) * y[node] for node, adj_dict in G.adjacency()]

        # for node, adj_dict in G.adjacency():
        #     f[node] = a * sum(S[node, j] * f[node]
        #                       for j in adj_dict.keys()) + (1-a)*y[node]

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
