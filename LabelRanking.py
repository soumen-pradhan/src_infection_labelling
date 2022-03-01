import graph2matrix as g2m
import numpy as np
import random
import heapq

def labelRanking(G, IG, alpha, k):
    nodes = list(G.nodes())
    N = len(nodes)
    infNodes = list(IG.nodes())

    W, LL = g2m.graph2matrix(G, weighted=False)
    D = np.zeros((N, N))
    for i in range(0, N):
        D[i][i] = (sum(W[i])) ** (-0.5)
    S = np.dot(np.dot(D, W), D)

    DD = np.zeros((N, N))
    for i in range(0, N):
        DD[i][i] = 1 / (sum(W[i]))
    #P1 = np.dot(DD, W)
    #P2 = P.transpose()

    MY = np.zeros(N)
    for i in range(0, N):
        MY[i] = -1
    for infNode in infNodes:
        i = nodes.index(infNode)
        MY[i] = 1

    MY = MY.T
    f = MY
    
    #Convergece of matrix
    while True:
        fnew = alpha * S.dot(f) + (1 - alpha) * MY
        if np.linalg.norm(f - fnew) < 0.0001 * N:
            break
        f = fnew

    #Ordering the nodes in terms of their probablity to become the source
    idx = heapq.nlargest(k, range(len(f)), f.take)
    ksources = []
    for i in idx:
        ksources.append(nodes[i])

    return ksources