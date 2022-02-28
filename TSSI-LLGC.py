import networkx as nx
import graph2Matrix as g2m
import numpy as np
import random
import copy
import math
import time

def initAll(f, nodes, initIdx, infNodes):
    for i in initIdx:
        if nodes[i] in infNodes:
            f[i][0] = 0
            f[i][1] = 1
        else:
            f[i][0] = 1
            f[i][1] = 0

def LLGC(G, IG, initIdx, alpha):
    # phase 1 : recover infection subgraph
    nodes = list(G.nodes())
    N = len(nodes)
    infNodes = list(IG.nodes())

    W, LL = g2m.graph2matrix(G, weighted=False)
    D = np.zeros((N, N))
    for i in range(0, N):
        D[i][i] = (sum(W[i])) ** (-0.5)
    S = np.dot(np.dot(D, W), D)

    F = np.zeros((N, 2))
    initAll(F, nodes, initIdx, infNodes)
    Y = F

    while True:
        Fnew = alpha * S.dot(F) + (1 - alpha) * Y
        if sum(sum(abs(F - Fnew))) < 0.0001 * N:
            break
        F = Fnew

    detectedInfNodes = []
    for i in range(0, N):
        if F[i][0] <= F[i][1]:
            detectedInfNodes.append(nodes[i])
            
     #phase 2 : single source detection
    MY = np.zeros(N)
    for i in range(0, N):
        MY[i] = -1
    for detectedInfNode in detectedInfNodes:
        i = nodes.index(detectedInfNode)
        MY[i] = 1

    MY = MY.T
    p = MY

    while True:
        pnew = alpha * S.dot(p) + (1 - alpha) * MY
        if sum(abs(p - pnew)) < 0.0001 * N:
            break
        p = pnew

    maxp = -10000
    detectedSource = nodes[random.randint(0, len(nodes) - 1)]
    for i in range(0, N):
        if p[i] > maxp:
            maxp = p[i]
            detectedSource = nodes[i]

    return detectedSource