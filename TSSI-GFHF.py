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

def GFHF(G, IG, initIdx, alpha):
    #stage 1
    nodes = list(G.nodes())
    N = len(nodes)
    infNodes = list(IG.nodes())

    A, _ = g2m.graph2matrix(G, weighted=False)
    Asum = A.sum(axis=1)
    P = A / Asum
    f = np.zeros((N, 2))

    initAll(f, nodes, initIdx, infNodes)
    while True:
        fnew = np.dot(P, f)
        initAll(fnew, nodes, initIdx, infNodes)
        if sum(sum(abs(f - fnew))) < 0.0001 * N:
            break
        f = fnew

    detectedInfNodes = []

    for i in range(0, N):
        if f[i][1] > f[i][0]:
            detectedInfNodes.append(nodes[i])

    D = np.zeros((N, N))
    for i in range(0, N):
        D[i][i] = (sum(A[i])) ** (-0.5)
    S = np.dot(np.dot(D, A), D)

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

#Updated
