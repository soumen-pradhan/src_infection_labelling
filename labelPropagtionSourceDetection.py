#Updated
from os import readlink
import graph2Matrix as g2m
import numpy as np
import random

#Basic Label Propagation Source Detection
def labelPropagtionSourceDetection(G, IG, alpha):
    nodes = list(G.nodes())
    N = len(nodes)
    infNodes = list(IG.nodes)
    
    W, LL = g2m.graph2matrix(G, weighted=False)
    D = np.zeros((N, N))
    for i in range(0, N):
        D[i][i] = (sum(W[i])) ** (-0.5)
    S = np.dot(np.dot(D, W), D) #S = D(-1/2)*W*D(-1/2)
    
    #We can alternately use a different method altogether
    DD = np.zeros((N, N))
    for i in range(0, N):
        DD[i][i] = 1/(sum(W[i]))
    #Variants - Try out with different variants while replacing with S
    P1 = np.dot(DD, W) #1st variant
    P2 = P1.transpose() #Transpose
    
    # Converting from set of infected nodes to an array
    # with infected nodes represented by 1 and non-infected nodes by -1 
    MY = np.zeros(N)
    for i in range(0, N):
        MY[i] = -1
    for infNode in infNodes:
        i = nodes.index(infNode)
        MY[i] = 1
    MY = MY.T #Array
    f = MY
    
    # Calculate till f reaches convergence
    while(True):
        fnew = alpha * S.dot(f) + (1- alpha) * MY
        if np.linalg.norm(f - fnew) < 0.0001 * N:
            break
        f = fnew
    
    maxf = -10000
    detectedSource = nodes[random.randint(0, len(nodes)-1)]
    for i in range(0, N):
        if f[i] > maxf and nodes[i] in infNode:
            maxf = f[i]
            detectedSource = nodes[i]
    return detectedSource
