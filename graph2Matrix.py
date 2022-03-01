import networkx as nx
import numpy as np
import copy

# graph2Matrix returns both the adjacency matrix as well
# as laplacianMatrix for a graph G. It takes 2 inputs, G, and
# weighted(bool) which signifies whether graph is weighted or not.

# Adjacency matrix : It is a matrix A(nxn) where n = number of nodes
# and edge between any two vertices is represented by M(i, j)
# for normal graphs, if edge exists : M(i, j) = 1 else M(i, j) = 0
# for weighted graphs, M(i, j) = k(weight of the edge)

# lapacian Matrix: It is a matrix L(nxn) where n = number of nodes in graph
# In this matrix, the diagonal nodes repesent the degree of i for a pair L(i, i)
# We have, L(i, j) = degree(node i), i==j for non-weighted graph, sum of weights of all its neighbours for a weighted graph
#                  = -1, i != j, but edge exists
#                  = 0, no edge

def graph2matrix(G, weighted=False):
    N = len(G.nodes())
    adjMatrix = np.zeros((N, N))
    
    nodes = list(G.nodes())
    edges = list(G.edges())
    
    for edge in edges:
        i = nodes.index(edge[0])
        j = nodes.index(edge[1])
        if not weighted:
            adjMatrix[i][j], adjMatrix[j][i] = 1, 1
        else:
            adjMatrix[i][j], adjMatrix[j][i] = G.get_edge_data(edge[0], edge[1])['weight'], G.get_edge_data(edge[0], edge[1])['weight']
            
    laplacianMatrix = copy.deepcopy(-adjMatrix)
    for i in range(0, N):
        if weighted:
            strength = 0
            for n in G.neighbors(nodes[i]):
                strength += G.get_edge_data(nodes[i], n)['weight']
            laplacianMatrix[i][i] = strength
        else:
            laplacianMatrix[i][i] = G.degrees(nodes[i])
            
    return adjMatrix, laplacianMatrix

#Updated 
