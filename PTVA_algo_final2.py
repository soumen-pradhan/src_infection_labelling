import math
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import ndlib.models.ModelConfig as mc
import ndlib.models.epidemics as ep
from matplotlib.lines import Line2D
import time

def inv(m):
    a, b = m.shape
    if a != b:
        raise ValueError("Only square matrices are invertible.")
    i = np.eye(a, a)
    return np.linalg.lstsq(m, i)[0]

def load(filename):
    df = pd.read_csv(filename + '.csv')
    Graphtype = nx.Graph()
    G = nx.from_pandas_edgelist(df, source='Source', target='Target', create_using=Graphtype)
    return G, len(G)


def infect_graph(g, filename):
    """
    Function to infect the graph using SI model.
    Parameters:
      g: Graph
    Returns:
      G : Infected graph
      t : Time of diffusion of each node
    """
    G = g
    # Model selection - diffusion time
    model = ep.SIModel(G)
    nos = 1 / len(G)
    # Model Configuration
    config = mc.Configuration()
    config.add_model_parameter('beta', 0.03)
    config.add_model_parameter("fraction_infected", 1/len(G))
    model.set_initial_status(config)

    # Simulation execution
    iterations = model.iteration_bunch(200)

    diffusionTime = {}
    for i in range(1, len(G)):
        diffusionTime[i] = -1

    for i in iterations:
        for j in i['status']:
            if i['status'][j] == 1:
                diffusionTime[j] = i['iteration']

    nodeColor = []
    source_nodes = []
    for i in G.nodes():
        if iterations[0]["status"][i] == 1:
            nodeColor.append('red')
            source_nodes.append(i)
        else:
            nodeColor.append('blue')
    
    sorted_values = sorted(diffusionTime.values())  # Sort the values
    sorted_dict = {}
    for i in sorted_values:
        for k in diffusionTime.keys():
            if diffusionTime[k] == i:
                sorted_dict[k] = diffusionTime[k]
    return G, sorted_dict, source_nodes


_legends = [Line2D([0], [0], marker='o', color='w', label='Source', markerfacecolor='r', markersize=15),
            Line2D([0], [0], marker='o', color='w', label='Observers', markerfacecolor='g', markersize=15),
            Line2D([0], [0], marker='o', color='w', label='Others', markerfacecolor='b', markersize=15), ]
node_color = []

# helpers ----------------------------------------------------------------
def delayVector(G, t1, observers):
    """Calcs delay with respect to the t1"""

    d = np.zeros(shape=(len(observers) - 1, 1))
    O_length = len(observers)
    for i in range(O_length-1):
        d[i][0] = G.nodes[observers[i + 1]]['time'] - t1
    return d


def nEdges(bfs, s, a):
    """ Returns list of edges from s -> a"""
    try:
        l = list(nx.all_simple_paths(bfs, s, a))[0]
        return l
    except:
        return [0]


def intersection(l1, l2):
    temp = set(l2)
    l = [x for x in l1 if x in temp]
    return len(l) - 1


def mu(bfs, mn, observers, s):
    """Calcs the determinticDelay w.r.t bfs tree."""

    o1 = observers[0]
    length_o1 = len(nEdges(bfs, s, o1))
    mu_k = np.zeros(shape=(len(observers) - 1, 1))
    for k in range(len(observers) - 1):
        mu_k[k][0] = len(nEdges(bfs, s, observers[k + 1])) - length_o1
    return np.dot(mu_k, mn)


def covariance(bfs, sigma2, observers, s):
    """Cals the delayCovariance of the bfs tree."""

    o1 = observers[0]
    delta_k = np.zeros(shape=(len(observers) - 1, len(observers) - 1))
    for k in range(len(observers) - 1):
        for i in range(len(observers) - 1):
            if k == i:
                delta_k[k][i] = len(nEdges(bfs, o1, observers[k + 1]))
            else:
                ne1 = nEdges(bfs, o1, observers[k + 1])
                ne2 = nEdges(bfs, o1, observers[i + 1])
                delta_k[k][i] = intersection(ne1, ne2)
    return sigma2 * delta_k


# Main Algo ----------------------------------------------------------------
def PTVA(G, observers, Ka, sigma2, mn):
    """Main Function for PTVA"""

    # selecting t1
    t1 = G.nodes[observers[0]]['time']
    d = delayVector(G, t1, observers)

    # score
    likelihood = {}

    for s in list(G.nodes()):
        bfs = nx.bfs_tree(G, source=s)
        mu_s = mu(bfs, mn, observers, s)
        delta_s = covariance(bfs, sigma2, observers, s)
        score = np.dot(np.dot(mu_s.T, inv(delta_s)), d - (0.5) * mu_s)
        likelihood[s] = score
    sortedLikelihood = sorted(likelihood.items(), key=lambda x: x[1], reverse=True)
    return sortedLikelihood


def sensor_node_selection(g1):
    sensor_nodes1 = []
    for sen1 in g1.nodes():
        count4 = int(sen1 % 20)
        if count4 == 0:
            sensor_nodes1.append(int(sen1))
            
    return sensor_nodes1


def PTVA_algo(G, filename, iterations):
    # Main Part -----------------------------------------------------------------
    repeat = iterations
    error_distance = []
    total_time = 0
    total_distance = 0
    time_list = []
    result = []
    
    algo = 'ptva'
    # # G, _ = load(filename)
    # G = nx.karate_club_graph()
    _ = len(G)
    length_between_nodes = dict(nx.all_pairs_shortest_path_length(G))
    c = 0
    for i in range(repeat):
        # Infect graph
        G, arrivalTime, sourceNodes = infect_graph(G, filename=filename)
        # Take observers
        start = time.time()
        # k0 = int(np.ceil(np.sqrt(len(G))))
        k0 = max(5, int(len(G.nodes)/20))
        # k0 = 8
        # np.random.seed(k0)
        all_observers = np.random.choice(G.nodes, k0, replace=False).tolist()
        observers = []
        for i in all_observers:
            if i in arrivalTime and arrivalTime[i] != -1:
                observers.append(i)
        
        # observers = sensor_node_selection(G)
        O_length = len(observers)
        
        # mean and variance
        t = []
        for i in observers:
            if arrivalTime[i] != -1:
                t.append(arrivalTime[i])
        mn = np.mean(t)
        sigma2 = np.var(t)
        for i in range(0, O_length):
            G.nodes[observers[i]]['time'] = t[i]

        score = PTVA(G, observers, k0, sigma2, mn)
        scoreList = [score[i][0] for i in range(5)]
        nodes = [list(a)[0] for a in G.nodes(data=True)]
        end = time.time()
        time_1 = end - start
        time_list.append(time_1)
        total_time = time_1 + total_time
        infected_nodes_and_shortest_path = []
        for key, value in length_between_nodes.items():
            if sourceNodes[0] == key:
                infected_nodes_and_shortest_path.append(dict(sorted(value.items(), key=lambda z: z[1])))
                
        for i, dictionary in enumerate(infected_nodes_and_shortest_path):
            for node, distance in dictionary.items():
                if scoreList[0] == node:
                    result.append((distance, time_1, len(list(G.neighbors(scoreList[0])))))
                    error_distance.append(distance)
                    total_distance = total_distance + distance
        c = c+ 1
        print(f'PTVA {scoreList[0]} -> {len(list(G.neighbors(scoreList[0])))}')
        
    return result