from urllib.request import urlopen
from io import BytesIO
from zipfile import ZipFile
from matplotlib.pyplot import xlabel
import networkx as nx
from GMLA_2 import *
from PTVA_algo_final2 import *
from TSSI_complete import *

G_football, _ = load("football")
G_football.nodes()

# with open("facebook.csv", "r") as data:
#     next(data, None)
#     G_facebook = nx.parse_edgelist(
#         data, delimiter=",", create_using=nx.Graph, nodetype=int
#     )

d_url = "http://www-personal.umich.edu/~mejn/netdata/dolphins.zip"
with urlopen(d_url) as sock, ZipFile(BytesIO(sock.read())) as zf:
    gml = zf.read("dolphins.gml").decode().split("\n")[1:]
    G_dolphin = nx.parse_gml(gml)

adj_noun = nx.read_gml("adjnoun.gml")

 # Counter -> dict {item: freq}
from collections import Counter, defaultdict


#For Algorithms using Partial Infection - GFHF, LGC
def gen_data_partial(algo, dataset, n_snaps=10):

    df_dist_err = defaultdict(list)
    df_time = defaultdict(list)
    err_freq = []

    for G_name, G in dataset.items():
        src = rand.choice(list(G.nodes()))
        N = G.number_of_nodes()

        known_dicts = simulatePartialInfection(
            G, src, threshold=0.3, sampling=0.2, n_snaps=n_snaps
        )

        score_time = [
            [alg(G, label, src) for label in known_dicts] for alg in algo.values()
        ]

        """
        alg_data = [alg1 [], alg2 [] ...]
        alg1 [] -> [(dist_err), (snapshot_time)]
        snapshot_score_dict = {node: score}
        """

        alg_data = [list(zip(*res)) for res in score_time]
        
        dict_freq = defaultdict(list)
        
        freq_hops = [Counter(err) for err, _ in alg_data]
        avg_dist_err = [stats.mean(err) for err, _ in alg_data]
        avg_time = [stats.mean(time) for _, time in alg_data]

        for alg_name, de, time, freq in zip(algo.keys(), avg_dist_err, avg_time, freq_hops):
            df_dist_err[alg_name].append(de)
            df_time[alg_name].append(time)
            dict_freq[alg_name] = [freq[i] if i in freq else 0 for i in range(4)]
        
        df = pd.DataFrame(dict_freq, columns=algo.keys(), index=list(range(4)))
        err_freq.append(df)
    
    df_de = pd.DataFrame(df_dist_err, columns=algo.keys(), index=dataset.keys())
    df_time = pd.DataFrame(df_time, columns=algo.keys(), index=dataset.keys())
                        
    return df_de, df_time, err_freq
    

#For Algorithms using Complete Observation - GMLA, PTVA
def gen_data_complete(algo, dataset, iterations):
    df_dist_err = defaultdict(list)
    df_time = defaultdict(list)
    err_freq = []

    for G_name, G in dataset.items():
        N = G.number_of_nodes()
        score_time = [alg(G, G_name, iterations) for alg in algo.values()]
        alg_data = [list(zip(*res)) for res in score_time]

        dict_freq = defaultdict(list)

        freq_hops = [Counter(err) for err, _ in alg_data]
        avg_dist_err = [stats.mean(err) for err, _ in alg_data]
        avg_time = [stats.mean(time) for _, time in alg_data]

        for alg_name, de, time, freq in zip(
            algo.keys(), avg_dist_err, avg_time, freq_hops
        ):
            df_dist_err[alg_name].append(de)
            df_time[alg_name].append(time)
            dict_freq[alg_name] = [freq[i] if i in freq else 0 for i in range(4)]

        df = pd.DataFrame(dict_freq, columns=algo.keys(), index=list(range(4)))
        err_freq.append(df)

    df_de = pd.DataFrame(df_dist_err, columns=algo.keys(), index=dataset.keys())
    df_time = pd.DataFrame(df_time, columns=algo.keys(), index=dataset.keys())

    return df_de, df_time, err_freq
    


# Datasets and algorithms
comp_algo = {"GMLA": GMLA, "PTVA": PTVA_algo}
par_algo = {"GFHF": TSSI_GFHF, "LGC": TSSI_LGC}

dataset = {
        # "Karate": nx.karate_club_graph(),
        "Football": G_football,
        # "Facebook": G_facebook,
        # "Dolphin": G_dolphin,
        # "albert barabasi": nx.barabasi_albert_graph(n=100, m=5),
        # "erdos renyi": nx.erdos_renyi_graph(n=100, p=0.2),
        # "Adjective Noun": adj_noun
}

de_comp, time_comp, freq_comp = gen_data_complete(comp_algo, dataset, 30)
de_par, time_par, err_freq_par = gen_data_partial(par_algo, dataset, 30)

# Plotting
de = pd.concat([de_par, de_comp], axis=1)
de.plot.bar(title="Distance Error", xlabel="Datasets", ylabel="distance error")
plt.show()

time = pd.concat([time_par, time_comp], axis=1)
time.plot.bar(title="Time of execution", xlabel="Datasets", ylabel="time (in ms)")
plt.show()

freq = [pd.concat([p, c], axis=1) for p, c in zip(err_freq_par, freq_comp)]
for f, title in zip(freq, dataset.keys()):
    f.plot.bar(title=title, xlabel="Number of hops", ylabel="frequency")
plt.show()