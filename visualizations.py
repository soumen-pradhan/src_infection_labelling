from typing import NoReturn
from urllib.request import urlopen
from io import BytesIO
from zipfile import ZipFile
from matplotlib.pyplot import xlabel
import networkx as nx
from GMLA import *
from PTVA import *
from TSSI_GFHF_LGC import *
import seaborn as sns
from datetime import date, datetime

 # Counter -> dict {item: freq}
from collections import Counter, defaultdict

G_football, _ = load("datasets/football")
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

adj_noun = nx.read_gml("datasets/adjnoun.gml")


#For Algorithms using Partial Infection - GFHF, LGC
def gen_data_partial(algo, dataset, n_snaps=10):
    df_dist_err = defaultdict(list)
    df_time = defaultdict(list)
    err_freq = []
    df_candidates = []
    
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
        alg1 [] -> [(dist_err), (snapshot_time), (no_of_suspected_candidates)]
        snapshot_score_dict = {node: score}
        """
        alg_data = [list(zip(*res)) for res in score_time]
        
        dict_freq = defaultdict(list)
        dict_cand = defaultdict(list)
        
        freq_hops = [Counter(err) for err, _, _ in alg_data]
        avg_dist_err = [stats.mean(err) for err, _, _ in alg_data]
        avg_time = [stats.mean(time) for _, time, _ in alg_data]
        sus_cand = [candidates for _, _, candidates in alg_data]
        
        for alg_name, de, time, freq, cand in zip(
            algo.keys(), avg_dist_err, avg_time, freq_hops, sus_cand
            ):
            df_dist_err[alg_name].append(de)
            df_time[alg_name].append(time)
            dict_freq[alg_name] = [freq[i] if i in freq else 0 for i in range(4)]
            dict_cand[alg_name] = cand
        
        df = pd.DataFrame(dict_freq, columns=algo.keys(), index=list(range(4)))
        err_freq.append(df)
        
        df = pd.DataFrame(dict_cand, columns=algo.keys())
        df_candidates.append(df)    
    
    df_de = pd.DataFrame(df_dist_err, columns=algo.keys(), index=dataset.keys())
    df_time = pd.DataFrame(df_time, columns=algo.keys(), index=dataset.keys())
                        
    return df_de, df_time, err_freq, df_candidates
    

# For Algorithms using Complete Observation - GMLA, PTVA
def gen_data_complete(algo, dataset, iterations):
    df_dist_err = defaultdict(list)
    df_time = defaultdict(list)
    df_candidates = []
    err_freq = []
    
    for G_name, G in dataset.items():
        N = G.number_of_nodes()
        score_time_cand = [alg(G, G_name, iterations) for alg in algo.values()]
        alg_data = [list(zip(*res)) for res in score_time_cand]
        dict_freq = defaultdict(list)
        dict_cand = defaultdict(list)
        
        freq_hops = [Counter(err) for err, _, _ in alg_data]
        avg_dist_err = [stats.mean(err) for err, _, _ in alg_data]
        avg_time = [stats.mean(time) for _, time, _ in alg_data]
        sus_cand = [candidates for _, _, candidates in alg_data]
        
        for alg_name, de, time, freq, cand in zip(
            algo.keys(), avg_dist_err, avg_time, freq_hops, sus_cand
        ):
            df_dist_err[alg_name].append(de)
            df_time[alg_name].append(time)
            dict_freq[alg_name] = [freq[i] if i in freq else 0 for i in range(4)]
            dict_cand[alg_name] = cand
            
        df = pd.DataFrame(dict_freq, columns=algo.keys(), index=list(range(4)))
        err_freq.append(df)
        
        df = pd.DataFrame(dict_cand, columns=algo.keys())
        df_candidates.append(df)
        
        
    df_de = pd.DataFrame(df_dist_err, columns=algo.keys(), index=dataset.keys())
    df_time = pd.DataFrame(df_time, columns=algo.keys(), index=dataset.keys())
    
    return df_de, df_time, err_freq, df_candidates
    

# Datasets and algorithms
comp_algo = {
    "GMLA": GMLA, 
    "PTVA": PTVA_algo
    }
par_algo = {
    "GFHF": TSSI_GFHF, 
    "LGC": TSSI_LGC
    }

dataset = {
        "Karate": nx.karate_club_graph(),
        "Football": G_football,
        # "Facebook": G_facebook,
        "Dolphin": G_dolphin,
        "albert barabasi": nx.barabasi_albert_graph(n=100, m=5),
        "erdos renyi": nx.erdos_renyi_graph(n=100, p=0.2),
        "Adjective Noun": adj_noun
}

# de_par, time_par, err_freq_par, cand_par = gen_data_partial(par_algo, dataset, 30)
# de_comp, time_comp, freq_comp, cand_comp = gen_data_complete(comp_algo, dataset, 30)

# # Plotting
# ## Distance Error
# de = pd.concat([de_par, de_comp], axis=1)
# de.plot.bar(title="Distance Error", xlabel="Datasets", ylabel="distance error")
# plt.xlabel("Datasets", rotation=0)
# plt.savefig(fname=f"figures/distance_err_{datetime.today()}.png", format="png")

# ## Time of Execution
# time = pd.concat([time_par, time_comp], axis=1)
# time.plot.bar(title="Time of execution", xlabel="Datasets", ylabel="time (in s)")
# plt.xlabel("Datasets", rotation=0)
# plt.savefig(fname=f"figures/execution_time_{datetime.today()}.png", format="png")

# ## Fequency of number of hops
# fig, axes = plt.subplots(nrows=math.ceil(len(dataset) / 2), ncols=2, figsize=(15, 15))
# fig.suptitle('Number of Hops')
# freq = [pd.concat([p, c], axis=1) for p, c in zip(err_freq_par, freq_comp)]
# for ax, f, title in zip(axes.flatten(), freq, dataset.keys()):
#     f.plot.bar(title=title, ax=ax, ylabel="frequency")
# plt.tight_layout()
# plt.savefig(fname=f"figures/number_of_hops_{datetime.today()}.png", format="png")


# ## Whisker Plot for Number of candidate sources
# fig, axes = plt.subplots(nrows=math.ceil(len(dataset) / 2), ncols=2, figsize=(15, 15))
# fig.suptitle('Number of Candidate Sources')
# cand = [pd.concat([p, c], axis=1) for p, c in zip(cand_par, cand_comp)]
# for ax, name, c in zip(axes.flatten(), dataset.keys(), cand):
#     box_axes = sns.boxplot(data=c, ax=ax)
#     box_axes.set(title = name, ylabel = 'No of candidate sources')
# plt.tight_layout()
# plt.savefig(fname=f"figures/number_of_candidate_src_{datetime.today()}.png", format="png")

    
# # Save - de, time, freq, cand
# cand_df = []
# for name, c in zip(dataset.keys(), cand):
#     cand_df.append(pd.DataFrame(c, columns=["GFHF", "LGC", "GMLA", "PTVA"]))
    
# with pd.ExcelWriter('output/output.xlsx', engine='xlsxwriter') as writer:  
#     de.to_excel(writer, sheet_name='distance_error')
#     time.to_excel(writer, sheet_name="execution_time")
#     for name, c in zip(dataset.keys(), cand_df):
#         c.to_excel(writer, sheet_name=f"{name}-candidates_src")

def plot_graph(dataset, iterations, graph_label, par_algo = par_algo, comp_algo = comp_algo):
    de_par, time_par, err_freq_par, cand_par = gen_data_partial(par_algo, dataset, iterations)
    de_comp, time_comp, freq_comp, cand_comp = gen_data_complete(comp_algo, dataset, iterations)

    # Plotting
    ## Distance Error
    de = pd.concat([de_par, de_comp], axis=1)
    de.plot.bar(title="Distance Error", xlabel="Datasets", ylabel="distance error")
    plt.savefig(fname=f"figures/distance_err_{datetime.today()}.png", format="png")
    de.plot.line(title="Distance Error", xlabel="Density", ylabel="distance error")
    plt.savefig(fname=f"figures/line_plot_distance_err{datetime.today()}.png", format="png")

    ## Time of Execution
    time = pd.concat([time_par, time_comp], axis=1)
    time.plot.bar(title="Time of execution", xlabel="Datasets", ylabel="time (in s)")
    plt.savefig(fname=f"figures/execution_time_{datetime.today()}.png", format="png")

    ## Fequency of number of hops
    fig, axes = plt.subplots(nrows=math.ceil(len(dataset) / 2), ncols=2, figsize=(15, 15))
    fig.suptitle('Number of Hops')
    freq = [pd.concat([p, c], axis=1) for p, c in zip(err_freq_par, freq_comp)]
    for ax, f, title in zip(axes.flatten(), freq, dataset.keys()):
        f.plot.bar(title=title, ax=ax, ylabel="frequency")
    plt.tight_layout()
    plt.savefig(fname=f"figures/number_of_hops_{datetime.today()}.png", format="png")

    ## Whisker Plot for Number of candidate sources
    fig, axes = plt.subplots(nrows=math.ceil(len(dataset) / 2), ncols=2, figsize=(15, 15))
    fig.suptitle('Number of Candidate Sources')
    cand = [pd.concat([p, c], axis=1) for p, c in zip(cand_par, cand_comp)]
    for ax, name, c in zip(axes.flatten(), dataset.keys(), cand):
        box_axes = sns.boxplot(data=c, ax=ax)
        box_axes.set(title = name, ylabel = 'No of candidate sources')
    plt.tight_layout()
    plt.savefig(fname=f"figures/number_of_candidate_src_{datetime.today()}.png", format="png")
    
    # Save - de, time, freq, cand
    cand_df = []
    for name, c in zip(dataset.keys(), cand):
        cand_df.append(pd.DataFrame(c, columns=["GFHF", "LGC", "GMLA", "PTVA"]))

    with pd.ExcelWriter(f'output/{graph_label}_output.xlsx', engine='xlsxwriter') as writer:  
        de.to_excel(writer, sheet_name='distance_error')
        time.to_excel(writer, sheet_name="execution_time")
        for name, c in zip(dataset.keys(), cand_df):
            c.to_excel(writer, sheet_name=f"{name}-candidates_src")
    