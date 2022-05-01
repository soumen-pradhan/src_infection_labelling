from visualizations import *


# Datasets and algorithms
comp_algo = {"GMLA": GMLA, "PTVA": PTVA_algo}
par_algo = {"GFHF": TSSI_GFHF, "LGC": TSSI_LGC}

## Datasets - BA - ER 
BA_EA_nodes = [500, 1000]
BA_EA_edge_connectivity = list(np.arange(1, 6));
erdos_renyi = defaultdict();
albert_barabasi = defaultdict();
for i in BA_EA_nodes:
    for j in BA_EA_edge_connectivity:
        G1, _ = load(f"datasets/random/BA_{i}_{j}_giant_edge_list");
        G2, _ = load(f"datasets/random/ER_{i}_{j}_giant_edge_list");
        erdos_renyi[f"ER_{i}_{j}"] = G1
        albert_barabasi[f"BA_{i}_{j}"] = G2

## barabasi with 100 nodes
barabasi2 = defaultdict()
for i in BA_EA_edge_connectivity:
     G, _ = load(f"datasets/random/barabasi_100_{i}_giant_edge_list");
     barabasi2[f"barabasi_100_{i}"] = G

## erdos renyi with density    
ER_dense = defaultdict() 
erdos_renyi_density = ["0.02", "0.04", "0.06", "0.08", "0.10"]
for i in erdos_renyi_density:
    G, _ = load(f"datasets/random/ER_100_{i}_giant_edge_list");
    ER_dense[f"ER_100_{i}"] = G
    
## facebook dataset
facebook, _ = load("datasets/random/facebook_150_giant_edge_list")
fb_dataset = {"facebook_150" : facebook}

plot_graph(ER_dense, 30, "ER_100_density")