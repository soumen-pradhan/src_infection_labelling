from visualizations import *


# Datasets and algorithms
comp_algo = {"GMLA": GMLA, "PTVA": PTVA_algo}
par_algo = {"GFHF": TSSI_GFHF, "LGC": TSSI_LGC}

## Datasets - BA - ER 
BA_EA_edge_connectivity = list(np.arange(1, 6));

erdos_renyi_500 = defaultdict();
albert_barabasi_500 = defaultdict();

erdos_renyi_1000 = defaultdict();
albert_barabasi_1000 = defaultdict();

for i in BA_EA_edge_connectivity:
    G1, _ = load(f"datasets/random/BA_500_{i}_giant_edge_list");
    G2, _ = load(f"datasets/random/ER_500_{i}_giant_edge_list");
    erdos_renyi_500[i] = G1
    albert_barabasi_500[i] = G2
    
    G1, _ = load(f"datasets/random/BA_1000_{i}_giant_edge_list");
    G2, _ = load(f"datasets/random/ER_1000_{i}_giant_edge_list");
    erdos_renyi_1000[i] = G1
    albert_barabasi_1000[i] = G2

## barabasi with 100 nodes
barabasi2 = defaultdict()
for i in BA_EA_edge_connectivity:
     G, _ = load(f"datasets/random/barabasi_100_{i}_giant_edge_list");
     barabasi2[i] = G

## erdos renyi with density    
ER_dense = defaultdict() 
erdos_renyi_density = ["0.02", "0.04", "0.06", "0.08", "0.10"]
for i in erdos_renyi_density:
    G, _ = load(f"datasets/random/ER_100_{i}_giant_edge_list");
    ER_dense[i] = G
    
## facebook dataset
facebook, _ = load("datasets/random/facebook_150_giant_edge_list")
fb_dataset = {"facebook_150" : facebook}

datasets = [E]
plot_graph(ER_dense, 30, "ER (100)")