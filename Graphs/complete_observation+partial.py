from TSSI_GFHF_LGC import *

#Complete Observation
G = nx.karate_club_graph()
src = np.random.choice(list(G.nodes()))

y = simulateInfection(G, src)
f, _= labelRankingScore(G, y)

detected_src_idx = max(enumerate(f), key=lambda x: x[1])
detected_src = list(G.nodes())[detected_src_idx[0]]

fig, ax = plt.subplots(figsize=(20, 20))
pos = nx.spring_layout(G, k=0.2, seed=42)
size = [v * 5000 if v > 0 else 800 for v in f.values()]
color = ['green' if n < 0 else 'red' for n in f.values()]

nx.draw_networkx(
        G,
        ax=ax,
        pos=pos,
        with_labels=True,
        node_color=color,
        node_size=size,
        edge_color="#adadad",
        alpha=0.4,
        font_weight='bold')

plt.show()


print(f'Actual Source: {src}\n'
      f'Predicted Source: {detected_src}')


#Partial Observation
G = nx.karate_club_graph()
src = np.random.choice(list(G.nodes()))
N = G.number_of_nodes()

known_dict = simulatePartialInfection(G, src)
known_dict = known_dict[0]
O, _ = GFHF(G, known_dict)
f, _ = labelRankingScore(G, O)

detected_src_idx = max(enumerate(f), key=lambda x: x[1])
detected_src = list(G.nodes())[detected_src_idx[0]]

fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(20, 20))
fig.tight_layout()

pos = nx.spring_layout(G, k=0.2, seed=42)
size = [v * 5000 if v > 0 else 500 for v in f.values()]

color1 = ['green' if n in known_dict['safe'] else
          'red' if n in known_dict['infected'] else
          'gray' for n in G.nodes]

color2 = ['green' if n < 0 else 'red' for n in f.values()]

nx.draw_networkx( # Partial Graph snapshot
        G, 
        ax=ax1,
        pos=pos,
        with_labels=True,
        node_color=color1,
        node_size=1000,
        edge_color="#adadad",
        alpha=0.4,
        font_weight='bold')

nx.draw_networkx( # Predicted Graph
        G,
        ax=ax2,
        pos=pos,
        with_labels=True,
        node_color=color2,
        node_size=size,
        edge_color="#adadad",
        alpha=0.4,
        font_weight='bold')

plt.show()


print(f'Actual Source: {src}\n'
      f'Predicted Source: {detected_src}')
