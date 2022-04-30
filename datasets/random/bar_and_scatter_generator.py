import numpy as np
import matplotlib.pyplot as plt

N = 6
ind = np.arange(N)
width = 0.2
fig, ax = plt.subplots(1, 2, sharey=False, figsize=(10, 4))
xvals = [1.94, 2.3, 2.8, 3.35, 3.15, 3.25]
ax[0].bar(ind, xvals, width, color='cornflowerblue')

yvals = [1.8, 2.08, 2.5, 2.7, 2.55, 2.8]
ax[0].bar(ind + width, yvals, width, color='lime')

zvals = [1.68, 1.75, 1.8, 2.3, 2.5, 2.1]
ax[0].bar(ind + width * 2, zvals, width, color='chocolate')
ax[0].grid(True)

ax[0].set_xlabel("Network", fontsize=18)
ax[0].set_ylabel('Distance error', fontsize=18)
# ax[1].title("KT network")

ax[0].set_xticks([0, 1, 2, 3, 4, 5])
ax[0].set_xticklabels(['KT', 'FL', 'FB2', 'BC1', 'BC2', 'WV'], size=16)
ax[0].set_title('Distance error', fontsize=18)
ax[0].tick_params(axis='y', which='major', labelsize=16)
ax[0].tick_params(axis='x', which='major', labelsize=16)
# plt.legend((bar1, bar2, bar3), ('PTVA', 'GMLA', 'ROSE'))
# plt.savefig("bar_graph_DDE_KT")
# plt.show()

N = 5
ind = np.arange(N)
width = 0.2
# fig, ax = plt.subplots(1, 2, sharey=False, figsize=(10, 4))
x1 = [13.5, 20.4, 35.3, 48.5, 62.5]
ax[1].bar(ind, x1, width, color='cornflowerblue', label='PTVA')

y2 = [7.8, 10.5, 15.5, 19.2, 35.6]
ax[1].bar(ind + width, y2, width, color='lime', label='GMLA')

z2 = [0.89, 1.2, 1.5, 3.18, 24.2]
ax[1].bar(ind + width * 2, z2, width, color='chocolate', label='ROSE')
ax[1].grid(True)
ax[1].set_xlabel("Network", fontsize=18)
ax[1].set_ylabel('Time (sec)', fontsize=18)
# plt.title("DL network")

ax[1].set_xticks([0, 1, 2, 3, 4])
ax[1].set_yticks([0, 10, 20, 30, 40, 50, 60])
ax[1].set_xticklabels(['FB2', 'BA1', 'BC1', 'BC2', 'WV'], size=16)
ax[1].tick_params(axis='x', which='major', labelsize=16)
ax[1].tick_params(axis='y', which='major', labelsize=16)

ax[1].set_title('Execution time', fontsize=18)
# fig.text(0.5, 0.02, 'Network', va='center', fontsize=18)
plt.legend()
plt.tight_layout()
# plt.legend((bar1, bar2, bar3), ('PTVA', 'GMLA', 'ROSE'))
plt.savefig("bar_graphs_DE_and_CT.eps", dpi=1200)
plt.show()

