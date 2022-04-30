import numpy as np
import matplotlib.pyplot as plt

N = 5
ind = np.arange(N)
width = 0.15
fig, ax = plt.subplots(1, 4, sharex=False, sharey=False, figsize=(12, 3))
xvals = [6, 24, 52, 18, 0]
ax[0].bar(ind, xvals, width, color='cornflowerblue', label='PTVA')

yvals = [4, 30, 46, 20, 0]
ax[0].bar(ind + width, yvals, width, color='lime', label='GMLA')

zvals = [4, 28, 50, 18, 0]
ax[0].bar(ind + width * 2, zvals, width, color='chocolate', label='ROSE')

svals = [6, 36, 40, 18, 0]
ax[0].bar(ind + width * 3, svals, width, color='cyan', label='NEW1')

tvals = [28, 4, 18, 50, 0]
ax[0].bar(ind + width * 4, tvals, width, color='mediumblue', label='NEW2')
ax[0].grid(True)
ax[0].legend()
# plt.xlabel("Distance error")
# plt.ylabel('Frequency [%]')
# ax[1].title("KT network")

ax[0].set_xticks([0, 1, 2, 3, 4])
# ax[0].set_xticklabels(['0', '1', '2', '3', '4'], size=12)
# ax[0].set_yticklabels(['0', '10', '20', '30', '40', '50', '60'], size=12)
ax[0].set_title('KT network', fontsize=12)
# plt.legend((bar1, bar2, bar3), ('PTVA', 'GMLA', 'ROSE'))
# plt.savefig("bar_graph_DDE_KT")
# plt.show()

N = 7
ind = np.arange(N)

xvals = [0, 8, 16, 8, 52, 16, 0]
ax[1].bar(ind, xvals, width, color='cornflowerblue', label='PTVA')

yvals = [0, 16, 30, 26, 14, 6, 4]
ax[1].bar(ind + width, yvals, width, color='lime', label='GMLA')

zvals = [8, 24, 42, 22, 4, 0, 0]
ax[1].bar(ind + width * 2, zvals, width, color='chocolate', label='ROSE')

svals = [6, 36, 40, 18, 0, 0, 0]
ax[1].bar(ind + width * 3, svals, width, color='cyan', label='New1')

tvals = [28, 4, 18, 50, 0, 0, 0]
ax[1].bar(ind + width * 4, tvals, width, color='mediumblue', label='New2')
ax[1].grid(True)
ax[1].legend()
# plt.xlabel("Distance error")
# plt.ylabel('Frequency [%]')
# plt.title("DL network")

ax[1].set_xticks([0, 1, 2, 3, 4, 5, 6])
# ax[1].set_xticklabels(['0', '1', '2', '3', '4', '5', '6'], size=12)
ax[1].set_title('DL network', fontsize=12)
# plt.legend((bar1, bar2, bar3), ('PTVA', 'GMLA', 'ROSE'))

# plt.savefig("bar_graph_DDE_DL")
# plt.show()

# plt.subplot(1, 4, 3, sharey=True)
N = 5
ind = np.arange(N)

xvals = [0, 18, 42, 40, 0]
ax[2].bar(ind, xvals, width, color='cornflowerblue', label='PTVA')

yvals = [0, 14, 42, 42, 2]
ax[2].bar(ind + width, yvals, width, color='lime', label='GMLA')

zvals = [2, 58, 36, 4, 0]
ax[2].bar(ind + width * 2, zvals, width, color='chocolate', label='ROSE')

svals = [6, 36, 40, 18, 0]
ax[2].bar(ind + width * 3, svals, width, color='cyan', label='New1')

tvals = [28, 4, 18, 50, 0]
ax[2].bar(ind + width * 4, tvals, width, color='mediumblue', label='New2')

ax[2].grid(True)
ax[2].legend()
# plt.xlabel("Distance error")
# plt.ylabel('Frequency [%]')
# plt.title("FL network")
# #
ax[2].set_xticks([0, 1, 2, 3, 4])
# ax[2].set_xticklabels(['0', '1', '2', '3', '4'], size=12)
# ax[2].set_yticklabels(['0', '10', '20', '30', '40', '50', '60'], size=14)
ax[2].set_title('FL network', fontsize=12)
# plt.legend((bar1, bar2, bar3), ('PTVA', 'GMLA', 'ROSE'))
# plt.savefig("bar_graph_DDE_FL")
# plt.show()

N = 7
ind = np.arange(N)

xvals = [0, 10, 26, 26, 22, 16, 0]
ax[3].bar(ind, xvals, width, color='cornflowerblue', label='PTVA')

yvals = [0, 12, 34, 36, 8, 10, 0]
ax[3].bar(ind + width, yvals, width, color='lime', label='GMLA')

zvals = [0, 58, 34, 6, 2, 0, 0]
ax[3].bar(ind + width * 2, zvals, width, color='chocolate', label='ROSE')

svals = [6, 36, 40, 18, 0, 0, 0]
ax[3].bar(ind + width * 3, svals, width, color='cyan', label='New1')

tvals = [28, 4, 18, 50, 0, 0, 0]
ax[3].bar(ind + width * 4, tvals, width, color='mediumblue', label='New2')
ax[3].grid(True)
# plt.xlabel("Distance error")
# plt.ylabel('Frequency [%]')
# plt.title("FB1 network")

ax[3].set_xticks([0, 1, 2, 3, 4, 5, 6])
# ax[3].set_xticklabels(['0', '1', '2', '3', '4', '5', '6'], size=12)
ax[3].legend()
ax[3].set_title('FB1 network', fontsize=12)
# plt.savefig("bar_graph_DDE_FB1")
# plt.xlabel('Distance error')
# plt.ylabel('Frequency [%]')
fig.text(0.5, 0.01, 'Distance error', ha='center', fontsize=12)
fig.text(0.001, 0.5, 'Frequency [%]', va='center', rotation='vertical', fontsize=12)

plt.tight_layout()
plt.savefig("bar_graph_combined_dde_with_titles_2_2.eps", dpi=1200)
plt.show()
