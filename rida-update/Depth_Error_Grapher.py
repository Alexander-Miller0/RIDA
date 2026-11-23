import numpy as np
import matplotlib.pyplot as plt

save_results = True # whether to save graphs to seperate .png files
print_results = True # whether to send graphs to output

raw_color = '#7B3E3E'
zne_color = "#444A64"
cnot_color = '#91BE8D'
rida_color = "#FF4400"
layerwise_color='black'

xlabels = ('4', '5', '6', '7')
rida_error = np.array([22.162, 22.066, 23.477, 23.748])
layerwise_error = np.array([0.047, 0.071, 0.638, 0.037])
errors = {'Standard RIDA': rida_error, 'Layerwise RIDA': layerwise_error}
colors = [rida_color, layerwise_color]

fig, ax = plt.subplots(figsize=(6, 4), layout='constrained') # 3:2, the same proportions as the compressed error multiplier graph in Batch_Grapher.py

res = ax.grouped_bar(errors, colors=colors, tick_labels=xlabels, group_spacing=1)
for container in res.bar_containers:
    ax.bar_label(container, padding=3, fontsize=10)

# Add some text for labels, title, etc. (sizes match the compressed error multiplier graph in Batch_Grapher.py)
ax.set_xlabel('Number of Qubits', fontsize=14)
ax.set_ylabel('Circuit Depth % Error', fontsize=14)
ax.legend(loc='upper left', ncols=3, fontsize=11)
ax.set_ylim(0, 100)
ax.tick_params(axis='both', labelsize=11)
for spine in ax.spines.values(): # make borders thicker
    spine.set_linewidth(2)

if save_results:
    plt.savefig('DEPTH GRAPH.png', dpi=300)
if print_results:
    plt.show()
else:
    plt.clf()