#!/usr/bin/env python
# coding: utf-8


## Settings ##

file_name = 'DEPO RESULTS - 7 qubits, 3x error.pkl' # copy paste the name of the file to graph
shot_focus = -1 # which index of the shot list to use for the unmitigated data that the optimal depolarization is fit to

graph_bar = False # whether to graph the bar graph with error bars
graph_violin = True # whether to graph the violin plot

use_rotations = True # include CNOT-only depolarization with random rotations
error_bar = True # whether to include error bars

save_results = True # whether to save graphs to seperate .png files
print_results = True # whether to send graphs to output

cnot_color = '#91BE8D'
rida_color = "#FF4400"


## Import Libraries ##

import pickle
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import os
import time

rng = np.random.default_rng()


## Compute Statistics ##

# optimal depolarization p for a set of (error-free, unmitigated) pairs
def optimal_depo(statevector, raw):
    denom = np.sum(statevector*raw)
    if denom <= 0:
        return np.nan
    return 1 - np.sum(raw**2)/denom

# weighted mean and standard error of the mean depolarization of one method
# circ is (number of frames, number of estimation circuits); weights is the number of Pauli strings in each frame
def method_stats(circ):
    nframes, ncirc = circ.shape
    frame_means = circ.mean(axis=1)
    frame_ses = circ.std(axis=1, ddof=1)/np.sqrt(ncirc) # standard error of each frame's estimate
    total = np.sum(weights)
    mean = np.sum(weights*frame_means)/total
    se_estimation = np.sqrt(np.sum(weights**2*frame_ses**2))/total
    return mean, se_estimation, frame_means

# the error bar size for a method under the chosen setting
def error_size(se_estimation):
    if error_bar:
        return se_estimation
    return None

# short description of the error bars for their legend entry
def error_caption():
    if error_bar:
        return 'Standard error from the\nestimation circuits' # two lines so the legend fits in the figure
    return None

# .png file name that does not overwrite an existing graph
def unique_path(path):
    if os.path.exists(path + '.png'):
        path = path + ' - ' + str(int(time.time())) # add a timestamp to avoid overwriting existing file
    return path + '.png'


## Graphs ##

# compare depolarization magnitudes in a bar graph with error bars
def graph_depo_bar():
    fig, ax = plt.subplots(constrained_layout=True)
    plt.axhline(y=true_depo, color='black', label='Optimal', linestyle='dashed')
    for method in methods:
        plt.bar(method['name'], method['mean'], yerr=method['err'], capsize=8, ecolor='black', color=method['color'],
                label=method['label'], hatch=method['hatch'], error_kw={'linewidth': 2, 'capthick': 2})
    if error_bar: # legend entry for the bars' error bars, drawn as a small capped error bar (the point itself is invisible)
        plt.errorbar(np.nan, np.nan, yerr=np.nan, fmt='none', ecolor='black', elinewidth=2, capsize=8, capthick=2, label=error_caption())

    plt.xlabel('Method', fontsize=16)
    plt.ylabel('Depolarization Estimate p', fontsize=16)
    plt.title(str(nqubits) + ' qubits, ' + str(error_multiplier) + 'x error', fontsize=16)
    for spine in plt.gca().spines.values():
        spine.set_linewidth(2)
    fig.legend(loc='outside lower center', fontsize=10, frameon=False, ncol=2)
    if save_results:
        plt.savefig(unique_path('BAR DEPOLARIZATION GRAPH (error bars) - ' + str(nqubits) + ' qubits, ' + str(error_multiplier) + 'x error'), dpi=500)
    if print_results:
        plt.show()
    else:
        plt.close(fig) # prevent figure from carrying over

# compare the distributions of individual depolarization estimates in a violin plot
def graph_depo_violin():
    fig, ax = plt.subplots(constrained_layout=True)
    plt.axhline(y=true_depo, color='black', label='Optimal', linestyle='dashed', zorder=1)
    positions = np.arange(1, len(methods)+1)
    for pos, method in zip(positions, methods):
        values = method['circ'].flatten()
        if len(values) < 2 or np.ptp(values) == 0:
            print('Skipping violin for', method['name'], '(not enough spread in the estimates)')
            continue
        body = plt.violinplot(values, positions=[pos], widths=0.8, showextrema=False)['bodies'][0]
        body.set_facecolor(method['color'])
        body.set_edgecolor('black')
        body.set_alpha(1)
        body.set_linewidth(1.5)
        body.set_zorder(2)
        if method['hatch']:
            body.set_hatch(method['hatch'])
        q1, median, q3 = np.percentile(values, [25, 50, 75])
        iqr = q3 - q1
        low_whisker = np.min(values[values >= q1 - 1.5*iqr]) # Tukey whiskers: furthest data points within 1.5 IQR of the quartiles
        high_whisker = np.max(values[values <= q3 + 1.5*iqr])
        plt.vlines(pos, low_whisker, high_whisker, color='black', linewidth=1.5, zorder=3) # whiskers
        plt.vlines(pos, q1, q3, color='black', linewidth=5, zorder=3) # interquartile range
        plt.scatter(pos, median, color='white', edgecolor='black', s=40, zorder=4) # median
    plt.xticks(positions, [method['name'] for method in methods])

    plt.xlabel('Method', fontsize=16)
    plt.ylabel('Depolarization Estimate p', fontsize=16)
    plt.title(str(nqubits) + ' qubits, ' + str(error_multiplier) + 'x error', fontsize=16)
    for spine in plt.gca().spines.values():
        spine.set_linewidth(2)
    handles = [Line2D([], [], color='black', linestyle='dashed', label='Optimal')]
    for method in methods: # one colored entry per method, as in the bar graph
        handles.append(Patch(facecolor=method['color'], hatch=method['hatch'], label=method['label']))
    fig.legend(handles=handles, loc='outside lower center', fontsize=10, frameon=False, ncol=2) # legend below the graph, laid out like the bar graph's
    if save_results:
        plt.savefig(unique_path('VIOLIN DEPOLARIZATION GRAPH - ' + str(nqubits) + ' qubits, ' + str(error_multiplier) + 'x error'), dpi=500)
    if print_results:
        plt.show()
    else:
        plt.close(fig) # prevent figure from carrying over


## Driver ##

# load relevant variables from a dictionary of a single run
with open(file_name, 'rb') as f:
    dict = pickle.load(f)
nqubits, error_multiplier, shots, statevector, raw, frame_index = \
dict['nqubits'], dict['error_multiplier'], dict['shots'], np.asarray(dict['statevector']), np.asarray(dict['raw']), np.asarray(dict['frame_index'])
raw_focus = raw[shot_focus]
nframes = len(dict['frames'])
weights = np.bincount(frame_index, minlength=nframes).astype(float) # number of Pauli strings in each frame

if dict['benchmark_size'] < 2 and error_bar:
    raise ValueError('Estimation error bars need at least 2 estimation circuits per frame')

# collect each method's data
methods = [{'name': 'No rotations', 'label': 'CNOT-Only Depolarization', 'circ': np.asarray(dict['cnot_circ_depo']), 'color': cnot_color, 'hatch': None}]
if use_rotations:
    methods.append({'name': 'Rotations', 'label': 'CNOT-Only Depolarization (With Rotations)', 'circ': np.asarray(dict['cnot_rotations_circ_depo']), 'color': cnot_color, 'hatch': '///'})
methods.append({'name': 'RIDA', 'label': 'RIDA', 'circ': np.asarray(dict['rida_circ_depo']), 'color': rida_color, 'hatch': None})
for method in methods:
    method['mean'], method['se_estimation'], method['frame_means'] = method_stats(method['circ'])
    method['err'] = error_size(method['se_estimation'])

true_depo = optimal_depo(statevector, raw_focus)

if graph_bar:
    graph_depo_bar()
if graph_violin:
    graph_depo_violin()