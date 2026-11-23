#!/usr/bin/env python
# coding: utf-8


## Settings ##

nqubits = 4 # number of qubits to graph
error_multipliers = [1, 2, 3, 4] # error multipliers to graph (one panel each)
shot_focus = -1 # what index of shot list to use for the unmitigated data

min_repeats = 5 # only include Pauli strings that appear at least this many times in the test set
measurement_error = False # whether to graph results with measurement error

save_results = True # whether to save graph to a .png file
print_results = False # whether to send graph to output


## Import Libraries ##

import pickle
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.patches import Patch
import time

plt.ioff() # prevents plots from showing automatically


## Helper Functions ##

# optimal depolarization p for a set of (error-free, unmitigated) pairs that share a Pauli string
# minimizes sum((statevector - raw/(1-p))^2), the same objective as optimal_depo() in the other graphers, in closed form
# returns nan if the fit is undefined (the unmitigated data is uncorrelated or anticorrelated with the error-free data)
def optimal_depo(statevector, raw):
    denom = np.sum(statevector*raw)
    if denom <= 0:
        return np.nan
    return 1 - np.sum(raw**2)/denom

# compute the optimal depolarization of each distinct Pauli string in a results dictionary
# returns a dictionary mapping each string (with at least min_repeats repeats) to (optimal p, number of repeats)
def depo_by_string(dict):
    operators = [str(op) for op in dict['operators']]
    statevector = np.asarray(dict['statevector'])
    raw = np.asarray(dict['raw'][shot_focus])

    groups = {} # Pauli string -> list of indices of the test circuits that measure it
    for j in range(len(operators)):
        groups.setdefault(operators[j], []).append(j)

    out = {}
    for pstring in groups:
        idx = groups[pstring]
        if len(idx) < min_repeats:
            continue
        p = optimal_depo(statevector[idx], raw[idx])
        if not np.isnan(p) and (p < 0 or p > 1):
            p = np.nan # drawn as a missing bar
        out[pstring] = (p, len(idx))
    return out


## Grapher ##

# write each bar's height above it (no label for missing, excluded bars)
def add_labels(axs, all_bars, values, decimals):
    for k in range(len(all_bars)):
        value_labels = ['' if np.isnan(v) else f'{v:.{decimals}f}' for v in values[k]]
        axs[k].bar_label(all_bars[k], labels=value_labels, label_type='edge', padding=2, fontsize=10)

# graph the optimal depolarization of each Pauli string as its own bar, one panel per error multiplier
def graph_bars(strings, repeats, values):
    ncols = 2
    nrows = int(np.ceil(len(error_multipliers)/ncols))
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(11, 4*nrows), constrained_layout=True, squeeze=False)
    axs = axs.flatten()
    x = np.arange(len(strings))
    labels = [strings[i] + ' (' + str(repeats[i]) + ' reps)' for i in range(len(strings))]
    lower = min(0, np.nanmin(values))
    upper = 1.12*np.nanmax(values) # consistent y range across panels, with room above the tallest bar for its label

    weights = [sum(c != 'I' for c in string) for string in strings] # number of non-identity elements in each string
    cmap = plt.get_cmap('viridis')
    weight_colors = {w: cmap(0.9*(w-1)/max(nqubits-1, 1)) for w in range(1, nqubits+1)} # one color per possible weight (the top of the map is skipped so it is not too light)
    bar_colors = [weight_colors[w] for w in weights]

    all_bars = []
    for k in range(len(error_multipliers)):
        ax = axs[k]
        all_bars.append(ax.bar(x, values[k], color=bar_colors, edgecolor='white', linewidth=0.5))
        ax.set_title(str(nqubits) + ' qubits, ' + str(error_multipliers[k]) + 'x error' + ('' if measurement_error else ', no measurement error'), fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
        ax.set_xlim(-0.7, len(strings)-0.3)
        ax.set_ylim(lower, upper)
        ax.tick_params(axis='y', labelsize=11)
        for spine in ax.spines.values(): # make borders thicker
            spine.set_linewidth(2)

    add_labels(axs, all_bars, values, 2) # label each bar with its height

    for k in range(len(error_multipliers), len(axs)): # hide unused panels
        axs[k].set_visible(False)
    fig.supxlabel('Pauli String', fontsize=16)
    key = [Patch(facecolor=weight_colors[w], edgecolor='none', label=str(w)) for w in sorted(set(weights))] # only weights that appear in the graph
    fig.legend(handles=key, title='Number of non-identity elements', loc='outside upper center', ncol=len(key), fontsize=12, title_fontsize=12, frameon=False)
    fig.supylabel('Optimal Depolarization Probability p', fontsize=16)

    if save_results:
        file_path = 'DEPOLARIZATION BY STRING - ' + str(nqubits) + ' qubits' + ('' if measurement_error else ', no measurement error')
        if os.path.exists(file_path + '.png'): # check if the file already exists
            file_path = file_path + str(time.time()) # add a timestamp to avoid overwriting existing file
        plt.savefig(file_path + '.png', dpi=500)
    if print_results:
        plt.show()
    else:
        plt.clf() # prevent figure from carrying over


## Driver ##

results = []
for multiplier in error_multipliers:
    file_name = 'RESULTS - ' + str(nqubits) + ' qubits, ' + str(multiplier) + 'x error' + ('' if measurement_error else ', no measurement error') + '.pkl'
    with open(file_name, 'rb') as f:
        dict = pickle.load(f)
    results.append(depo_by_string(dict))

# use the Pauli strings that meet the repeat threshold in every file, so bars line up across panels
strings = sorted(set.intersection(*[set(r) for r in results]))
if len(strings) == 0:
    raise ValueError('No Pauli string has at least ' + str(min_repeats) + ' repeats - lower min_repeats')
repeats = [results[0][s][1] for s in strings]
values = np.array([[results[k][s][0] for s in strings] for k in range(len(error_multipliers))]) # rows: error multiplier, columns: string

keys = [np.nanmean(values[:,i]) if np.any(~np.isnan(values[:,i])) else np.inf for i in range(len(strings))]
order = np.argsort(keys)
strings = [strings[i] for i in order]
repeats = [repeats[i] for i in order]
values = values[:, order]

graph_bars(strings, repeats, values)