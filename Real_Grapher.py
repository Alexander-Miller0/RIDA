#!/usr/bin/env python
# coding: utf-8


## Settings ##

file_name = 'Real hardware results.pkl' # copy paste the name of the file to graph

seperate_plegend = True # whether to save Pauli graph legend as a seperate file

save_results = False # whether to save graphs to seperate .png files
print_results = True # whether to send graphs to output

raw_color = '#7B3E3E'
zne_color = "#444A64"
cnot_color = '#91BE8D'
rida_color = "#FF4400"


## Import Libraries

import pickle
import matplotlib.pyplot as plt
import numpy as np
import os
import time
from scipy.optimize import curve_fit


## Compute Error ##

# calculate the RMSE across all Pauli strings
def rmse():
    raw_rmse = 0
    rida_rmse = 0
    cnot_rmse = 0
    zne_rmse = 0
    for j in range(num_strings): # add squared errors
        raw_rmse += (statevector[j] - raw[j])**2
        rida_rmse += (statevector[j] - rida[j])**2
        cnot_rmse += (statevector[j] - cnot[j])**2
        zne_rmse += (statevector[j] - zne[j])**2

    # divide by number of samples and square root to find rmse
    raw_rmse = np.sqrt(raw_rmse/(num_strings))
    rida_rmse = np.sqrt(rida_rmse/(num_strings))
    cnot_rmse = np.sqrt(cnot_rmse/(num_strings))
    zne_rmse = np.sqrt(zne_rmse/(num_strings))
    return raw_rmse, rida_rmse, cnot_rmse, zne_rmse


## Grapher ##

# graph error as a function of error-free expectations of Pauli strings
def graph_pauli():
    # compute errors for each Pauli string (not RMSE across Pauli strings)
    raw_error = np.abs(raw - statevector)
    rida_error = np.abs(rida - statevector)
    cnot_error = np.abs(cnot - statevector)
    zne_error = np.abs(zne - statevector)
    fig, ax = plt.subplots(constrained_layout=True)

    height_cutoff = 1 # y limit of the graph - some outliers may exceed this
    base_size = 2 # size of points
    size_factor = 500/num_strings # make bigger dots when there are less points
    size = base_size*size_factor

    plt.scatter(statevector, raw_error, color=raw_color, label='Unmitigated', s=size, alpha=0.6, marker='o', zorder=1)
    plt.scatter(statevector, rida_error, color=rida_color, label='RIDA', s=size, alpha=0.6, marker='s', zorder=4)
    plt.scatter(statevector, zne_error, color=zne_color, label='ZNE + TREX', s=size, alpha=0.6, marker='^', zorder=2)
    plt.scatter(statevector, cnot_error, color=cnot_color, label='CNOT-Only Depolarization + Quadratic ZNE', s=size, alpha=0.6, marker='x', zorder=3)
    plt.xlabel('Error-Free Expectation', fontsize=17)
    plt.ylabel('Error', fontsize=17)
    plt.xlim(-1,1)
    plt.ylim(0, height_cutoff) # bottom cannot be zero if logscale
    for spine in ax.spines.values():
        spine.set_linewidth(2)
    fig.suptitle('Real Hardware: ' + str(nqubits) + ' qubits, ' + str(shots_per_circ) + ' shots', fontsize=17)
    if seperate_plegend and save_results:
        handles, labels = ax.get_legend_handles_labels()
        fig_legend = plt.figure(figsize=(5, 0.1))
        legend = fig_legend.legend(handles, labels, ncol=5, markerscale=3/np.sqrt(size_factor), loc='center', frameon=False)
        fig_legend.gca().set_axis_off()
        file_path = 'LEGEND - paulis'
        if os.path.exists(file_path + '.png'): # check if the file already exists
            file_path = file_path + str(time.time()) # add a timestamp to avoid overwriting existing file
        fig_legend.savefig(file_path + '.png', bbox_inches='tight', dpi=500)
        plt.close(fig_legend)
    else:
        fig.legend(loc='outside lower center', ncol=2, fontsize=10, markerscale=3/np.sqrt(size_factor), frameon=False) # place the legend below the title
    if save_results:
        file_path = 'PAULI GRAPH - ' + str(nqubits) + ' qubits, real hardware'
        if os.path.exists(file_path + '.png'): # check if the file already exists
            file_path = file_path + str(time.time()) # add a timestamp to avoid overwriting existing file
        plt.savefig(file_path + '.png', dpi=500)
    plt.ylim(0,height_cutoff)
    if print_results:
        plt.show()
    else:
        plt.clf() # prevent figure from carrying over

def graph_bar():
    fig, ax = plt.subplots(constrained_layout=True)
    fig.suptitle('Real Hardware: ' + str(nqubits) + ' qubits, ' + str(shots_per_circ) + ' shots', fontsize=17)
    plt.bar('Unmitigated', np.mean(raw_rmse), color=raw_color, label='Unmitigated')
    plt.bar('CNOT-Only', np.mean(cnot_rmse), color=cnot_color, label='CNOT-Only Depolarization')
    plt.bar('ZNE', np.mean(zne_rmse), color=zne_color, label='Exponential ZNE + TREX')
    plt.bar('RIDA', np.mean(rida_rmse), color=rida_color, label='RIDA')

    plt.xlabel('Method', fontsize=16)
    plt.ylabel('RMSE' ,fontsize=16)
    for spine in plt.gca().spines.values():
        spine.set_linewidth(2)
    if not seperate_plegend:
        fig.legend(loc='outside lower center', fontsize=10, frameon=False, ncol=2)
    if save_results:
        file_path = 'BAR GRAPH - ' + str(nqubits) + ' qubits, real hardware'
        if os.path.exists(file_path + '.png'): # check if the file already exists
            file_path = file_path + str(time.time()) # add a timestamp to avoid overwriting existing file
        plt.savefig(file_path + '.png', dpi=500)
    if print_results:
        plt.show()
    else:
        plt.clf() # prevent figure from carrying over


## Driver ##

# load relevant variables from a dictionary of a single run
with open(file_name, 'rb') as f:
    dict = pickle.load(f)
nqubits, shots_per_circ, rida_depo, cnot_depo, statevector, raw, rida, cnot, zne = \
dict['nqubits'], dict['shots_per_circ'], dict['rida_depo'], dict['cnot_depo'], \
dict['statevector'], dict['raw'], dict['rida'], dict['cnot'], dict['zne']
num_strings = 30

# create graphs
graph_pauli()
raw_rmse, rida_rmse, cnot_rmse, zne_rmse = rmse()
print(rmse())
raw_past = 0
zne_past = 0
cnot_past = 0
rida_past = 0
for i in range(30):
    if np.abs(statevector[i] - raw[i]) < 0.1:
        raw_past += 1
    if np.abs(statevector[i] - zne[i]) < 0.1:
        zne_past += 1
    if np.abs(statevector[i] - cnot[i]) < 0.1:
        cnot_past += 1
    if np.abs(statevector[i] - rida[i]) < 0.1:
        rida_past += 1
print(raw_past, zne_past, cnot_past, rida_past)
graph_bar()