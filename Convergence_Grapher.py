#!/usr/bin/env python
# coding: utf-8


## Settings ##

file_name = 'CONVERGENCE RESULTS - 7 qubits, 3x error.pkl'

raw_color = '#7B3E3E'
zne_color = '#444A64'
cnot_color = '#91BE8D'
rida_color = '#FF4400'

save_results = True # whether to save graphs to seperate .png files
print_results = False # whether to send graphs to output


## Import Libraries ##

import pickle
import numpy as np
import matplotlib.pyplot as plt
import time
import os

plt.ioff() # prevents plots from showing automatically


## Grapher ##

# graph rmse of the estimated p as a function of number of circuits
def graph_convergence():
    fig, ax = plt.subplots(constrained_layout=True)
    domain = [k+1 for k in range(max_circuits)]
    plt.xlim(1, max_circuits) # eliminate margins
    plt.xscale('log')
    plt.yscale('log')
    plt.ylim(0.9*min(np.min(rida_rmse), np.min(cnot_rmse)), 1.3*max(np.max(rida_rmse), np.max(cnot_rmse))) # eliminate margins
    plt.plot(domain, cnot_rmse, linewidth=3, color=cnot_color, label='CNOT-Only Depolarization')
    plt.plot(domain, rida_rmse, linewidth=3, color=rida_color, label='RIDA')
    plt.xlabel('Estimation Circuits', fontsize=16)
    plt.ylabel('Depolarization Estimate Error', fontsize=16)
    plt.title(str(nqubits) + ' qubits, ' + str(error_multiplier) + 'x error, ' + str(int(str(shots_per_circuit)[0])) + '\u00D7' + rf'${10}^{{{int(np.log10(shots_per_circuit))}}}$' + ' shots per circuit', fontsize=16)
    plt.legend(loc='upper left', bbox_to_anchor=(0, 0.94), frameon=False)
    for spine in plt.gca().spines.values():
        spine.set_linewidth(2)
    if save_results:
        file_path = 'CONVERGENCE GRAPH - ' + str(nqubits) + ' qubits, ' + str(error_multiplier) + 'x error'
        if os.path.exists(file_path + '.png'): # check if the file already exists
            file_path = file_path + str(time.time()) # add a timestamp to avoid overwriting existing file
        plt.savefig(file_path + '.png', dpi=500)
    if print_results:
        plt.show()


## Driver ##

with open(file_name, 'rb') as f: # load input data from special convergence results file
    dict = pickle.load(f)
    nqubits, error_multiplier, max_circuits, rida_rmse, cnot_rmse, shots_per_circuit = dict['nqubits'], \
    dict['error_multiplier'], dict['max_circuits'], dict['rida_rmse'], dict['cnot_rmse'], dict['shots_per_circuit']
graph_convergence()