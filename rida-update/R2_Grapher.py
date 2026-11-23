#!/usr/bin/env python
# coding: utf-8


## Settings ##

error_multiplier = 1 # error multiplier to graph
shot_focus = -1 # what index of shots to use for the unmitigated data (-1 = largest number of shots)

qubit_numbers = [4, 5, 6, 7] # numbers of qubits to graph (Efficient SU2 ansatz tests)
graph_ghz = True # whether to also graph R^2 for a GHZ state, on the same graph
ghz_qubits = 8 # number of qubits in the GHZ test (reads 'RESULTS - Nqubits, Mx error (GHZ).pkl' from GHZ_Tester.py)
graph_qft = True # whether to also graph R^2 for a QFT of a random product state, between the ansatz and GHZ bars
qft_qubits = 7 # number of qubits in the QFT test (reads 'RESULTS - Nqubits, Mx error (QFT).pkl' from QFT_Tester.py)
graph_real = True # whether to also graph R^2 for the real hardware results
real_file_name = 'Real hardware results.pkl' # results file from Real_Hardware.py
real_shots = 2976 # shot number to use when graph_real is on in order to match the real shots

save_results = True # whether to save graph to a .png file
print_results = False # whether to send graph to output

raw_color = '#7B3E3E'
qft_color = '#444A64'
ghz_color = 'black'
real_color = '#2F6B4F'


## Import Libraries ##

import pickle
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import textwrap

plt.ioff() # prevents plots from showing automatically


## Helper Functions ##

shots_used = set() # numbers of shots found in the results files being graphed

# when the real hardware bar is on, every bar uses the results files run with the same number of shots as the hardware test
shot_tag = (', ' + str(real_shots) + ' shots') if graph_real else ''

# R^2 of the strict depolarizing model: raw = (1-p) * error-free expectation (no intercept)
# the best-fit (1-p) is found by least squares through the origin
def depo_r2(statevector, raw):
    statevector = np.asarray(statevector)
    raw = np.asarray(raw)
    slope = np.sum(statevector*raw)/np.sum(statevector**2) # least-squares slope through the origin, i.e. 1-p
    residual = np.sum((raw - slope*statevector)**2)
    total = np.sum((raw - np.mean(raw))**2)
    return 1 - residual/total, slope

# compute R^2 for each number of qubits at the chosen error multiplier
# suffix is appended to the file names (' (GHZ)' for GHZ files)
def compute_r2(qubits, suffix=''):
    r2 = np.zeros(len(qubits))
    for q in range(len(qubits)):
        file_name = 'RESULTS - ' + str(qubits[q]) + ' qubits, ' + str(error_multiplier) + 'x error' + suffix + shot_tag + '.pkl'
        with open(file_name, 'rb') as f:
            dict = pickle.load(f)
        r2[q], _ = depo_r2(dict['statevector'], dict['raw'][shot_focus])
        shots_used.add(int(dict['shots'][shot_focus])) # keep track of the number of shots in the data being graphed
    return r2

# describe the shots used in the graphed files, either as plain text (for printing) or with a superscript (for the graph title)
def shots_label(mathtext=False):
    if len(shots_used) != 1:
        return 'shots vary between files: ' + ', '.join(str(k) for k in sorted(shots_used)) # files with different shot numbers are mixed
    k = list(shots_used)[0]
    if k > 0 and (k & (k-1)) == 0: # check if the number of shots is a power of two
        return (rf'$2^{{{int(np.log2(k))}}}$' if mathtext else '2^' + str(int(np.log2(k)))) + ' shots'
    return str(k) + ' shots'

# compute R^2 for the real hardware results (raw is a 1D array with one value per Pauli string, unlike the simulated results)
# returns R^2 and the number of qubits
def compute_r2_real():
    with open(real_file_name, 'rb') as f:
        dict = pickle.load(f)
    if dict['shots_per_circ'] != real_shots:
        print('Warning: the real hardware file used ' + str(dict['shots_per_circ']) + ' shots per circuit, but real_shots is ' + str(real_shots))
    r2, _ = depo_r2(dict['statevector'], dict['raw'])
    return r2, dict['nqubits']


## Grapher ##

# bar graph of R^2: one bar per number of qubits for the Efficient SU2 ansatz, then separately labeled QFT, GHZ, and real hardware bars
def graph_r2(r2, r2_qft, r2_ghz, r2_real, real_qubits):
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True) # taller and wider to fit the long y label and extra bars
    positions = list(np.arange(len(qubit_numbers)))
    tick_labels = [str(q) for q in qubit_numbers]

    ansatz_bars = ax.bar(positions, r2, width=0.7, color=raw_color, label=textwrap.fill('Efficient SU2 Ansatz', 16))
    ax.bar_label(ansatz_bars, fmt='%.3f', label_type='edge', padding=-16, color='white', fontsize=11)
    all_values = list(r2)

    # extra groups, in order from left to right: (name, qubits, value, color, hatch)
    extras = []
    if r2_qft is not None:
        extras.append(('QFT', qft_qubits, r2_qft, qft_color, '\\', 'QFT of Random Product State'))
    if r2_ghz is not None:
        extras.append(('GHZ', ghz_qubits, r2_ghz, ghz_color, '//', 'GHZ State'))
    if r2_real is not None:
        extras.append(('Real\nHardware', real_qubits, r2_real, real_color, 'xx', 'EfficientSU2 on IBM Marrakesh'))

    for k in range(len(extras)):
        name, nq, value, color, hatch, label = extras[k]
        position = len(qubit_numbers) - 1 + 1.4*(k+1) # leave a gap to separate each group
        positions.append(position)
        tick_labels.append(str(nq) + '\n(' + name + ')')
        bars = ax.bar([position], [value], width=0.7, color=color, edgecolor='white', hatch=hatch, label=textwrap.fill(label, 16))
        for text in ax.bar_label(bars, fmt='%.3f', label_type='edge', padding=-16, color='white', fontsize=11):
            text.set_bbox(dict(facecolor=color, edgecolor='none', pad=1.5)) # keep the value readable over the hatching
        all_values.append(value)
        ax.axvline(position - 0.7, color='black', linestyle='dashed', linewidth=1, alpha=0.5) # divider between groups

    ax.set_xticks(positions)
    ax.set_xticklabels(tick_labels, fontsize=14)
    ax.set_xlabel('Number of Qubits', fontsize=16)
    ax.set_ylabel(r'$R^2$ of Depolarizing Approximation', fontsize=16)
    ax.set_ylim(0.0, 1.0)
    ax.tick_params(axis='y', labelsize=12)
    for spine in ax.spines.values(): # make borders thicker
        spine.set_linewidth(2)
    fig.suptitle(shots_label(mathtext=True), fontsize=16)
    fig.legend(loc='outside lower center', ncol=(1+len(extras) if len(extras) < 5 else 2), fontsize=11, frameon=False)

    if save_results:
        file_path = 'R2 GRAPH'
        if os.path.exists(file_path + '.png'): # check if the file already exists
            file_path = file_path + str(time.time()) # add a timestamp to avoid overwriting existing file
        plt.savefig(file_path + '.png', dpi=500)
    if print_results:
        plt.show()
    else:
        plt.clf() # prevent figure from carrying over


## Driver ##

r2 = compute_r2(qubit_numbers)

r2_qft = None
if graph_qft:
    try:
        r2_qft = compute_r2([qft_qubits], ' (QFT)')[0]
    except FileNotFoundError as e:
        print('Skipping QFT bar, results file not found: ' + str(e.filename) + ' (run QFT_Tester.py first)')

r2_ghz = None
if graph_ghz:
    try:
        r2_ghz = compute_r2([ghz_qubits], ' (GHZ)')[0]
    except FileNotFoundError as e:
        print('Skipping GHZ bar, results file not found: ' + str(e.filename) + ' (run GHZ_Tester.py first)')

r2_real = None
real_qubits = None
if graph_real:
    try:
        r2_real, real_qubits = compute_r2_real()
    except FileNotFoundError as e:
        print('Skipping real hardware bar, results file not found: ' + str(e.filename) + ' (run Real_Hardware.py first)')

graph_r2(r2, r2_qft, r2_ghz, r2_real, real_qubits)