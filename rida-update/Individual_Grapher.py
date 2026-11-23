#!/usr/bin/env python
# coding: utf-8


## Settings ##

file_name = 'RESULTS - 7 qubits, 3x error.pkl' # copy paste the name of the file to graph
shot_focus = -1 # for graphs that need to set constant the number of shots, which index on the shot list to use

shot_graph = False # whether to graph RMSE as a function of shots
use_rotations = False # add an additional curve for CNOT-only with random rotations (shot and depolarization_bar graphs only)
prediction = False # whether to graph shot error predictions for RIDA and ZNE
depolarization = False # whether to graph depolarizing model
mitigated = True # whether to graph unmitigated and mitigated values against error-free expectations, with a perfect mitigation line
depolarization_bar = False # whether to plot a bar graph comparing depolarizations
paulis = False # whether to graph error as a function of true expectations of Pauli strings
seperate_plegend = True # whether to save Pauli graph legend as a seperate file
compare_layerwise = False # whether to graph layerwise results

save_results = True # whether to save graphs to seperate .png files
print_results = True # whether to send graphs to output

raw_color = '#7B3E3E'
zne_color = "#444A64"
cnot_color = '#91BE8D'
rida_color = "#FF4400"
layerwise_color='black'


## Import Libraries ##

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
    if compare_layerwise:
        layerwise_rmse = 0
    for j in range(num_strings): # add squared errors
        raw_rmse += (statevector[j] - raw[:,j])**2
        rida_rmse += (statevector[j] - rida[:,j])**2
        cnot_rmse += (statevector[j] - cnot[:,j])**2
        zne_rmse += (statevector[j] - zne[:,j])**2
        if compare_layerwise:
            layerwise_rmse += (statevector[j] - layerwise[:,j])**2

    # divide by number of samples and square root to find rmse
    raw_rmse = np.sqrt(raw_rmse/(num_strings))
    rida_rmse = np.sqrt(rida_rmse/(num_strings))
    cnot_rmse = np.sqrt(cnot_rmse/(num_strings))
    zne_rmse = np.sqrt(zne_rmse/(num_strings))
    if compare_layerwise:
        layerwise_rmse = np.sqrt(layerwise_rmse/(num_strings))
        return raw_rmse, rida_rmse, cnot_rmse, zne_rmse, layerwise_rmse
    return raw_rmse, rida_rmse, cnot_rmse, zne_rmse

# depolarization function for least-squares fit
def depo_func(x, a):
    return x/(1-a)

# determine the true (optimal) depolarizing approximation by minimizing the least-squares error
def optimal_depo():
    return curve_fit(depo_func, raw[shot_focus], statevector, [0])[0][0]


## Grapher ##

# graph RMSE as a function of shots
def graph_shots():
    fig, ax = plt.subplots(constrained_layout=True) # get figure for later operations
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(shots[0], shots[-1]) # cut blank edges off graph

    if use_rotations or not compare_layerwise:
        if error_multiplier != 0:
            plt.plot(shots, raw_rmse, linewidth=2, color=raw_color, label='Unmitigated')
        else:
            plt.plot(shots, raw_rmse, linewidth=2, color=raw_color, label='Unmitigated', linestyle='dashed', zorder=10)
    if use_rotations: # order of legend is different is using random rotations
        plt.plot(shots, zne_rmse, linewidth=2, color=zne_color, label='Exponential ZNE + TREX')
        plt.plot(shots, rida_rmse, linewidth=2, color=rida_color, label='RIDA')
        plt.plot(shots, cnot_rmse, linewidth=2, color=cnot_color, label='CNOT-Only Depolarization + Quadratic ZNE')
        plt.plot(shots, cnot_rotations_rmse, linewidth=2, color=cnot_color, label='CNOT-Only Depolarization + Quadratic ZNE\n(With Rotations)', linestyle='dashed')
        if compare_layerwise:
            plt.plot(shots, layerwise_rmse, linewidth=2, color=layerwise_color, label='Layerwise RIDA', linestyle='dashed')
    else:
        plt.plot(shots, rida_rmse, linewidth=2, color=rida_color, label='RIDA')
        if compare_layerwise:
            plt.plot(shots, layerwise_rmse, linewidth=2, color=layerwise_color, label='Layerwise RIDA', linestyle='dashed')
        else:
            plt.plot(shots, zne_rmse, linewidth=2, color=zne_color, label='Exponential ZNE + TREX')
            plt.plot(shots, cnot_rmse, linewidth=2, color=cnot_color, label='CNOT-Only Depolarization + Quadratic ZNE')   
    
    plt.xlabel('Shots', fontsize=16)
    plt.ylabel('RMSE', fontsize=16)
    fig.legend(loc='outside lower center', ncol=2, fontsize=10, frameon=False) # place the legend below the title
    if 'coherent' in file_name:
        fig.suptitle('Coherent: ' + str(nqubits) + ' qubits, ' + str(error_multiplier) + 'x error', fontsize=16)
    else:
        fig.suptitle(str(nqubits) + ' qubits, ' + str(error_multiplier) + 'x error', fontsize=16)
    for spine in ax.spines.values(): # make borders thicker
        spine.set_linewidth(2)
    if save_results:
        if use_rotations:
            if compare_layerwise:
                file_path = 'LAYERWISE AND ROTATIONS GRAPH - ' + str(nqubits) + ' qubits, ' + str(error_multiplier) + 'x error'
            else:
                file_path = 'ROTATIONS GRAPH - ' + str(nqubits) + ' qubits, ' + str(error_multiplier) + 'x error'
        else:
            if compare_layerwise:
                file_path = 'LAYERWISE GRAPH - ' + str(nqubits) + ' qubits, ' + str(error_multiplier) + 'x error'
            else:
                file_path = 'SHOTS GRAPH - ' + str(nqubits) + ' qubits, ' + str(error_multiplier) + 'x error'
        if 'coherent' in file_name:
            file_path = file_path + ', coherent'
        if os.path.exists(file_path + '.png'): # check if the file already exists
            file_path = file_path + str(time.time()) # add a timestamp to avoid overwriting existing file
        plt.savefig(file_path + '.png', dpi=500)
    if print_results:
        plt.show()
    else:
        plt.clf() # prevent figure from carrying over

# graph RMSE as a function of shots, including predictions
def graph_predictions():
    fig, ax = plt.subplots(constrained_layout=True) # get figure for later operations
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(shots[0], shots[-1]) # cut blank edges off graph

    # calculate expected exponential ZNE shot error scaling
    uu = np.mean(1-true_depo)**2
    zslope = []
    zslope.append( 1 + (uu+uu**(1/2))**(-1) + uu*(uu+uu**(1/2))**(-2)*(0.5*uu**(-1/2)+1) )
    zslope.append( -(uu+uu**(1/2))**(-1) + (uu+uu**(1/2))**(-2)*(0.5*uu**(-1/2)+1)*(uu-1) )
    zslope.append( (uu+uu**(1/2))**(-2)*(0.5*uu**(-1/2)+1) )
    zslope = np.sqrt(zslope[0]**2+zslope[1]**2+zslope[2]**2)

    plt.plot(shots, raw_rmse, linewidth=1, color=raw_color, label='Unmitigated', zorder=1)
    plt.plot(shots, rida_rmse, linewidth=1, color=rida_color, label='RIDA', zorder=4)
    plt.plot(shots, zne_rmse, linewidth=1, color=zne_color, label='Exponential ZNE + TREX', zorder=2)
    plt.plot(shots, [1/(np.sqrt(k)*np.mean(1-rida_depo)) for k in shots], linewidth=1, color=rida_color, label='RIDA Shot Prediction', zorder=4, linestyle='dashed') # theoretical shot error
    plt.plot(shots, [zslope/(np.sqrt(k/3)) for k in shots], linewidth=1, color=zne_color, label='Exponential ZNE\nShot Prediction', zorder=2, linestyle='dashed') # shots split among 3 circuits
    
    plt.xlabel('Shots', fontsize=16)
    plt.ylabel('RMSE', fontsize=16)
    if 'coherent' in file_name:
        fig.suptitle('Coherent: ' + str(nqubits) + ' qubits, ' + str(error_multiplier) + 'x error', fontsize=16)
    else:
        fig.suptitle(str(nqubits) + ' qubits, ' + str(error_multiplier) + 'x error', fontsize=16)
    fig.legend(loc='outside lower center', ncol=3, fontsize=10, frameon=False) # place the legend below the title
    for spine in ax.spines.values(): # make borders thicker
        spine.set_linewidth(2)
    if save_results:
        file_path = 'PREDICTIONS GRAPH - ' + str(nqubits) + ' qubits, ' + str(error_multiplier) + 'x error'
        if 'coherent' in file_name:
            file_path = file_path + ', coherent'
        if os.path.exists(file_path + '.png'): # check if the file already exists
            file_path = file_path + str(time.time()) # add a timestamp to avoid overwriting existing file
        plt.savefig(file_path + '.png', dpi=500)
    if print_results:
        plt.show()
    else:
        plt.clf() # prevent figure from carrying over


## Plot Depolarizations ##

# write the depolarization probability p of a depolarizing model just above its line, which has slope 1-p, with the text following the line
def label_depo_line(p, name, x, gap):
    value = '0' if p == 0 else f'{p:.2f}'
    plt.text(x, (1-p)*x + gap, r'$p_{\mathrm{' + name + r'}}=' + value + r'$', fontsize=12,
             rotation=np.degrees(np.arctan(1-p)), rotation_mode='anchor', transform_rotates_text=True, ha='center', va='bottom')

# graph depolarizing model: unmitigated vs. error-free expectation with trendlines comparing depolarizing models
def graph_depo():
    depo_label_x = [0.3, 0.65, -0.6] # x values for labels
    depo_label_gap = [0.03, 0.03, 0.07] # spacing between the lines and labels

    plt.scatter(statevector, raw[-1], color=raw_color, s=6, alpha=0.5, label='Unmitigated Data')
    plt.plot([-1, 1], [np.mean(rida_depo)-1, 1-np.mean(rida_depo)], color=rida_color, label='RIDA Model')
    plt.plot([-1, 1], [np.mean(cnot_depo)-1, 1-np.mean(cnot_depo)], color=cnot_color, label='CNOT-Only Depolarization Model')
    plt.plot([-1, 1], [-1, 1], color='black', alpha=0.5, label='No Model', linestyle='dashed')
    label_depo_line(0, r'no\ model', depo_label_x[0], depo_label_gap[0]) # no model is the same as a depolarization probability of 0
    label_depo_line(np.mean(cnot_depo), 'CNOT', depo_label_x[1], depo_label_gap[1])
    label_depo_line(np.mean(rida_depo), 'RIDA', depo_label_x[2], depo_label_gap[2])
    plt.xlabel(r'Error-Free Expectation $\langle O \rangle$', fontsize=16)
    plt.ylabel(r'Unmitigated Value $\langle O_{\epsilon} \rangle$', fontsize=16)
    plt.legend(loc='upper left', markerscale=3, fontsize=10, frameon=False)
    for spine in plt.gca().spines.values(): # make borders thicker
        spine.set_linewidth(2)
    plt.xlim(-1,1) # expectation values range from -1 to 1
    plt.ylim(-1,1) # expectation values range from -1 to 1
    if 'coherent' in file_name:
        plt.title('Determination of the Depolarization Probability\n' + 'Coherent: ' + str(nqubits) + ' qubits, ' +  rf'${2}^{{{int(np.log2(shots[shot_focus]))}}}$' + ' shots', fontsize=16)
    else:
        plt.title('Determination of the Depolarization Probability\n' + str(nqubits) + ' qubits, ' + str(error_multiplier) + 'x error, ' + rf'${2}^{{{int(np.log2(shots[shot_focus]))}}}$' + ' shots', fontsize=16)
    if save_results:
        file_path = 'DEPOLARIZATION GRAPH - ' + str(nqubits) + ' qubits, ' + str(error_multiplier) + 'x error'
        if 'coherent' in file_name:
            file_path = file_path + ', coherent'
        if os.path.exists(file_path + '.png'): # check if the file already exists
            file_path = file_path + str(time.time()) # add a timestamp to avoid overwriting existing file
        plt.savefig(file_path + '.png', dpi=500, bbox_inches='tight') # tight bounding box so the two-line title and the axis labels are not cut off
    if print_results:
        plt.show()
    else:
        plt.clf() # prevent figure from carrying over

# graph mitigation results: unmitigated and mitigated values vs. error-free expectation, where the dashed line is perfect mitigation
def graph_mitigated():
    fig, ax = plt.subplots(figsize=(6.4, 6.4), constrained_layout=True)
    ax.set_aspect('equal')
    plt.scatter(statevector, raw[shot_focus], color=raw_color, s=6, alpha=0.5, marker='o', label='Unmitigated', zorder=1)
    plt.scatter(statevector, zne[shot_focus], color=zne_color, s=6, alpha=0.5, marker='^', label='Exponential ZNE + TREX', zorder=2)
    plt.scatter(statevector, cnot[shot_focus], color=cnot_color, s=6, alpha=0.5, marker='x', label='CNOT-Only Depolarization + Quadratic ZNE', zorder=3)
    plt.scatter(statevector, rida[shot_focus], color=rida_color, s=6, alpha=0.5, marker='s', label='RIDA', zorder=4)
    plt.plot([-1, 1], [-1, 1], color='black', alpha=0.5, label='Perfect Mitigation', linestyle='dashed', zorder=5)
    plt.xlabel('Error-Free Expectation', fontsize=16)
    plt.ylabel('Mitigated Value', fontsize=16)
    plt.legend(loc='upper left', markerscale=3, fontsize=10, frameon=False)
    for spine in plt.gca().spines.values(): # make borders thicker
        spine.set_linewidth(2)
    plt.xlim(-1,1) # expectation values range from -1 to 1
    plt.ylim(-1,1) # expectation values range from -1 to 1
    if 'coherent' in file_name:
        plt.title('Coherent: ' + str(nqubits) + ' qubits, ' +  rf'${2}^{{{int(np.log2(shots[shot_focus]))}}}$' + ' shots', fontsize=16)
    else:
        plt.title(str(nqubits) + ' qubits, ' + str(error_multiplier) + 'x error, ' + rf'${2}^{{{int(np.log2(shots[shot_focus]))}}}$' + ' shots', fontsize=16)
    if save_results:
        file_path = 'MITIGATED GRAPH - ' + str(nqubits) + ' qubits, ' + str(error_multiplier) + 'x error'
        if 'coherent' in file_name:
            file_path = file_path + ', coherent'
        if os.path.exists(file_path + '.png'): # check if the file already exists
            file_path = file_path + str(time.time()) # add a timestamp to avoid overwriting existing file
        plt.savefig(file_path + '.png', dpi=500)
    if print_results:
        plt.show()
    else:
        plt.clf() # prevent figure from carrying over

# compare depolarization magnitudes in a bar graph
def graph_depo_bar():
    fig, ax = plt.subplots(constrained_layout=True)
    plt.axhline(y=true_depo, color='black', label='Optimal', linestyle='dashed')
    plt.bar('No rotations', np.mean(cnot_depo), color=cnot_color, label='CNOT-Only Depolarization')
    if use_rotations:
        plt.bar('Rotations', np.mean(cnot_rotations_depo), color=cnot_color, label='CNOT-Only Depolarization (With Rotations)', hatch='///')
    plt.bar('RIDA', np.mean(rida_depo), color=rida_color, label='RIDA')

    plt.xlabel('Method', fontsize=16)
    plt.ylabel('Depolarization Estimate p' ,fontsize=16)
    if 'coherent' in file_name:
        plt.title('Coherent: ' + str(nqubits) + ' qubits, ' + str(error_multiplier) + 'x error', fontsize=16)
    else:
        plt.title(str(nqubits) + ' qubits, ' + str(error_multiplier) + 'x error', fontsize=16)
    for spine in plt.gca().spines.values():
        spine.set_linewidth(2)
    fig.legend(loc='outside lower center', fontsize=10, frameon=False, ncol=2)
    if save_results:
        file_path = 'BAR DEPOLARIZATION GRAPH - ' + str(nqubits) + ' qubits, ' + str(error_multiplier) + 'x error'
        if 'coherent' in file_name:
            file_path = file_path + ', coherent'
        if os.path.exists(file_path + '.png'): # check if the file already exists
            file_path = file_path + str(time.time()) # add a timestamp to avoid overwriting existing file
        plt.savefig(file_path + '.png', dpi=500)
    if print_results:
        plt.show()
    else:
        plt.clf() # prevent figure from carrying over

# graph error as a function of error-free expectations of Pauli strings
def graph_pauli():
    # compute errors for each Pauli string (not RMSE across Pauli strings)
    raw_error = np.abs(raw - statevector)[shot_focus]
    rida_error = np.abs(rida - statevector)[shot_focus]
    cnot_error = np.abs(cnot - statevector)[shot_focus]
    zne_error = np.abs(zne - statevector)[shot_focus]
    fig, ax = plt.subplots(constrained_layout=True)

    height_cutoff = 1.1 # y limit of the graph - some outliers may exceed this
    base_size = 8 # size of points
    size_factor = 500/num_strings # make bigger dots when there are less points
    size = base_size*size_factor

    plt.scatter(statevector, raw_error, color=raw_color, label='Unmitigated', s=size, alpha=0.6, marker='o', zorder=1)
    plt.scatter(statevector, rida_error, color=rida_color, label='RIDA', s=size, alpha=0.6, marker='s', zorder=4)
    plt.scatter(statevector, zne_error, color=zne_color, label='Exponential ZNE + TREX', s=size, alpha=0.6, marker='^', zorder=2)
    plt.scatter(statevector, cnot_error, color=cnot_color, label='CNOT-Only Depolarization + Quadratic ZNE', s=size, alpha=0.6, marker='x', zorder=3)
    plt.xlabel('Error-Free Expectation', fontsize=17)
    plt.ylabel('Error', fontsize=17)
    plt.xlim(-1,1)
    plt.ylim(0, height_cutoff) # bottom cannot be zero if logscale
    for spine in ax.spines.values():
        spine.set_linewidth(2)
    if 'coherent' in file_name:
        fig.suptitle('Coherent Error: ' + str(nqubits) + ' qubits, ' + rf'${2}^{{{int(np.log2(shots[shot_focus]))}}}$' + ' shots', fontsize=17)
    else:
        fig.suptitle('Incoherent Error: ' + str(nqubits) + ' qubits, ' + str(error_multiplier) + 'x error, ' + rf'${2}^{{{int(np.log2(shots[shot_focus]))}}}$' + ' shots', fontsize=17)
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
        file_path = 'PAULI GRAPH - ' + str(nqubits) + ' qubits, ' + str(error_multiplier) + 'x error'
        if 'coherent' in file_name:
            file_path = file_path + ', coherent'
        if os.path.exists(file_path + '.png'): # check if the file already exists
            file_path = file_path + str(time.time()) # add a timestamp to avoid overwriting existing file
        plt.savefig(file_path + '.png', dpi=500)
    plt.ylim(0,height_cutoff)
    if print_results:
        plt.show()
    else:
        plt.clf() # prevent figure from carrying over


## Driver ##

# load relevant variables from a dictionary of a single run
with open(file_name, 'rb') as f:
    dict = pickle.load(f)
nqubits, error_multiplier, shots, rida_depo, cnot_depo, statevector, raw, rida, cnot, zne, operators = \
dict['nqubits'], dict['error_multiplier'], dict['shots'], dict['rida_depo'], dict['cnot_depo'], \
dict['statevector'], dict['raw'], dict['rida'], dict['cnot'], dict['zne'], dict['operators']
if compare_layerwise:
    layerwise = dict['layerwise']
num_strings = len(operators)
if use_rotations and (shot_graph or depolarization_bar): # add RMSE from CNOT-only with random rotations
    with open(file_name[:-4] + ', rotations.pkl', 'rb') as f:
        new_dict = pickle.load(f)
        cnot_rotations = new_dict['cnot']
        cnot_rotations_depo = new_dict['cnot_depo']
        new_num_strings = len(new_dict['operators'])
        cnot_rotations_rmse = 0
        for j in range(num_strings):
            cnot_rotations_rmse += (statevector[j] - cnot_rotations[:,j])**2
        cnot_rotations_rmse = np.sqrt(cnot_rotations_rmse/new_num_strings)

# create graphs
if shot_graph or prediction:
    if compare_layerwise:
        raw_rmse, rida_rmse, cnot_rmse, zne_rmse, layerwise_rmse = rmse()
    else:
        raw_rmse, rida_rmse, cnot_rmse, zne_rmse = rmse()
if prediction or depolarization_bar:
    true_depo = optimal_depo()
if shot_graph:
    graph_shots()
if prediction:
    graph_predictions()
if depolarization:
    graph_depo()
if mitigated:
    graph_mitigated()
if paulis:
    graph_pauli()
if depolarization_bar:
    graph_depo_bar()