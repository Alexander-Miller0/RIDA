#!/usr/bin/env python
# coding: utf-8


## Settings ##

shot_focus = -1 # what index of shots to use for graphs that only use one set number of shots

shot_graph = True # whether to graph RMSE as a function of shots
prediction = False # whether to graph shot error predictions for RIDA and ZNE
multipliers = True # whether to graph RMSE as a function of error multiplier
depolarization = False # whether to graph depolarization model
depolarization_summary = False # whether to graph depolarizing rates as a function of error multiplier
paulis = False # whether to graph error as a function of error-free expectations of Pauli strings
compress = True # whether to graph a subset of graphs for multiplier and shot graphs - if both are graphed, this also places their shared legend in a seperate graph
truncate = True # whether to truncate points on Pauli graph

save_results = True # whether to save graphs to seperate .png files
print_results = False # whether to send graphs to output

raw_color = '#7B3E3E'
zne_color = '#444A64'
cnot_color = '#91BE8D'
rida_color = '#FF4400'

# 2D list of input files - do not edit unless qubit numbers or error multipliers are changed
file_name = [['RESULTS - 4 qubits, 1x error.pkl',
             'RESULTS - 4 qubits, 2x error.pkl',
             'RESULTS - 4 qubits, 3x error.pkl',
             'RESULTS - 4 qubits, 4x error.pkl'],
             ['RESULTS - 5 qubits, 1x error.pkl',
             'RESULTS - 5 qubits, 2x error.pkl',
             'RESULTS - 5 qubits, 3x error.pkl',
             'RESULTS - 5 qubits, 4x error.pkl'],
             ['RESULTS - 6 qubits, 1x error.pkl',
             'RESULTS - 6 qubits, 2x error.pkl',
             'RESULTS - 6 qubits, 3x error.pkl',
             'RESULTS - 6 qubits, 4x error.pkl'],
             ['RESULTS - 7 qubits, 1x error.pkl',
             'RESULTS - 7 qubits, 2x error.pkl',
             'RESULTS - 7 qubits, 3x error.pkl',
             'RESULTS - 7 qubits, 4x error.pkl']]

## Import Libraries ##

import pickle
import matplotlib.pyplot as plt
import numpy as np
import os
import time
from scipy.optimize import curve_fit

plt.ioff() # prevents plots from showing automatically


## Helper Functions ##

# calculate the RMSE across all Pauli strings
def rmse(dict):
    raw_rmse = 0
    rida_rmse = 0
    cnot_rmse = 0
    zne_rmse = 0
    guess_rmse = 0
    for j in range(len(dict['operators'])): # add squared errors
        raw_rmse += (dict['statevector'][j] - dict['raw'][:,j])**2
        rida_rmse += (dict['statevector'][j] - dict['rida'][:,j])**2
        cnot_rmse += (dict['statevector'][j] - dict['cnot'][:,j])**2
        zne_rmse += (dict['statevector'][j] - dict['zne'][:,j])**2
        guess_rmse += dict['statevector'][j]**2 # guess rmse is the rmse compared to 0
    
    # divide by number of samples and square root to find rmse
    dict['raw_rmse'] = np.sqrt(raw_rmse/(len(dict['operators'])))
    dict['rida_rmse'] = np.sqrt(rida_rmse/(len(dict['operators'])))
    dict['cnot_rmse'] = np.sqrt(cnot_rmse/(len(dict['operators'])))
    dict['zne_rmse'] = np.sqrt(zne_rmse/(len(dict['operators'])))
    dict['guess_rmse'] = np.sqrt(guess_rmse/(len(dict['operators'])))
    return dict # return rmses as new elements in input dictionary

# depolarization function for least-squares fit
def depo_func(x, a):
    return x/(1-a)

# determine the true (optimal) depolarizing approximation by minimizing the least-squares error
def true_depo(dict):
    dict['true_depo'] = curve_fit(depo_func, dict['raw'][shot_focus], dict['statevector'], [0])[0][0]
    return dict


## Grapher ##

# graph RMSE as a function of shots
def graph_shots(dict, ax):
    
    ax.plot(dict['shots'], dict['raw_rmse'], linewidth=1+compress, color=raw_color, label='Unmitigated')
    ax.plot(dict['shots'], dict['rida_rmse'], linewidth=1+compress, color=rida_color, label='RIDA')
    ax.plot(dict['shots'], dict['zne_rmse'], linewidth=1+compress, color=zne_color, label='Exponential ZNE + TREX')
    ax.plot(dict['shots'], dict['cnot_rmse'], linewidth=1+compress, color=cnot_color, label='CNOT-Only Depolarization + Quadratic ZNE')
    ax.set_title(str(dict['nqubits']) + ' qubits, ' + str(dict['error_multiplier']) + 'x error', fontsize=8+12*compress)
    ax.set_xscale('log')
    ax.set_yscale('log')

    ax.set_xlim(dict['shots'][0], dict['shots'][-1]) # cut blank edges off graph

    ax.tick_params(axis='both', labelsize=7+9*compress)
    for spine in ax.spines.values(): # make borders thicker
        spine.set_linewidth(2)

# graph RMSE as a function of shots, including predictions
def graph_predictions(dict, ax):
    # calculate expected exponential ZNE shot error scaling
    uu = np.mean(1-dict['true_depo'])**2
    zslope = []
    zslope.append( 1 + (uu+uu**(1/2))**(-1) + uu*(uu+uu**(1/2))**(-2)*(0.5*uu**(-1/2)+1) )
    zslope.append( -(uu+uu**(1/2))**(-1) + (uu+uu**(1/2))**(-2)*(0.5*uu**(-1/2)+1)*(uu-1) )
    zslope.append( (uu+uu**(1/2))**(-2)*(0.5*uu**(-1/2)+1) )
    zslope = np.sqrt(3*(zslope[0]**2+zslope[1]**2+zslope[2]**2))
    
    ax.plot(dict['shots'], dict['raw_rmse'], linewidth=1, color=raw_color, label='Unmitigated')
    ax.plot(dict['shots'], dict['rida_rmse'], linewidth=1, color=rida_color, label='RIDA')
    ax.plot(dict['shots'], dict['zne_rmse'], linewidth=1, color=zne_color, label='Exponential ZNE + TREX')
    ax.plot(dict['shots'], [1/(np.sqrt(k)*np.mean(1-dict['rida_depo'])) for k in dict['shots']], linewidth=1, color=rida_color, label='RIDA Shot Prediction', zorder=4, linestyle='dashed') # theoretical shot error
    ax.plot(dict['shots'], [zslope/(np.sqrt(k)) for k in dict['shots']], linewidth=1, color=zne_color, label='Exponential ZNE Shot Prediction', zorder=2, linestyle='dashed')
    ax.set_title(str(dict['nqubits']) + ' qubits, ' + str(dict['error_multiplier']) + 'x error', fontsize=8)
    ax.set_xscale('log')
    ax.set_yscale('log')

    ax.set_xlim(dict['shots'][0], dict['shots'][-1]) # cut blank edges off graph

    ax.tick_params(axis='both', labelsize=7)
    for spine in ax.spines.values(): # make borders thicker
        spine.set_linewidth(2)

# graph RMSE as a function of error multiplier, and show convergence behavior to the maximally mixed state
def graph_error_multiplier(dict, ax): 
    # pull values from input dictionaries
    error_multipliers = [dict[k]['error_multiplier'] for k in range(len(dict))]
    guess_rmse = [dict[k]['guess_rmse'] for k in range(len(dict))]
    raw_rmse = [dict[k]['raw_rmse'][shot_focus] for k in range(len(dict))]
    zne_rmse = [dict[k]['zne_rmse'][shot_focus] for k in range(len(dict))]
    cnot_rmse = [dict[k]['cnot_rmse'][shot_focus] for k in range(len(dict))]
    rida_rmse = [dict[k]['rida_rmse'][shot_focus] for k in range(len(dict))]

    upper_bound = 1.05*guess_rmse[0]
    width = 1 + compress

    ax.plot(error_multipliers, raw_rmse, linewidth=width, color=raw_color, label='Unmitigated')
    ax.plot(error_multipliers, rida_rmse, linewidth=width, color=rida_color, label='RIDA')
    ax.plot(error_multipliers, zne_rmse, linewidth=width, color=zne_color, label='Exponential ZNE + TREX')
    if compress:
        ax.plot(error_multipliers, cnot_rmse, linewidth=width, color=cnot_color, label='CNOT-Only Depolarization\n+ Quadratic ZNE')
    else:
        ax.plot(error_multipliers, cnot_rmse, linewidth=width, color=cnot_color, label='CNOT-Only Depolarization + Quadratic ZNE')
    last_within = 0 # find the last point for CNOT-only depolarization within the graph bounds
    while last_within < len(dict) and cnot_rmse[last_within] < upper_bound:
        last_within += 1
    if last_within < len(dict): # add arrow if CNOT-only depolarization line goes out of bounds
        last_within -= 1 # account for overshoot of 1
        slope = (cnot_rmse[last_within+1] - cnot_rmse[last_within]) # find the slope of the line that exceeds bounds
        xpos = last_within + 1 + (upper_bound - cnot_rmse[last_within])/slope # find the x position for CNOT-only arrow
        ax.annotate('', xy=(xpos, upper_bound), xytext=(xpos-0.01, upper_bound-0.01*slope), arrowprops={'arrowstyle': '->', 'shrinkA': 0, 'shrinkB':0, 'lw': width, 'color': cnot_color})
    ax.plot(error_multipliers, guess_rmse, linewidth=width, color='black', linestyle='dashed', label='Maximally Mixed')
    if compress:
        ax.set_title(str(dict[0]['nqubits']) + ' qubits, ' + rf'${2}^{{{int(np.log2(dict[0]['shots'][shot_focus]))}}}$' + ' shots', fontsize=14)
    else:
        ax.set_title(str(dict[0]['nqubits']) + ' qubits', fontsize=12)
    ax.set_xlim(error_multipliers[0], error_multipliers[-1]) # cut blank edges off graph
    ax.set_ylim(0, upper_bound) # cut off beyond maximally mixed state
    ax.set_xticks(error_multipliers)
    ax.tick_params(axis='both', labelsize=7+4*compress)
    for spine in ax.spines.values(): # make borders thicker
        spine.set_linewidth(2)

# graph depolarizing model: unmitigated vs. error-free expectation with trendlines comparing depolarizing models
def graph_depo(dict, ax):
    ax.scatter(dict['statevector'], dict['raw'][shot_focus], color=raw_color, s=1, alpha=0.5, label='Unmitigated Data')
    ax.plot([-1, 1], [np.mean(dict['rida_depo'])-1, 1-np.mean(dict['rida_depo'])], color=rida_color, label='RIDA Model', linewidth=0.6)
    ax.plot([-1, 1], [np.mean(dict['cnot_depo'])-1, 1-np.mean(dict['cnot_depo'])], color=cnot_color, label='CNOT-Only Depolarization Model', linewidth=0.6)
    ax.plot([-1, 1], [-1, 1], color='black', alpha=0.5, label='No Model', linewidth=0.6, linestyle='dashed')
    ax.set_title(str(dict['nqubits']) + ' qubits, ' + str(dict['error_multiplier']) + 'x error', fontsize=8)
    ax.set_xlim(-1,1)
    ax.set_ylim(-1,1)
    ax.tick_params(axis='both', labelsize=7)
    for spine in ax.spines.values(): # make borders thicker
        spine.set_linewidth(2)

def graph_depo_summary(dict, ax):
    # pull values from input dictionaries
    error_multipliers = [dict[k]['error_multiplier'] for k in range(len(dict))]
    true_depos = [1/(1-dict[k]['true_depo']) for k in range(len(dict))] # calculate coefficient 1/(1-p)
    rida_depos = [1/np.mean(1-dict[k]['rida_depo']) for k in range(len(dict))]
    cnot_depos = [1/np.mean(1-dict[k]['cnot_depo']) for k in range(len(dict))]

    ax.plot(error_multipliers, true_depos, linewidth=2, color='black', label='Optimal', linestyle='dashed', zorder=1)
    ax.plot(error_multipliers, rida_depos, linewidth=2, color=rida_color, label='RIDA', zorder=0)
    ax.plot(error_multipliers, cnot_depos, linewidth=2, color=cnot_color, label='CNOT-Only Depolarization', zorder=0)
    ax.set_title(str(dict[0]['nqubits']) + ' qubits', fontsize=10)
    ax.set_xlim(error_multipliers[0], error_multipliers[-1]) # cut blank edges off graph
    ax.set_ylim(0, upper_ds_bound) # set consistent upper bound
    ax.set_xticks(error_multipliers)
    ax.tick_params(axis='both', labelsize=7)
    for spine in ax.spines.values(): # make borders thicker
        spine.set_linewidth(2)

# graph error as a function of error-free expectations of Pauli strings
def graph_pauli(dict, ax):
    # compute errors for each Pauli string (not RMSE across Pauli strings)
    raw_error = np.abs(dict['raw'] - dict['statevector'])[shot_focus]
    zne_error = np.abs(dict['zne'] - dict['statevector'])[shot_focus]
    cnot_error = np.abs(dict['cnot'] - dict['statevector'])[shot_focus]
    rida_error = np.abs(dict['rida'] - dict['statevector'])[shot_focus]
    
    ax.scatter(dict['statevector'], raw_error, color=raw_color, label='Unmitigated', s=1, alpha=0.5, marker='o')
    ax.scatter(dict['statevector'], zne_error, color=zne_color, label='Exponential ZNE + TREX', s=1, alpha=0.5, marker='^')
    ax.scatter(dict['statevector'], cnot_error, color=cnot_color, label='CNOT-Only Depolarization + Quadratic ZNE', s=1, alpha=0.5, marker='x')
    ax.scatter(dict['statevector'], rida_error, color=rida_color, label='RIDA', s=1, alpha=0.5, marker='s')
    ax.set_title(str(dict['nqubits']) + ' qubits, ' + str(dict['error_multiplier']) + 'x error', fontsize=10)
    ax.set_xlim(-1, 1) # error-free expectations range from -1 to 1
    if truncate:
        ax.set_ylim(0, min(1.1, max(np.max(raw_error), np.max(zne_error), np.max(cnot_error)))) # cut off y axis at 1.1 if points exceed it
    else:
        ax.set_ylim(bottom=0)
    ax.tick_params(axis='both', labelsize=7)
    for spine in ax.spines.values(): # make borders thicker
        spine.set_linewidth(2)


## Driver ##

with open(file_name[0][0], 'rb') as f:
    ndiff_shots = len(pickle.load(f)['shots']) # count the number of different shots in order to create error arrays

# create a 2D array of dictionaries for each input file
dict = []
for q in range(len(file_name)):
    dict.append([0]*len(file_name[q]))
    for i in range(len(file_name[q])): # upload each individual dictionary
        with open(file_name[q][i], 'rb') as f:
            dict[q][i] = pickle.load(f)
        dict[q][i] = rmse(dict[q][i])
        dict[q][i] = true_depo(dict[q][i])

# determine minimum and maximum value to create consistent bounds across graphs
upper_ds_bound = 0 # upper bound for depolarizing factor summary graphs
for q in range(len(file_name)): # iterate over different numbers of qubits
    for i in range(len(file_name[q])): # iterate over different error multipliers
        # update bounds for predicted shot errors
        uu = np.mean(1-dict[q][i]['true_depo'])**2
        zslope = []
        zslope.append( 1 + (uu+uu**(1/2))**(-1) + uu*(uu+uu**(1/2))**(-2)*(0.5*uu**(-1/2)+1) )
        zslope.append( -(uu+uu**(1/2))**(-1) + (uu+uu**(1/2))**(-2)*(0.5*uu**(-1/2)+1)*(uu-1) )
        zslope.append( (uu+uu**(1/2))**(-2)*(0.5*uu**(-1/2)+1) )
        zslope = np.sqrt(zslope[0]**2+zslope[1]**2+zslope[2]**2)

        # update bounds for depolarizing factor
        upper_ds_bound = max( upper_ds_bound, 1/(1-dict[q][i]['true_depo']), 1/np.mean(1-dict[q][i]['rida_depo']), 1/np.mean(1-dict[q][i]['cnot_depo']))

if shot_graph:
    if compress: # only graph smallest and largest numbers of qubits and error multipliers
        fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 7), constrained_layout=True) # create overarching figure
        for i in (0, -1):
            for j in (0, -1):
                graph_shots(dict[i][j], axs[i][j]) # graph each subplot
        handles, labels = axs[0][0].get_legend_handles_labels()
        if not multipliers or not compress or not save_results:
            fig.legend(handles, labels, loc='outside upper center', ncol=2, fontsize=16, frameon=False)
        fig.supxlabel('Shots', fontsize=24)
        fig.supylabel('RMSE', fontsize=24)
        file_path = 'BATCH GRAPH - Shots, compressed'
    else:
        fig, axs = plt.subplots(nrows=len(file_name), ncols=len(file_name[0]), figsize=(10, 8), constrained_layout=True) # create overarching figure
        for i in range(len(file_name)):
            for j in range(len(file_name[0])):
                graph_shots(dict[i][j], axs[i][j]) # graph each subplot
        handles, labels = axs[0][0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='outside upper center', ncol=4, fontsize=11, frameon=False)
        file_path = 'BATCH GRAPH - Shots'
        fig.supxlabel('Shots')
        fig.supylabel('RMSE')
    if save_results:
        if os.path.exists(file_path + '.png'): # check if the file already exists
            file_path = file_path + str(time.time()) # add a timestamp to avoid overwriting existing file
        plt.savefig(file_path + '.png', dpi=500)
    if print_results:
        plt.show()

if prediction:
    fig, axs = plt.subplots(nrows=len(file_name), ncols=len(file_name[0]), figsize=(10, 8), constrained_layout=True) # create overarching figure
    for i in range(len(file_name)):
        for j in range(len(file_name[0])):
            graph_predictions(dict[i][j], axs[i][j]) # graph each subplot
    fig.supxlabel('Shots')
    fig.supylabel('RMSE')
    handles, labels = axs[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside upper center', ncol=3, fontsize=11, frameon=False)
    if save_results:
        file_path = 'BATCH GRAPH - Predictions'
        if os.path.exists(file_path + '.png'): # check if the file already exists
            file_path = file_path + str(time.time()) # add a timestamp to avoid overwriting existing file
        plt.savefig(file_path + '.png', dpi=500)
    if print_results:
        plt.show()

if multipliers:
    if compress: # only graph largest number of qubits
        fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
        graph_error_multiplier(dict[-1], ax)
        handles, labels = ax.get_legend_handles_labels()
        if shot_graph and save_results and compress:
            fig_legend = plt.figure(figsize=(5, 0.1))
            legend = fig_legend.legend(handles, labels, ncol=5, loc='center', frameon=False)
            fig_legend.gca().set_axis_off()
            file_path = 'LEGEND - multipliers'
            if os.path.exists(file_path + '.png'): # check if the file already exists
                file_path = file_path + str(time.time()) # add a timestamp to avoid overwriting existing file
            fig_legend.savefig(file_path + '.png', bbox_inches='tight', dpi=500)
            plt.close(fig_legend)
        else:
            fig.legend(handles, labels, loc='outside upper center', ncol=3, fontsize=11, frameon=False)
        file_path = 'BATCH GRAPH - Error multipliers, compressed'
        fig.supxlabel('Error Multiplier', fontsize=14)
        fig.supylabel('RMSE', fontsize=14)
    else:
        fig, axs = plt.subplots(ncols=4, figsize=(10, 2.5), constrained_layout=True) # create overarching figure
        for i in range(len(file_name)):
            graph_error_multiplier(dict[i], axs[i]) # graph each subplot
        handles, labels = axs[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='outside upper center', ncol=5, fontsize=8, frameon=False)
        file_path = 'BATCH GRAPH - Error multipliers'
        fig.supxlabel('Error Multiplier')
        fig.supylabel('RMSE')
    if save_results:
        if os.path.exists(file_path + '.png'): # check if the file already exists
            file_path = file_path + str(time.time()) # add a timestamp to avoid overwriting existing file
        plt.savefig(file_path + '.png', dpi=500)
    if print_results:
        plt.show()

if depolarization:
    fig, axs = plt.subplots(nrows=len(file_name), ncols=len(file_name[0]), figsize=(10, 8), constrained_layout=True) # create overarching figure
    for i in range(len(file_name)):
        for j in range(len(file_name[i])):
            graph_depo(dict[i][j], axs[i][j]) # graph each subplot
    fig.supxlabel('Error-Free Expectation')
    fig.supylabel('Unmitigated Value')
    handles, labels = axs[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside upper center', ncol=4, fontsize=11, markerscale=5, frameon=False)
    if save_results:
        file_path = 'BATCH GRAPH - Depolarizations'
        if os.path.exists(file_path + '.png'): # check if the file already exists
            file_path = file_path + str(time.time()) # add a timestamp to avoid overwriting existing file
        plt.savefig(file_path + '.png', dpi=500)
    if print_results:
        plt.show()

if depolarization_summary:
    fig, axs = plt.subplots(ncols=4, figsize=(10, 2.5), constrained_layout=True) # create overarching figure
    for i in range(len(file_name)):
        graph_depo_summary(dict[i], axs[i]) # graph each subplot
    fig.supxlabel('Error Multiplier')
    fig.supylabel('1/(1-p)')
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside upper center', ncol=3, fontsize=11, frameon=False)
    if save_results:
        file_path = 'BATCH GRAPH - Depolarization Summaries'
        if os.path.exists(file_path + '.png'): # check if the file already exists
            file_path = file_path + str(time.time()) # add a timestamp to avoid overwriting existing file
        plt.savefig(file_path + '.png', dpi=500)
    if print_results:
        plt.show()

if paulis:
    fig, axs = plt.subplots(nrows=len(file_name), ncols=len(file_name[0]), figsize=(10, 8), constrained_layout=True) # create overarching figure
    for i in range(len(file_name)):
        for j in range(len(file_name[i])):
            graph_pauli(dict[i][j], axs[i][j]) # graph each subplot
    fig.supxlabel('Error-Free Expectation')
    fig.supylabel('Error')
    if truncate:
        handles, labels = axs[0][0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='outside upper center', ncol=4, fontsize=11, markerscale=5, frameon=False)
    else:
        fig.suptitle('No Truncation')
    if save_results:
        if truncate:
            file_path = 'BATCH GRAPH - Paulis'
        else:
            file_path = 'BATCH GRAPH - Paulis (no truncation)'
        if os.path.exists(file_path + '.png'): # check if the file already exists
            file_path = file_path + str(time.time()) # add a timestamp to avoid overwriting existing file
        plt.savefig(file_path + '.png', dpi=500)
    if print_results:
        plt.show()