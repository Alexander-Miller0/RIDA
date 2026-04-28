#!/usr/bin/env python
# coding: utf-8


## Import Libraries ##

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
import os
import time
from scipy.optimize import minimize
plt.ioff() # prevents plots from showing automatically


## Settings ##

shot_bounds = [2**10, 2**20] # bounds on the graphed numbers of shots
depo_bounds = [0, 1] # bounds on the graphed depolarization error probability
npoints = 2000 # number of depolarization probabilities sampled

save_results = True # whether to save graphs to seperate .png files
print_results = False # whether to send graphs to output

raw_color = '#7B3E3E'
rida_color = '#FF4400'


## Setup ##

depo_domain = np.linspace(depo_bounds[0], depo_bounds[1], npoints) # list of depolarization values as input points

# boundary between RIDA and unmitigated
def cutoff(pp):
    if pp == 1:
        pp -= 1/(npoints)**2 # small perturbation to avoid division by zero
    return max(pp*(2+2*pp-pp**2)/((2-pp)*(1-pp)**2), 1)

# objective function for finding value of p at the minimum number of shots
def obj_fun(pp):
    return (cutoff(pp) - shot_bounds[0])**2


## Driver ##

fig, ax = plt.subplots()

# plot cutoff functions and fill appropriate colors inbetween
yvals = [cutoff(depo) for depo in depo_domain]
ax.fill_between(depo_domain, yvals, shot_bounds[1], color=rida_color, label='RIDA', linewidth=0)
ax.fill_between(depo_domain, shot_bounds[0], yvals, color=raw_color, label='Unmitigated', linewidth=0)

# set axes and legend
ax.set_yscale('log')
ax.set_ylim(shot_bounds[0], shot_bounds[1])
ax.set_xlim(depo_bounds[0], depo_bounds[1])
ax.legend(loc='upper left')
ax.set_xlabel('Depolarization Probability', fontsize=16)
ax.set_ylabel('Shots', fontsize=16)

# create inset
intersection = minimize(obj_fun, [0.9]).x[0]
ax_inset = inset_axes(ax, width='50%', height='50%', loc='upper right', borderpad=2)
ax_inset.fill_between(depo_domain, yvals, shot_bounds[1], color=rida_color, label='RIDA', linewidth=0)
ax_inset.fill_between(depo_domain, shot_bounds[0], yvals, color=raw_color, label='Unmitigated', linewidth=0)
ax_inset.set_yscale('log')
ax_inset.set_ylim(shot_bounds[0], shot_bounds[1])
ax_inset.set_xlim(intersection, depo_bounds[1])
ax_inset.tick_params(axis='both', labelsize=7)
ax_inset.set_xlabel('Depolarization Probability (' + str(int(1/(1-intersection)+0.5)) + 'x scale)', fontsize=10)
ax_inset.set_ylabel('Shots', fontsize=10)

# Create the shadow rectangle behind the inset axes
ax.add_patch(plt.Rectangle( (0.46, 0.45), 0.493, 0.489, fc='black', alpha=0.3, transform=ax.transAxes, zorder=1))

for spine in ax.spines.values():
    spine.set_linewidth(2)
for spine in ax_inset.spines.values():
    spine.set_linewidth(2)

if save_results:
    file_path = 'THEORY GRAPH'
    if os.path.exists(file_path + '.png'): # check if the file already exists
        file_path = file_path + str(time.time()) # add a timestamp to avoid overwriting existing file
    plt.savefig(file_path + '.png', dpi=500)
if print_results:
    plt.show()