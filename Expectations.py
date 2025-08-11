#!/usr/bin/env python
# coding: utf-8


## Settings ##

num_strings = 2 # number of Pauli strings to generate
nqubits = 4 # number of qubits in ansatz
layers = 12 # number of layers in ansatz - assumed to be 12 by default in RIDA.py

base_qubits = 4 # if running on HTC, smallest number of qubits for test case generation

save_results = True # whether to save results to a pickle file
progress_flags = True # print checkpoints during testing
print_results = True # print time taken and RMSE of target vs. true expectations


## Import Libraries ##

import numpy as np
import time
import pickle
from qiskit.circuit.library import EfficientSU2
from scipy.optimize import minimize
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector
import sys
import os

rng = np.random.default_rng()


## Helper Functions ##

# determine the noiseless expectation of a circuit
def sv_expectation(circ):
    sump = 0.0
    probs = Statevector(circ).probabilities(measurements)
    for i in range(len(probs)):
        if sum(int(digit) for digit in bin(i)[2:])%2 == 1: #check if it results in a 1 on the classical bit
            sump += probs[i]
    return 1 - 2*sump # the expectation value of an operator is 1-2p(1)

# objective function to optimize the true expectation to the target
def obj_fun(params):
    return (sv_expectation(ansatz.assign_parameters(params).compose(circ)) - target)**2


## Driver ##

# import settings from .sub file - ignore if not using
try:
    nqubits = int(sys.argv[1])+base_qubits
    progress_flags = False
except:
    pass

ansatz = EfficientSU2(nqubits, reps=layers, entanglement='full', skip_unentangled_qubits=False, parameter_prefix='a')
nparams = ansatz.num_parameters

param_list = []
pauli_list = []
rmse = 0 # root mean squared error of the expectations vs. target - negligible and not important since targets are random

start = time.time()
if progress_flags:
    print('-------Generating Paulis-------')
for i in range(num_strings):
    target = 2*rng.random()-1
    circ = QuantumCircuit(nqubits)
    measurements = []
    pstring = ''
    identity = True
    while identity: # re-generate if the Pauli string is all identity
        pstring = ''
        for j in range(nqubits):
            option = rng.random()
            if option < 0.25:
                pstring = pstring + 'I'
            elif option < 0.5:
                pstring = pstring + 'X'
                identity = False
                measurements.append(j)
                circ.h(j)
            elif option < 0.75:
                pstring = pstring + 'Y'
                identity = False
                measurements.append(j)
                circ.s(j)
                circ.h(j)
            else:
                pstring = pstring + 'Z'
                identity = False
                measurements.append(j)
    res = minimize(obj_fun, 2*np.pi*np.random.rand(nparams)-np.pi) # find parameters that create the target expectation
    param_list.append(res['x'])
    pauli_list.append(pstring[::-1])
    if print_results:
        rmse += obj_fun(res['x'])
    if progress_flags:
        print('Pauli String ', i+1, '/', num_strings, ' completed')
        rem_time = (time.time() - start) * (num_strings - i - 1) / (i + 1)
        print('Estimated time remaining: ', int(rem_time/3600), 'hours,', int(rem_time/60%60), 'minutes')

if progress_flags:
    rmse = np.sqrt(rmse/num_strings)
    print()
    print('----Data Generation Complete----')
    print()
if print_results:
    print('RMSE: ', rmse)
    print()
    time_taken = time.time() - start
    print('Time taken: ', int(time_taken/3600), 'hours,', int(time_taken/60%60), 'minutes')

if save_results:
    file_path = str(nqubits) + 'qbt_params'
    if os.path.exists(file_path + '.pkl'): # check if the file already exists
        file_path = file_path + str(time.time()) # add a timestamp to avoid overwriting existing file
    with open(file_path + '.pkl', 'wb') as f:
        pickle.dump(param_list, f)

    file_path = str(nqubits) + 'qbt_paulis'
    if os.path.exists(file_path + '.pkl'): # check if the file already exists
        file_path = file_path + str(time.time()) # add a timestamp to avoid overwriting existing file
    with open(file_path + '.pkl', 'wb') as f:
        pickle.dump(pauli_list, f)