#!/usr/bin/env python
# coding: utf-8


## Settings ##

shots = 2*10**5 # number of shots used in each depolarization estimation circuit
max_circuits = 50 # largest number of depolarizing estimation circuits tested
pool_size = 500 # total number of depolarizing estimation circuits to test
nsubsets = 5000 # total number of random subsets to use for each number of circuits

raw_color = '#7B3E3E'
zne_color = '#444A64'
cnot_color = '#91BE8D'
rida_color = '#FF4400'

nqubits = 7 # number of qubits to use for convergence testing
error_multiplier = 3 # error multiplier to use for convergence testing
layers = 12 # assumed to be 12 by default in RIDA.py
random_rotations = False # implement random rotations for CNOT-only depolarization (WARNING: only works with certain circuits)

save_results = True # whether to save graphs to seperate .png files
print_results = True # whether to send graphs to output
progress_flags = True # print checkpoints during testing

# error model: to proportionally increase the error rate, change the error multiplier instead
# median Kingston error rates as of 7/11/2025
error_1q = 2.25e-4
error_2q = 2.07e-3
ro_error = 7.32e-3
t1 = 269.88e3
t2 = 143.41e3
gate_time = 68
one_qubit_time_factor = 0.1


## Import Libraries ##

import pickle
import numpy as np
import matplotlib.pyplot as plt
import time
import qiskit
from scipy.optimize import curve_fit
import os

from qiskit.circuit.library import EfficientSU2
from qiskit.dagcircuit import DAGCircuit
from qiskit.circuit import QuantumCircuit, QuantumRegister
from qiskit import transpile
from qiskit_ibm_runtime.fake_provider import FakeTorino
from qiskit_aer.noise import NoiseModel
from qiskit_aer import AerSimulator
from qiskit_aer.noise import (
    NoiseModel,
    ReadoutError,
    depolarizing_error,
    thermal_relaxation_error
)

rng = np.random.default_rng()


## Set up Noise Model ##

# apply error settings
error_1q *= error_multiplier
error_2q *= error_multiplier
gate_time *= error_multiplier
ro_error *= error_multiplier

backend_torino = FakeTorino()
gates_torino = backend_torino.operation_names # get native gates of hardware for transpilation
noise_model = NoiseModel()

error_1 = depolarizing_error(error_1q, 1)
error_2 = depolarizing_error(error_2q, 2)

error_thermal_1 = thermal_relaxation_error(t1, t2, gate_time*one_qubit_time_factor)
error_thermal_2 = thermal_relaxation_error(t1, t2, gate_time).expand(thermal_relaxation_error(t1, t2, gate_time))
noise_model.add_all_qubit_quantum_error(error_thermal_1, ['id'])
noise_model.add_all_qubit_quantum_error(error_1.compose(error_thermal_1), ['rz', 'sx', 'x', 'rx'])
noise_model.add_all_qubit_quantum_error(error_2.compose(error_thermal_2), ['cz', 'cx', 'rzz'])
noise_model.add_all_qubit_readout_error(ReadoutError([[1 - ro_error, ro_error], [ro_error, 1 - ro_error]]))
aerbackend = AerSimulator(noise_model = noise_model)


## Import Data ##

file_name = 'RESULTS - ' + str(nqubits) + ' qubits, ' + str(error_multiplier) + 'x error.pkl'
with open(file_name, 'rb') as f:
    dict = pickle.load(f)
num_strings = len(dict['operators'])

ansatz = EfficientSU2(nqubits, reps=layers, entanglement='full', skip_unentangled_qubits=False, parameter_prefix='a')
nparams = ansatz.num_parameters

measurement_map = [[]] # list of what measurements are required by each Pauli string

# create the end of the circuit applied after the ansatz
circuits = [None]*num_strings
for i in range(num_strings):
    circ = QuantumCircuit(nqubits)
    pstring = dict['operators'][i][::-1]
    measurement_map.append([])
    for j in range(len(pstring)):
        if str(pstring[j]) == 'Z':
            measurement_map[i].append(j)
        elif str(pstring[j]) == 'X':
            measurement_map[i].append(j)
            circ.h(j)
        elif str(pstring[j]) == 'Y':
            measurement_map[i].append(j)
            circ.s(j)
            circ.h(j)
    circuits[i] = circ


## Helper Functions ##

# count the number of gates in a given circuit, or either one or two qubit gates depending on size variable
def count_gates(circ_t, size = None):
    ngates = 0
    if size == None:
        for gate in circ_t.data:
            if gate.operation.num_qubits > 0:
                ngates += 1
    else:
       for gate in circ_t.data:
            if gate.operation.num_qubits == size:
                ngates += 1 
    return ngates

# transpile the ciruit with hardware basis gates, or the list of Pauli twirled circuits
def transpile_circ(circ):
    return transpile(circ, aerbackend, basis_gates = gates_torino)

# re-apply circuit measurements, which are initially excluded for ease of circuit operations
def insert_measurements(circ, measured_qubits):
    flip_result = False # if an odd number of measured qubits are bit-flipped, the final result needs to be classically bit-flipped
    circ = circ.compose(QuantumCircuit(nqubits, len(measured_qubits)))
    for i in range(len(measured_qubits)):
        circ.measure(measured_qubits[i], i)
    return circ, flip_result

# determine the expectation value of a circuit, or average among a list of Pauli twirled circuits
def run_circ(circ, shots, measured_qubits):
    return run_indiv_circ(circ, shots, measured_qubits)

# determine the expectation value of a circuit
def run_indiv_circ(circ, shots, measured_qubits):
    mes_circ, flip_result = insert_measurements(circ, measured_qubits)
    counts = aerbackend.run(mes_circ, shots = shots).result().get_counts()
    sump = 0
    for result in counts:
        if sum(int(digit) for digit in result)%2 == 1: # check if it results in a 1 on the classical bit
            sump += counts[result]
    if flip_result:
        return 2*sump/shots - 1
    return 1-2*sump/shots # the expectation value of an operator is 1-2p(1)

# generate an ansatz with random parameters
def generate_random_ansatz():
    params = np.random.rand(nparams)*2*np.pi-np.pi
    return EfficientSU2(nqubits, reps=layers, entanglement='full', skip_unentangled_qubits=False, parameter_prefix='a').assign_parameters(params)


## Random Inverse Depolarizing Approximation ##

# determine an estimation circuit for RIDA
def random_inverse(circ_t, measured_qubits):
    # set up the number of one and two qubit gates needed and remaining
    ngates_1q = count_gates(circ_t, 1)
    ngates_2q = count_gates(circ_t, 2)
    gates_needed_1q = ngates_1q/2
    gates_left_1q = ngates_1q
    gates_needed_2q = ngates_2q/2
    gates_left_2q = ngates_2q

    # intialize template circuit to add gates to
    est_circ = DAGCircuit()
    register = QuantumRegister(nqubits)
    est_circ.add_qreg(register)

    # intialize circuit for gates on temrinal qubits
    end_circ = DAGCircuit()
    end_circ.add_qreg(register)

    # determine which qubits are terminal
    tail_qubits = np.full(nqubits, True)
    for qubit in measured_qubits:
        tail_qubits[qubit] = False

    # iterate over each gate backward
    for i in range(len(circ_t.data)-1, -1, -1):
        gate = circ_t.data[i]
        if gate.operation.num_qubits == 1:
            if tail_qubits[gate.qubits[0]._index]: # ignore single qubit gates on terminal qubits
                gates_needed_1q -= 0.5
            elif rng.random() < gates_needed_1q / gates_left_1q: # add the gate to template circuit
                est_circ.apply_operation_front(gate.operation, [register[gate.qubits[0]._index]])
                gates_needed_1q -= 1
            gates_left_1q -= 1
        elif gate.operation.num_qubits == 2:
            if tail_qubits[gate.qubits[0]._index] or tail_qubits[gate.qubits[1]._index]: # check if either qubit is terminal
                if tail_qubits[gate.qubits[0]._index] == False or tail_qubits[gate.qubits[1]._index] == False: # if either is non-terminal, add the gate to terminal gate circuit
                    end_circ.apply_operation_back(gate.operation, [register[gate.qubits[0]._index], register[gate.qubits[1]._index]])
                    tail_qubits[gate.qubits[0]._index] = False
                    tail_qubits[gate.qubits[1]._index] = False
                gates_needed_2q -= 0.5
            elif rng.random() < gates_needed_2q / gates_left_2q: # add the gate to template circuit
                est_circ.apply_operation_front(gate.operation, [register[gate.qubits[0]._index], register[gate.qubits[1]._index]])
                gates_needed_2q -= 1
            gates_left_2q -= 1

    est_circ = qiskit.converters.dag_to_circuit(est_circ)
    est_circ = est_circ.compose(est_circ.inverse())
    est_circ = est_circ.compose(qiskit.converters.dag_to_circuit(end_circ)) # return template circuit followed by terminal gate circuit
    return est_circ


## CNOT-only Depolarization ##

# filter a circuit for only CNOT gates
def filter_2qbt(circ_t):
    rotations = QuantumCircuit(nqubits)
    if random_rotations:
        for i in range(nqubits):
            rotations.rx(rng.random()*2*np.pi-np.pi, i)
    est_circ = DAGCircuit()
    register = QuantumRegister(nqubits)
    est_circ.add_qreg(register)
    for gate in circ_t.decompose().decompose().data:
        if gate.operation.num_qubits == 2:
            est_circ.apply_operation_back(gate.operation, [register[gate.qubits[0]._index], register[gate.qubits[1]._index]])
    est_circ = rotations.compose(qiskit.converters.dag_to_circuit(est_circ)).compose(rotations.inverse())
    return est_circ


## Generate Depolarization Data ##

# generate depolarization magnitudes for RIDA and CNOT-only depolarization
def generate_benchmarks():
    rida_depo = np.zeros(pool_size)
    cnot_depo = np.zeros(pool_size)

    # use a single depolarization estimate for all Pauli strings
    for j in range(pool_size): # average depolarization magnitude across many circuits
        index = int(rng.random()*num_strings) # pick a random operator
        measured_qubits = measurement_map[index]
        logical_circ = generate_random_ansatz().compose(circuits[index])
        circ_t = transpile_circ(logical_circ)
        ri_circ = random_inverse(circ_t, measured_qubits)
        rida_depo[j] = 1 - run_circ(ri_circ, shots, measured_qubits)
        cnot_depo_circ = filter_2qbt(circ_t)
        cnot_depo[j] = 1 - run_circ(cnot_depo_circ, shots, measured_qubits)

        if progress_flags and (j+1)%50 == 0:
            print('Test ', j+1, '/', pool_size, ' completed')
            rem_time = (time.time() - start) * ((pool_size-1-j)/(j+1))
            print('Estimated time remaining: ', int(rem_time/3600), 'hours,', int(rem_time/60%60),'minutes')

    return rida_depo, cnot_depo

# depolarization function for least-squares fit
def depo_func(x, a):
    return x/(1-a)

# determine the true (optimal) depolarizing approximation by minimizing the least-squares error
def optimal_depo():
    return curve_fit(depo_func, dict['raw'][-1], dict['statevector'], [0])[0][0]

# calculate RMSE of the depolarizing probability p for different numbers of circuits
def errors():
    rida_rmse = np.zeros(max_circuits)
    cnot_rmse = np.zeros(max_circuits)
    for i in range(max_circuits): # seperate test case for each possible number of circuits
        for j in range(nsubsets): # create subsets of depolarizing estimations to average
            rida_avg = 0
            cnot_avg = 0
            remaining_selections = i+1
            for k in range(pool_size): # average a random subset of depolarizing estimations
                if rng.random() < remaining_selections/(pool_size-k): # chance equal to runs that need to be selected divided by remaining runs
                    rida_avg += rida_depo[k]/(i+1)
                    cnot_avg += cnot_depo[k]/(i+1)
                    remaining_selections -= 1
            rida_rmse[i] += (rida_avg-true_depo)**2 # calculate rmse of the subset to the true depolarizing rate
            cnot_rmse[i] += (cnot_avg-true_depo)**2 # calculate rmse of the subset to the true depolarizing rate
    rida_rmse = np.sqrt(rida_rmse/nsubsets)
    cnot_rmse = np.sqrt(cnot_rmse/nsubsets)
    return rida_rmse, cnot_rmse


## Driver ##

start = time.time()
if progress_flags:
    print('-----Generating Benchmarks-----')
rida_depo, cnot_depo = generate_benchmarks()
if print_results:
    mid = time.time()
true_depo = optimal_depo()
if progress_flags:
    print('---------Pooling Data----------')
rida_rmse, cnot_rmse = errors()
if print_results:
    print()
    end = time.time()
    print('Data generation time: ', int((mid-start)/3600), 'hours', int((mid-start)/60%60), 'minutes')
    print('Data pooling time: ', int((end-mid)/3600), 'hours', int((end-mid)/60%60), 'minutes')
if save_results:
    save_dict = {'nqubits': nqubits, 'error_multiplier': error_multiplier, 'max_circuits': max_circuits,
                 'rida_rmse': rida_rmse, 'cnot_rmse': cnot_rmse}
    file_path = 'CONVERGENCE RESULTS - ' + str(nqubits) + ' qubits, ' + str(error_multiplier) + 'x error'
    if os.path.isfile(file_path + '.pkl'):
        file_path = file_path + ' - ' + str(time.time())
    with open(file_path + '.pkl', 'wb') as f:
        pickle.dump(save_dict, f)