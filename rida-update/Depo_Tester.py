#!/usr/bin/env python
# coding: utf-8


## Settings ##

nqubits = 7 # number of qubits in circuit
error_multiplier = 3 # multiplies all sources of error
layers = 12 # number of ansatz layers - default is 12
shots = [2**20] # list of shots for the unmitigated data
max_strings = 500 # maximum number of Pauli strings to test

benchmark_shots = 10**7 # total number of shots used in the estimation circuits of each frame and method
benchmark_size = 50 # number of different depolarization estimation circuits per frame and method

match_real_shots = False # use 2976 shots for the unmitigated data and mark it on the result file name

progress_flags = True # print checkpoints during testing
print_results = True # print total time taken
save_results = True # save results to a pickle file

benchmark_shots = int(benchmark_shots/benchmark_size)

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
import time
import qiskit
import os

from qiskit.circuit.library import EfficientSU2
from qiskit.dagcircuit import DAGCircuit
from qiskit.circuit import QuantumCircuit, QuantumRegister
from qiskit import transpile
from qiskit.quantum_info import Statevector
from qiskit_ibm_runtime.fake_provider import FakeTorino
from qiskit_aer import AerSimulator
from qiskit_aer.noise import (
    NoiseModel,
    ReadoutError,
    depolarizing_error,
    thermal_relaxation_error
)

rng = np.random.default_rng()


## Import Pauli Strings and Params ##

if match_real_shots:
    shots = [2976]

# import Pauli strings
input_path = str(nqubits)+'qbt_paulis'
with open(input_path + '.pkl', 'rb') as f:
    operators = pickle.load(f)

# import parameters
input_path = str(nqubits)+'qbt_params'
with open(input_path + '.pkl', 'rb') as f:
    data = pickle.load(f)

# remove identity operators
for i in range(len(operators)):
    flag = True
    for char in operators[i]:
        if str(char) != 'I':
            flag = False
    if flag:
        del operators[i]
        break

ansatz = EfficientSU2(nqubits, reps=layers, entanglement='full', skip_unentangled_qubits=False, parameter_prefix='a')
nparams = len(data[0])
if len(operators) > max_strings: # reduce the number of operators for testing
    operators = operators[:max_strings]
num_strings = len(operators)

# create the end of the circuit applied after the ansatz
circuits = [None]*num_strings
measurement_map = [] # list of what measurements are required by each Pauli string
for i in range(num_strings):
    circ = QuantumCircuit(nqubits)
    pstring = operators[i][::-1]
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

# index of a qubit within a circuit
def qubit_index(circ_t, qubit):
    return circ_t.find_bit(qubit).index

# the frame of a Pauli string: all non-identity operators are treated as similar
def frame_of(operator):
    return str(operator[::-1]).replace('X', 'O').replace('Y', 'O').replace('Z', 'O')

# transpile the ciruit with hardware basis gates
def transpile_circ(circ):
    return transpile(circ, aerbackend, basis_gates = gates_torino)

# re-apply circuit measurements, which are initially excluded for ease of circuit operations
def insert_measurements(circ, measured_qubits):
    circ = circ.compose(QuantumCircuit(nqubits, len(measured_qubits)))
    for i in range(len(measured_qubits)):
        circ.measure(measured_qubits[i], i)
    return circ

# determine the expectation value of a circuit
def run_circ(circ, shots, measured_qubits):
    mes_circ = insert_measurements(circ, measured_qubits)
    counts = aerbackend.run(mes_circ, shots = int(shots)).result().get_counts()
    sump = 0
    for result in counts:
        if sum(int(digit) for digit in result)%2 == 1: # check if it results in a 1 on the classical bit
            sump += counts[result]
    return 1-2*sump/shots # the expectation value of an operator is 1-2p(1)

# generate an ansatz with random parameters
def generate_random_ansatz():
    params = np.random.rand(nparams)*2*np.pi-np.pi
    return EfficientSU2(nqubits, reps=layers, entanglement='full', skip_unentangled_qubits=False, parameter_prefix='a').assign_parameters(params)

# determine the noiseless expectation of a circuit
def sv_expectation(circ, measured_qubits):
    sump = 0.0
    probs = Statevector(circ).probabilities(measured_qubits)
    for i in range(len(probs)):
        if sum(int(digit) for digit in bin(i)[2:])%2 == 1: #check if it results in a 1 on the classical bit
            sump += probs[i]
    return 1 - 2*sump # the expectation value of an operator is 1-2p(1)


## Random Inverse Depolarizing Approximation ##

# determine an estimation circuit for RIDA (same procedure as RIDA.py)
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
        qbts = [qubit_index(circ_t, q) for q in gate.qubits]
        if gate.operation.num_qubits == 1:
            if tail_qubits[qbts[0]]: # ignore single qubit gates on terminal qubits
                gates_needed_1q -= 0.5
            elif rng.random() < gates_needed_1q / gates_left_1q: # add the gate to template circuit
                est_circ.apply_operation_front(gate.operation, [register[qbts[0]]])
                gates_needed_1q -= 1
            gates_left_1q -= 1
        elif gate.operation.num_qubits == 2:
            if tail_qubits[qbts[0]] or tail_qubits[qbts[1]]: # check if either qubit is terminal
                if tail_qubits[qbts[0]] == False or tail_qubits[qbts[1]] == False: # if either is non-terminal, add the gate to terminal gate circuit
                    end_circ.apply_operation_back(gate.operation, [register[qbts[0]], register[qbts[1]]])
                    tail_qubits[qbts[0]] = False
                    tail_qubits[qbts[1]] = False
                gates_needed_2q -= 0.5
            elif rng.random() < gates_needed_2q / gates_left_2q: # add the gate to template circuit
                est_circ.apply_operation_front(gate.operation, [register[qbts[0]], register[qbts[1]]])
                gates_needed_2q -= 1
            gates_left_2q -= 1

    est_circ = qiskit.converters.dag_to_circuit(est_circ)
    est_circ = est_circ.compose(est_circ.inverse())
    est_circ = est_circ.compose(qiskit.converters.dag_to_circuit(end_circ)) # return template circuit followed by terminal gate circuit
    return est_circ


## CNOT-only Depolarization ##

# filter a circuit for only two qubit gates, optionally surrounded by a layer of random rotations and its inverse
def filter_2qbt(circ_t, rotate):
    rotations = QuantumCircuit(nqubits)
    if rotate:
        for i in range(nqubits):
            rotations.rx(rng.random()*2*np.pi-np.pi, i)
    est_circ = DAGCircuit()
    register = QuantumRegister(nqubits)
    est_circ.add_qreg(register)
    decomposed = circ_t.decompose().decompose()
    for gate in decomposed.data:
        if gate.operation.num_qubits == 2:
            est_circ.apply_operation_back(gate.operation, [register[qubit_index(decomposed, q)] for q in gate.qubits])
    est_circ = rotations.compose(qiskit.converters.dag_to_circuit(est_circ)).compose(rotations.inverse())
    return est_circ


## Generate Depolarization Data ##

# generate the depolarization value of every individual estimation circuit for each frame and method
def generate_benchmarks():
    frames = [] # unique frames in order of first occurence
    frame_ids = {} # frame -> position in frames
    first_string = [] # index of the first Pauli string with each frame
    frame_index = np.zeros(num_strings, dtype=int) # position in frames of each Pauli string
    for i in range(num_strings):
        frame = frame_of(operators[i])
        if frame not in frame_ids:
            frame_ids[frame] = len(frames)
            frames.append(frame)
            first_string.append(i)
        frame_index[i] = frame_ids[frame]
    nframes = len(frames)

    rida_circ = np.zeros((nframes, benchmark_size))
    cnot_circ = np.zeros((nframes, benchmark_size))
    cnot_rotations_circ = np.zeros((nframes, benchmark_size))
    bench_start = time.time()

    for f in range(nframes):
        measured_qubits = measurement_map[first_string[f]]
        logical_circ = generate_random_ansatz().compose(circuits[first_string[f]])
        for j in range(benchmark_size):
            circ_t = transpile_circ(logical_circ)
            ri_circ = random_inverse(circ_t, measured_qubits)
            rida_circ[f, j] = 1 - run_circ(ri_circ, benchmark_shots, measured_qubits)
            cnot_circ[f, j] = 1 - run_circ(filter_2qbt(circ_t, False), benchmark_shots, measured_qubits)
            cnot_rotations_circ[f, j] = 1 - run_circ(filter_2qbt(circ_t, True), benchmark_shots, measured_qubits)
        if progress_flags:
            print('Frame ', f+1, '/', nframes, ' completed')
            rem_time = (time.time() - bench_start) * ((nframes-1-f)/(f+1))
            print('Estimated time remaining: ', int(rem_time/3600), 'hours,', int(rem_time/60%60), 'minutes')

    return frames, frame_index, rida_circ, cnot_circ, cnot_rotations_circ


## Trials ##

# generate the error-free and unmitigated expectation of each Pauli string
def run_trials():
    statevector = np.zeros(num_strings)
    raw = np.zeros((len(shots), num_strings))
    for j in range(num_strings):
        measured_qubits = measurement_map[j]
        logical_circ = ansatz.assign_parameters(data[j]).compose(circuits[j])
        statevector[j] = sv_expectation(logical_circ, measured_qubits)
        circ_t = transpile_circ(logical_circ)
        for i in range(len(shots)):
            raw[i][j] = run_circ(circ_t, shots[i], measured_qubits)
        if progress_flags:
            print('Pauli ', j+1, '/', num_strings, ' completed')
            rem_time = (time.time() - mid) * ((num_strings-1-j)/(j+1))
            print('Estimated time remaining: ', int(rem_time/3600), 'hours,', int(rem_time/60%60),'minutes')
    return statevector, raw


## Driver ##

start = time.time()
if progress_flags:
    print('-----Generating Benchmarks-----')
frames, frame_index, rida_circ_depo, cnot_circ_depo, cnot_rotations_circ_depo = generate_benchmarks()
mid = time.time()
if progress_flags:
    print('-------Generating Trials-------')
statevector, raw = run_trials()
end = time.time()

# per-string depolarization magnitudes: the average over the estimation circuits of each string's frame (same as RIDA.py's partial sharing)
rida_depo = rida_circ_depo.mean(axis=1)[frame_index]
cnot_depo = cnot_circ_depo.mean(axis=1)[frame_index]
cnot_rotations_depo = cnot_rotations_circ_depo.mean(axis=1)[frame_index]

# print time taken and a summary
if print_results:
    print()
    print('Benchmark time: ', int((mid-start)/3600), 'hours', int((mid-start)/60%60), 'minutes')
    print('Trial time: ', int((end-mid)/3600), 'hours', int((end-mid)/60%60), 'minutes')

# create dictionary with relevant values to export
if save_results:
    save_dict = {'nqubits': nqubits, 'error_multiplier': error_multiplier,
                 'pauli_sharing': 'partial', 'shots': shots, 'layers': layers,
                 'benchmark_shots': benchmark_shots, 'benchmark_size': benchmark_size,
                 'statevector': statevector, 'raw': raw, 'operators': operators,
                 'frames': frames, 'frame_index': frame_index,
                 'rida_circ_depo': rida_circ_depo, 'cnot_circ_depo': cnot_circ_depo,
                 'cnot_rotations_circ_depo': cnot_rotations_circ_depo,
                 'rida_depo': rida_depo, 'cnot_depo': cnot_depo, 'cnot_rotations_depo': cnot_rotations_depo}
    file_path = 'DEPO RESULTS - ' + str(nqubits) + ' qubits, ' + str(error_multiplier) + 'x error'
    if match_real_shots:
        file_path = file_path + ', 2976 shots'
    if os.path.isfile(file_path + '.pkl'):
        file_path = file_path + ' - ' + str(time.time())
    with open(file_path + '.pkl', 'wb') as f:
        pickle.dump(save_dict, f)