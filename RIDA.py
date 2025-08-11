#!/usr/bin/env python
# coding: utf-8


## Settings ##

nqubits = 4 # number of qubits in circuit - will automatically change if run from .sub file
error_multiplier = 1 # multiplies all sources of error
layers = 12 # number of ansatz layers - default is 12
shots = [2**k for k in range(10, 21)] # list of shots to test
max_strings = 20 # maximum number of Pauli strings to test

base_qubits = 4 # if running on HTC, smallest number of qubits tested
nmultipliers = 4 # if running on HTC, number of different error multipliers tested

benchmark_shots = 10**7 # total number of shots used in depolarization estimation circuits
benchmark_size = 50 # number of different depolarization estimation circuits
pauli_sharing = 'partial' # sharing depolarization magnitudes across Pauli strings: full, partial, none
random_rotations = False # implement random rotations for CNOT-only depolarization (WARNING: only works with certain circuits)

coherent_error = True # add coherent rx rotation error in addition to incoherent error
twirling = True # implements both Pauli twirling and readout twirling
num_twirls = 50 # how many different circuits to create during Pauli twirling

progress_flags = True # print checkpoints during testing
print_results = True # print total time taken
save_results = True # save results to a pickle file

readout_error = True # turns readout error on/off
coherent_rotation = 0.05 # amount of rotation error (coherent_error only)

# error model: to proportionally increase the error rate, change the error multiplier instead
# median Kingston error rates as of 7/11/2025
error_1q = 2.25e-4
error_2q = 2.07e-3
ro_error = 7.32e-3
t1 = 269.88e3
t2 = 143.41e3
gate_time = 68
one_qubit_time_factor = 0.1

# benchmark_size = int(benchmark_size/num_twirls+0.99999) # twirling automatically creates additional benchmark circuits
benchmark_shots /= benchmark_size # split estimation circuit shots among each circuit


## Import Libraries ##

import pickle
import numpy as np
import time
import qiskit
import sys
import os

from qiskit.circuit.library import EfficientSU2
from qiskit.dagcircuit import DAGCircuit
from qiskit.circuit import QuantumCircuit, QuantumRegister
from qiskit import transpile
from qiskit.quantum_info import Statevector, Operator, pauli_basis
from qiskit_ibm_runtime.fake_provider import FakeTorino
from qiskit_aer.noise import NoiseModel
from qiskit_aer import AerSimulator
from qiskit.circuit.library import CXGate, ECRGate, RXGate, CZGate
from qiskit.transpiler import PassManager
from qiskit.transpiler.basepasses import TransformationPass
from qiskit_aer.noise import (
    NoiseModel,
    ReadoutError,
    depolarizing_error,
    thermal_relaxation_error, 
    coherent_unitary_error
)

rng = np.random.default_rng()


## Import Paul Strings and Params ##

measurement_map = [[]] # list of what measurements are required by each Pauli string

# import settings from .sub file - ignore if not using
try:
    set_code = int(sys.argv[1])
    error_multiplier = set_code % nmultipliers + 1
    nqubits = int(set_code/nmultipliers) + base_qubits
    progress_flags = False # progress flags do not work if run from .sub file
except:
    pass

# import Pauli strings
input_path = str(nqubits)+'qbt_paulis' # default layers not specified in file name
with open(input_path + '.pkl', 'rb') as f:
    operators = pickle.load(f)

# import parameters
input_path = str(nqubits)+'qbt_params' # default layers not specified in file name
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
coherent_rotation *= error_multiplier
if not readout_error:
    ro_error = 0

backend_torino = FakeTorino()
gates_torino = backend_torino.operation_names # get native gates of hardware for transpilation
if coherent_error:
    noise_model = NoiseModel()
    over_rotation = coherent_unitary_error(RXGate(coherent_rotation).to_matrix())
    noise_model.add_all_qubit_quantum_error(over_rotation.expand(over_rotation), ['cz', 'cx', 'rzz'])
    aerbackend = AerSimulator(noise_model = noise_model)
else:
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

# transpile the ciruit with hardware basis gates, or the list of Pauli twirled circuits
def transpile_circ(circ):
    return transpile(circ, aerbackend, basis_gates = gates_torino)

# re-apply circuit measurements, which are initially excluded for ease of circuit operations
def insert_measurements(circ, measured_qubits):
    flip_result = False # if an odd number of measured qubits are bit-flipped, the final result needs to be classically bit-flipped
    circ = circ.compose(QuantumCircuit(nqubits, len(measured_qubits)))
    if twirling:
        for i in range(len(measured_qubits)):
            op_choice = rng.random()
            if op_choice < 0.25:
                circ.x(measured_qubits[i])
                flip_result = not flip_result
            elif op_choice < 0.5:
                circ.y(measured_qubits[i])
                flip_result = not flip_result
            elif op_choice < 0.75:
                circ.z(measured_qubits[i])
    for i in range(len(measured_qubits)):
        circ.measure(measured_qubits[i], i)
    return circ, flip_result

# determine the expectation value of a circuit, or average among a list of Pauli twirled circuits
def run_circ(circ, shots, measured_qubits):
    if twirling:
        sumc = 0
        for i in range(num_twirls):
            if i == num_twirls-1:
                sumc += run_indiv_circ(pm.run(circ), shots/num_twirls + shots%num_twirls, measured_qubits)
            else:
                sumc += run_indiv_circ(pm.run(circ), shots/num_twirls, measured_qubits)
        return sumc/num_twirls
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

# determine the noiseless expectation of a circuit
def sv_expectation(circ, measured_qubits):
    sump = 0.0
    probs = Statevector(circ).probabilities(measured_qubits)
    for i in range(len(probs)):
        if sum(int(digit) for digit in bin(i)[2:])%2 == 1: #check if it results in a 1 on the classical bit
            sump += probs[i]
    return 1 - 2*sump # the expectation value of an operator is 1-2p(1)


## Pauli Twirling ##
#sourced from https://docs.quantum.ibm.com/guides/custom-transpiler-pass

# add Pauli twirls to two-qubit gates
class PauliTwirl(TransformationPass):
 
    def __init__(
        self,
        gates_to_twirl = None,
    ):
        # gates_to_twirl: names of gates to twirl - the default behavior is to twirl all two-qubit basis gates
        if gates_to_twirl is None:
            gates_to_twirl = [CXGate(), ECRGate(), CZGate()]
        self.gates_to_twirl = gates_to_twirl
        self.build_twirl_set()
        super().__init__()
 
    # build a set of Paulis to twirl for each gate and store internally as .twirl_set
    def build_twirl_set(self):

        self.twirl_set = {}
 
        # iterate through gates to be twirled
        for twirl_gate in self.gates_to_twirl:
            twirl_list = []
 
            # iterate through Paulis on left of gate to twirl
            for pauli_left in pauli_basis(2):
                # iterate through Paulis on right of gate to twirl
                for pauli_right in pauli_basis(2):
                    # save pairs that produce identical operation as gate to twirl
                    if (Operator(pauli_left) @ Operator(twirl_gate)).equiv(
                        Operator(twirl_gate) @ pauli_right
                    ):
                        twirl_list.append((pauli_left, pauli_right))
 
            self.twirl_set[twirl_gate.name] = twirl_list
 
    def run(
        self,
        dag: DAGCircuit,
    ) -> DAGCircuit:
        # collect all nodes in DAG and proceed if it is to be twirled
        twirling_gate_classes = tuple(
            gate.base_class for gate in self.gates_to_twirl
        )
        for node in dag.op_nodes():
            if not isinstance(node.op, twirling_gate_classes):
                continue
 
            # random integer to select Pauli twirl pair
            pauli_index = np.random.randint(
                0, len(self.twirl_set[node.op.name])
            )
            twirl_pair = self.twirl_set[node.op.name][pauli_index]
 
            # instantiate mini_dag and attach quantum register
            mini_dag = DAGCircuit()
            register = QuantumRegister(2)
            mini_dag.add_qreg(register)
 
            # apply left Pauli, gate to twirl, and right Pauli to empty mini-DAG
            mini_dag.apply_operation_back(
                twirl_pair[0].to_instruction(), [register[0], register[1]]
            )
            mini_dag.apply_operation_back(node.op, [register[0], register[1]])
            mini_dag.apply_operation_back(
                twirl_pair[1].to_instruction(), [register[0], register[1]]
            )
 
            # substitute gate to twirl node with twirling mini-DAG
            dag.substitute_node_with_dag(node, mini_dag)
 
        return dag

pm = PassManager([PauliTwirl()])


## Zero Noise Extrapolation ##

class ZNE3(TransformationPass):
 
    def run(
        self,
        dag: DAGCircuit,
    ) -> DAGCircuit:
        for node in dag.op_nodes():
            nq = (node.name == 'cx' or node.name == 'cz') + 1
            if node.name == 'barrier' or node.name == 'measure':
                continue
            
            # instantiate mini_dag and attach quantum register
            mini_dag = DAGCircuit()
            register = QuantumRegister(nq)
            mini_dag.add_qreg(register)
 
            # apply gate, inverse, re-apply gate
            if( nq == 1 ):
                mini_dag.apply_operation_back(node.op, [register[0]])
                mini_dag.apply_operation_back(node.op.inverse(), [register[0]])
                mini_dag.apply_operation_back(node.op, [register[0]])
            else:
                mini_dag.apply_operation_back(node.op, [register[0], register[1]])
                mini_dag.apply_operation_back(node.op.inverse(), [register[0], register[1]])
                mini_dag.apply_operation_back(node.op, [register[0], register[1]])

            # substitute node with ZNE mini-DAG
            dag.substitute_node_with_dag(node, mini_dag)
 
        return dag

class ZNE5(TransformationPass):
 
    def run(
        self,
        dag: DAGCircuit,
    ) -> DAGCircuit:
        for node in dag.op_nodes():
            nq = (node.name == 'cx' or node.name == 'cz') + 1
            if node.name == 'barrier' or node.name == 'measure':
                continue
            
            # instantiate mini_dag and attach quantum register
            mini_dag = DAGCircuit()
            register = QuantumRegister(nq)
            mini_dag.add_qreg(register)
 
            # apply gate, inverse, re-apply gate
            if( nq == 1 ):
                mini_dag.apply_operation_back(node.op, [register[0]])
                mini_dag.apply_operation_back(node.op.inverse(), [register[0]])
                mini_dag.apply_operation_back(node.op, [register[0]])
                mini_dag.apply_operation_back(node.op.inverse(), [register[0]])
                mini_dag.apply_operation_back(node.op, [register[0]])
            else:
                mini_dag.apply_operation_back(node.op, [register[0], register[1]])
                mini_dag.apply_operation_back(node.op.inverse(), [register[0], register[1]])
                mini_dag.apply_operation_back(node.op, [register[0], register[1]])
                mini_dag.apply_operation_back(node.op.inverse(), [register[0], register[1]])
                mini_dag.apply_operation_back(node.op, [register[0], register[1]])

            # substitute node with ZNE mini-DAG
            dag.substitute_node_with_dag(node, mini_dag)
 
        return dag
    
pm3 = PassManager([ZNE3()])
pm5 = PassManager([ZNE5()])

def e_extrapolate( x1, x3, x5 ):
    if (x1 == x3) or (x1 == x5): # check edge case where C=0
        return x1
    if (x1 < x3) != (x1 < x5): # check edge case 1
        return (x1+x3)/2
    if (x5 < x3) != (x5 < x1) or x3 == x5: # check edge case 2
        return (3*x1-x3)/2
    uu = (x3-x5)/(x1-x3)
    return x1 + (x1-x3)/(uu+np.sqrt(uu)) # exponential fit

def q_extrapolate( x1, x3, x5 ):
    return 15*x1/8 - 5*x3/4 + 3*x5/8 # quadratic fit


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
    rida_depo = np.zeros(num_strings)
    cnot_depo = np.zeros(num_strings)

    # use a single depolarization estimate for all Pauli strings
    if pauli_sharing == 'full':
        for j in range(benchmark_size): # average depolarization magnitude across many circuits
            index = int(rng.random()*num_strings) # pick a random operator
            measured_qubits = measurement_map[index]
            logical_circ = generate_random_ansatz().compose(circuits[index])
            circ_t = transpile_circ(logical_circ)
            ri_circ = random_inverse(circ_t, measured_qubits)
            rida_depo[0] += 1 - run_circ(ri_circ, benchmark_shots/benchmark_size, measured_qubits)
            cnot_depo_circ = filter_2qbt(circ_t)
            cnot_depo[0] += 1 - run_circ(cnot_depo_circ, benchmark_shots/benchmark_size, measured_qubits)
        rida_depo[0] /= benchmark_size
        cnot_depo[0] /= benchmark_size
        rida_depo.fill(rida_depo[0])
        cnot_depo.fill(cnot_depo[0])

    # use the same depolarization estimate for circuits with the same qubits with non-identity operators
    elif pauli_sharing == 'partial':
        sim_paulis = {} # unique Pauli strings (all non-identity operators similar) mapped to index of first occurence in operator list
        sim_cnot = {} # CNOT-only depolarization for each Pauli string
        sim_rida = {} # RIDA depolarization for each Pauli string
        for i in range(num_strings):
            frame = str(operators[i][::-1]).replace('X', 'O').replace('Y', 'O').replace('Z', 'O')
            if not frame in sim_paulis:
                sim_paulis[frame] = i
        index = 0

        # calcuate depolarization for each unique Pauli string
        for pstring in sim_paulis:
            measured_qubits = measurement_map[sim_paulis[pstring]]
            sim_rida[pstring] = 0
            sim_cnot[pstring] = 0
            logical_circ = generate_random_ansatz().compose(circuits[sim_paulis[pstring]])
            for j in range(benchmark_size): # average depolarization magnitude across many circuits
                circ_t = transpile_circ(logical_circ)
                ri_circ = random_inverse(circ_t, measured_qubits)
                sim_rida[pstring] += 1 - run_circ(ri_circ, benchmark_shots/benchmark_size, measured_qubits)
                cnot_depo_circ = filter_2qbt(circ_t)
                sim_cnot[pstring] += 1 - run_circ(cnot_depo_circ, benchmark_shots/benchmark_size, measured_qubits)
            sim_rida[pstring] /= benchmark_size
            sim_cnot[pstring] /= benchmark_size
            if progress_flags:
                print('Pauli ', index+1, '/', len(sim_paulis), ' completed')
                rem_time = (time.time() - start) * ((len(sim_paulis)-1-index)/(index+1))
                print('Estimated time remaining: ', int(rem_time/3600), 'hours,', int(rem_time/60%60), 'minutes')
            index += 1

        # distribute depolarization magnitudes to list of Pauli strings
        for i in range(num_strings):
            frame = str(operators[i][::-1]).replace('X', 'O').replace('Y', 'O').replace('Z', 'O')
            rida_depo[i] = sim_rida[frame]
            cnot_depo[i] = sim_cnot[frame]

    # create a seperate depolarization estimate for each Pauli string
    elif pauli_sharing == 'none':
        for i in range(num_strings):
            measured_qubits = measurement_map[i]
            logical_circ = generate_random_ansatz().compose(circuits[i])
            for j in range(benchmark_size): # average depolarization magnitude across many circuits
                circ_t = transpile_circ(logical_circ)
                ri_circ = random_inverse(circ_t, measured_qubits)
                rida_depo[i] += 1 - run_circ(ri_circ, benchmark_shots/benchmark_size, measured_qubits)
                cnot_depo_circ = filter_2qbt(circ_t)
                cnot_depo[i] += 1 - run_circ(cnot_depo_circ, benchmark_shots/benchmark_size, measured_qubits)
            rida_depo[i] /= benchmark_size
            cnot_depo[i] /= benchmark_size
            if progress_flags:
                print('Pauli ', i+1, '/', num_strings, ' completed')
                rem_time = (time.time() - start) * ((num_strings-1-i)/(i+1))
                print('Estimated time remaining: ', int(rem_time/3600), 'hours,', int(rem_time/60%60), 'minutes')
    else:
        raise ValueError('Incorrect Pauli Sharing Option')

    return rida_depo, cnot_depo


## Twirled Readout Error Extinction ##

def generate_trex():
    trex_start = time.time()
    sim_paulis = {} # unique Pauli strings (all non-identity operators similar) mapped to index of first occurence in operator list
    sim_trex = {} # trex error for each Pauli string
    for i in range(num_strings):
        frame = str(operators[i][::-1]).replace('X', 'O').replace('Y', 'O').replace('Z', 'O')
        if not frame in sim_paulis:
            sim_paulis[frame] = i
    index = 0

    # calcuate trex for each unique Pauli string
    for pstring in sim_paulis:
        measured_qubits = measurement_map[sim_paulis[pstring]]
        sim_trex[pstring] = 1 - run_circ(transpile_circ(QuantumCircuit(nqubits)), benchmark_shots, measured_qubits)
        if progress_flags:
            print('Pauli ', index+1, '/', len(sim_paulis), ' completed')
            rem_time = (time.time() - trex_start) * ((len(sim_paulis)-1-index)/(index+1))
            print('Estimated time remaining: ', int(rem_time/3600), 'hours,', int(rem_time/60%60), 'minutes')
        index += 1
        
    # distribute error magnitudes to list of Pauli strings
    trex = np.zeros(num_strings)
    for i in range(num_strings):
        frame = str(operators[i][::-1]).replace('X', 'O').replace('Y', 'O').replace('Z', 'O')
        trex[i] = sim_trex[frame]
    return trex


## Trials ##

def run_trials():
    global raw, rida, cnot, zne
    
    # arrays of results for different choices of Pauli strings and number of shots
    raw = np.zeros((len(shots), num_strings))
    rida = np.zeros((len(shots), num_strings))
    cnot = np.zeros((len(shots), num_strings))
    zne = np.zeros((len(shots), num_strings))

    # generate error-free expectations with statevector
    statevector = np.zeros(num_strings)
    for j in range(num_strings):
        measured_qubits = measurement_map[j]
        logical_circ = ansatz.assign_parameters(data[j]).compose(circuits[j])
        statevector[j] = sv_expectation(logical_circ, measured_qubits)

    # collect data from Aer Simulator
    for j in range(num_strings):
        measured_qubits = measurement_map[j]
        logical_circ = ansatz.assign_parameters(data[j]).compose(circuits[j])
        
        for i in range(len(shots)):
            circ_t = transpile_circ(logical_circ)
            
            raw[i][j] = run_circ(circ_t, shots[i], measured_qubits)
                
            rida[i][j] = (raw[i][j])/(1 - rida_depo[j])

            # split shots among three circuits for methods that require three data points
            p1s = np.zeros(3)
            p1s[0] = run_circ(circ_t, shots[i]/3 + shots[i]%3, measured_qubits)
            p1s[1] = run_circ(pm3.run(circ_t), shots[i]/3, measured_qubits)
            p1s[2] = run_circ(pm5.run(circ_t), shots[i]/3, measured_qubits)

            zne[i][j] = e_extrapolate(p1s[0], p1s[1], p1s[2])
            if readout_error:
                zne[i][j] = (zne[i][j])/(1 - trex[j])
        
            p1s = (p1s)/(1 - cnot_depo[j]) # perform depolarizing approximation before applying quadratic ZNE
            cnot[i][j] = q_extrapolate(p1s[0], p1s[1], p1s[2])
        if progress_flags:
            print('Pauli ', j+1, '/', num_strings, ' completed')
            rem_time = (time.time() - mid) * ((num_strings-1-j)/(j+1))
            print('Estimated time remaining: ', int(rem_time/3600), 'hours,', int(rem_time/60%60),'minutes')

    return statevector, raw, rida, cnot, zne


## Driver ##

start = time.time()
if progress_flags:
    print('-----Generating Benchmarks-----')
rida_depo, cnot_depo = generate_benchmarks()
if readout_error:
    if progress_flags:
        print('---Generating TREX Benchmarks---')
    trex = generate_trex()
mid = time.time()
if progress_flags:
    print('-------Generating Trials-------')
statevector, raw, rida, cnot, zne = run_trials()    

# print time taken
if print_results:
    print()
    end = time.time()
    print('Benchmark time: ', int((mid-start)/3600), 'hours', int((mid-start)/60%60), 'minutes')
    print('Trial time: ', int((end-mid)/3600), 'hours', int((end-mid)/60%60), 'minutes')

# create dictionary with relevant values to export
if save_results:
    save_dict = {'nqubits': nqubits, 'error_multiplier': error_multiplier, 'readout_error': readout_error, 
                 'pauli_sharing': pauli_sharing, 'shots': shots, 'layers': layers,
                 'twirling': twirling, 'random_rotations': random_rotations,
                 'rida_depo': rida_depo, 'cnot_depo': cnot_depo,
                 'statevector': statevector, 'raw': raw, 'rida': rida, 'cnot': cnot, 'zne': zne, 
                 'operators': operators}
    file_path = 'RESULTS - ' + str(nqubits) + ' qubits, ' + str(error_multiplier) + 'x error'
    if twirling:
        file_path = file_path + ', twirled'
    if coherent_error:
        file_path = file_path + ', coherent'
    if random_rotations:
        file_path = file_path + ', rotations'
    if os.path.isfile(file_path + '.pkl'):
        file_path = file_path + ' - ' + str(time.time())
    with open(file_path + '.pkl', 'wb') as f:
        pickle.dump(save_dict, f)