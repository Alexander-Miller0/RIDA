#!/usr/bin/env python
# coding: utf-8


## Settings ##

nqubits = 7 # number of qubits in circuit - will automatically change if run from .sub file
layers = 1 # number of ansatz layers - default is 12
shots_per_circ = 2976 # number of shots per circuit
max_strings = 30 # number of Pauli strings to test
benchmark_size = 6 # number of different estimation circuit sets

progress_flags = True # print checkpoints during testing
print_results = True # print total time taken
save_results = True # save results to a pickle file

circ_list = [] # keep track of the circuits and shots to run
res_list = [] # list results of the circuits
index_tracker = 0 # keep track of index on list of circuits


## Import Libraries ##

import pickle
import numpy as np
import time
import qiskit
import sys
import os

from qiskit.circuit.library import efficient_su2
from qiskit.dagcircuit import DAGCircuit
from qiskit.circuit import QuantumCircuit, QuantumRegister
from qiskit import transpile
from qiskit.quantum_info import Statevector
from qiskit.transpiler import PassManager
from qiskit.transpiler.basepasses import TransformationPass
from qiskit_ibm_runtime.fake_provider import FakeMarrakesh
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_runtime import EstimatorOptions
from qiskit_ibm_runtime import EstimatorV2 as Estimator

rng = np.random.default_rng()


## Import Params ##

# import parameters
input_path = 'real_' + str(nqubits)+'qbt_params' # default layers not specified in file name
with open(input_path + '.pkl', 'rb') as f:
    data = pickle.load(f)

ansatz = efficient_su2(nqubits, reps=layers, entanglement='full', skip_unentangled_qubits=False, parameter_prefix='a')
nparams = len(data[0])
measured_qubits = [i for i in range(nqubits)]


## Set up Hardware ##
service = QiskitRuntimeService() # IMPORTANT: insert your token and instance info here

backend = service.backend('ibm_marrakesh')
estimator = Estimator(backend)
estimator.options.dynamical_decoupling.enable = True
estimator.options.dynamical_decoupling.sequence_type = 'XpXm'
estimator.options.twirling.enable_gates = True
estimator.options.twirling.num_randomizations = 'auto'
estimator.options.twirling.shots_per_randomization = 'auto'
estimator.options.resilience.measure_mitigation = False

observable = SparsePauliOp('Z'*nqubits)
compiler = generate_preset_pass_manager(backend=backend, optimization_level=1, layout_method='trivial')
basic_compiler = generate_preset_pass_manager(backend=backend, optimization_level=0, layout_method='trivial')
mapped_observable = observable.apply_layout(compiler.run(QuantumCircuit(nqubits)).layout)


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

# generate an ansatz with random parameters
def generate_random_ansatz():
    params = np.random.rand(nparams)*2*np.pi-np.pi
    return efficient_su2(nqubits, reps=layers, entanglement='full', skip_unentangled_qubits=False, parameter_prefix='a').assign_parameters(params)

# determine the noiseless expectation of a circuit
def sv_expectation(circ, measured_qubits):
    sump = 0.0
    probs = Statevector(circ).probabilities(measured_qubits)
    for i in range(len(probs)):
        if sum(int(digit) for digit in bin(i)[2:])%2 == 1: #check if it results in a 1 on the classical bit
            sump += probs[i]
    return 1 - 2*sump # the expectation value of an operator is 1-2p(1)

def push_circ(input): # add a circuit to the list of circuits to be run
    global circ_list
    circ_list.append((input[0], 1/np.sqrt(input[1])))

def pop_result(): # grab the next result from the list of circuits
    global index_tracker
    index_tracker = index_tracker + 1
    return res_list[index_tracker-1]

# fix qubit labels on ciruits originating from DAGs
def fix_DAG_labels(old_circ):
    new_circ = QuantumCircuit(nqubits)
    qubit_map = {old: new for old, new in zip(old_circ.qubits[:nqubits], new_circ.qubits)}
    for instruction in old_circ.data:
        new_qubits = [qubit_map[q] for q in instruction.qubits]
        new_circ.append(instruction.operation, new_qubits)
    return basic_compiler.run(new_circ)

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
    return fix_DAG_labels(est_circ)


## CNOT-only Depolarization ##

# filter a circuit for only two qubit gates
def filter_2qbt(circ_t):
    rotations = QuantumCircuit(nqubits)
    est_circ = DAGCircuit()
    register = QuantumRegister(nqubits)
    est_circ.add_qreg(register)
    for gate in circ_t.decompose().decompose().data:
        if gate.operation.num_qubits == 2:
            est_circ.apply_operation_back(gate.operation, [register[gate.qubits[0]._index], register[gate.qubits[1]._index]])
    est_circ = rotations.compose(qiskit.converters.dag_to_circuit(est_circ)).compose(rotations.inverse())
    return fix_DAG_labels(est_circ)


## Generate Data ##

# create the circuits to run
def setup():
    tindex = 0 # track the index of the target circuit

    # set up estimation circuits for TREX, RIDA, and CNOT-Only
    for i in range(benchmark_size): # average estimations across many circuits
        push_circ((compiler.run(QuantumCircuit(nqubits)), shots_per_circ)) # add TREX circuit
        logical_circ = generate_random_ansatz() # generate baseline for estimation circuits
        circ_t = compiler.run(logical_circ)
        ri_circ = random_inverse(circ_t, measured_qubits)
        push_circ((ri_circ, shots_per_circ)) # add RIDA circuit
        cnot_depo_circ = filter_2qbt(logical_circ)
        cnot_depo_circ3 = filter_2qbt(pm3.run(circ_t))
        cnot_depo_circ5 = filter_2qbt(pm5.run(circ_t))
        push_circ((cnot_depo_circ, shots_per_circ/3)) # add CNOT-Only 1x circuit
        push_circ((cnot_depo_circ3, shots_per_circ/3)) # add CNOT-Only 3x circuit
        push_circ((cnot_depo_circ5, shots_per_circ/3)) # add CNOT-Only 5x circuit

        # collect trial data, which is chronologically mixed with estimation data in order to spead out error rates
        for j in range(int(max_strings/benchmark_size) + (i==0)*(max_strings%benchmark_size)):
            logical_circ = ansatz.assign_parameters(data[tindex])
            circ_t = compiler.run(logical_circ)
            push_circ((circ_t, shots_per_circ))

            # split shots among three circuits for methods that require three data points
            push_circ((circ_t, shots_per_circ/3))
            push_circ((fix_DAG_labels(pm3.run(circ_t)), shots_per_circ/3))
            push_circ((fix_DAG_labels(pm5.run(circ_t)), shots_per_circ/3))
            tindex += 1

# run every circuit
def run_circs():
    result = estimator.run([(i[0], mapped_observable, None, i[1]) for i in circ_list]).result()
    return [i.data.evs.item() for i in result]

# parse the results of circuits
def parse():
    # generate error-free expectations with statevector
    statevector = np.zeros(max_strings)
    for j in range(max_strings):
        logical_circ = ansatz.assign_parameters(data[j])
        statevector[j] = sv_expectation(logical_circ, measured_qubits)

    # create variables for target circuit results
    raw = np.zeros(max_strings)
    tindex = 0 # track the index of the target circuit

    # grab the result of estimation circuits
    trex = 0
    rida_depo = 0
    cnot_depo = 0
    cnot_depo3 = 0
    cnot_depo5 = 0
    p1s = np.zeros((max_strings, 3))
    for i in range(benchmark_size): # average estimations across many circuits
        trex += 1 - pop_result()
        rida_depo += 1 - pop_result()
        cnot_depo += 1 - pop_result() # evaluate depolarization probability on 1x error extrapolation point
        cnot_depo3 += 1 - pop_result()
        cnot_depo5 += 1 - pop_result()

        # collect trial data, which is chronologically mixed with estimation data in order to spead out error rates
        for j in range(int(max_strings/benchmark_size) + (i==0)*(max_strings%benchmark_size)):            
            raw[tindex] = pop_result()

            # split shots among three circuits for methods that require three data points
            p1s[tindex][0] = pop_result()
            p1s[tindex][1] = pop_result()
            p1s[tindex][2] = pop_result()
            tindex += 1
        
    trex /= benchmark_size
    rida_depo /= benchmark_size
    cnot_depo /= benchmark_size
    cnot_depo3 /= benchmark_size
    cnot_depo5 /= benchmark_size
        
    # compute mitigated estimates based on target and estimation circuit data
    rida = raw/(1 - rida_depo)
    cnot = np.zeros(max_strings)
    zne = np.zeros(max_strings)
    for i in range(max_strings): # perform exponential extrapolation
        zne[i] = e_extrapolate(p1s[i][0], p1s[i][1], p1s[i][2])
    for i in range(max_strings): # perform CNOT-Only depolarization on each point, then quadratic extrapolation
        cnot[i] = q_extrapolate(p1s[i][0]/(1-cnot_depo), p1s[i][1]/(1-cnot_depo3), p1s[i][2]/(1-cnot_depo5))

    return trex, rida_depo, cnot_depo, cnot_depo3, cnot_depo5, statevector, raw, rida, cnot, zne


## Driver ##

start = time.time()
if progress_flags:
    print('-----Creating Circuits----')
setup()
mid = time.time()
if progress_flags:
    print('-----Running Circuits-----')
res_list = run_circs()
end = time.time()
if progress_flags:
    print('------Parsing Results-----') 
trex, rida_depo, cnot_depo, cnot_depo3, cnot_depo5, statevector, raw, rida, cnot, zne = parse()

# print time taken
if print_results:
    print()
    end = time.time()
    print('Setup time: ', int((mid-start)/3600), 'hours', int((mid-start)/60%60), 'minutes')
    print('Runtime: ', int((end-mid)/3600), 'hours', int((end-mid)/60%60), 'minutes')

# create dictionary with relevant values to export
if save_results:
    save_dict = {'nqubits': nqubits, 'shots_per_circ': shots_per_circ, 'layers': layers,
                 'rida_depo': rida_depo, 'cnot_depo': cnot_depo,
                 'statevector': statevector, 'raw': raw, 'rida': rida, 'cnot': cnot, 'zne': zne}
    file_path = 'Real hardware results'
    if os.path.isfile(file_path + '.pkl'):
        file_path = file_path + ' - ' + str(time.time())
    with open(file_path + '.pkl', 'wb') as f:
        pickle.dump(save_dict, f)