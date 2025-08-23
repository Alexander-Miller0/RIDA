# necessary packages
import numpy as np
import qiskit
from qiskit.dagcircuit import DAGCircuit
from qiskit.circuit import QuantumCircuit, QuantumRegister

# determine an estimation circuit for RIDA, or estimation circuits for a list of Pauli twirled circuits
def random_inverse(circ_t):
    rng = np.random.default_rng() # set up rng
    nqubits = circ_t.num_qubits # count number of qubits
    ncbits = circ_t.num_clbits # count number of classical bits
    mes_circ = QuantumCircuit(nqubits, ncbits)
    temp_circ = QuantumCircuit(nqubits, ncbits) # create temporary circuit for non-measurement gates
    measured_qubits = []
    for instruction, qbt, cbt in circ_t.data: # determine which qubits are measured
        if instruction.name == 'measure':
            measured_qubits.append(qbt[0]._index)
            mes_circ.measure(qbt, cbt) # add measurements to mes circ
        else:
            temp_circ.append(instruction, qbt) # add all gates to temporary circ (no measurements)
    circ_t = temp_circ # set original circ to temporary circ with no measurements

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
                opr = gate.operation
                if opr.is_parameterized and len(opr.params) == 1: # set parameters to random
                    opr = opr.to_mutable()
                    opr.params[0] = 2*rng.random()*np.pi - np.pi
                est_circ.apply_operation_front(opr, [register[gate.qubits[0]._index]])
                gates_needed_1q -= 1
            gates_left_1q -= 1
        elif gate.operation.num_qubits == 2:
            if tail_qubits[gate.qubits[0]._index] or tail_qubits[gate.qubits[1]._index]: # check if either qubit is terminal
                if tail_qubits[gate.qubits[0]._index] == False or tail_qubits[gate.qubits[1]._index] == False: # if either is non-terminal, add the gate to terminal gate circuit
                    end_circ.apply_operation_front(gate.operation, [register[gate.qubits[0]._index], register[gate.qubits[1]._index]])
                    tail_qubits[gate.qubits[0]._index] = False
                    tail_qubits[gate.qubits[1]._index] = False
                gates_needed_2q -= 0.5
            elif rng.random() < gates_needed_2q / gates_left_2q: # add the gate to template circuit
                est_circ.apply_operation_front(gate.operation, [register[gate.qubits[0]._index], register[gate.qubits[1]._index]])
                gates_needed_2q -= 1
            gates_left_2q -= 1

    est_circ = qiskit.converters.dag_to_circuit(est_circ)
    est_circ = est_circ.compose(est_circ.inverse())
    est_circ = est_circ.compose(qiskit.converters.dag_to_circuit(end_circ)).compose(mes_circ) # return template circuit followed by terminal gate circuit
    return est_circ

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