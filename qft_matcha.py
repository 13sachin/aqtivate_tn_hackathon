import numpy as np
from qiskit import QuantumCircuit
from qtealeaves.observables import TNState2File, TNObservables
from qtealeaves.emulator import MPS
from qtealeaves.convergence_parameters import TNConvergenceParameters
from qmatchatea import run_simulation, QCIO
from qmatchatea.utils.qk_utils import GHZ_qiskit
from qmatchatea.utils.utils import print_state 
import os
from PIL import Image
from scipy.fft import dct, fft, idct
import pickle


def qft_rotations(circuit, n):
    if n == 0: # Exit function if circuit is empty
        return circuit
    n -= 1 # Indexes start from 0
    circuit.h(n) # Apply the H-gate to the most significant qubit
    for qubit in range(n):
        circuit.cp(np.pi/2**(n-qubit), qubit, n)
    qft_rotations(circuit, n)

def swap_registers(circuit, n):
    for quit in range(n//2):
        circuit.swap(quit, n-quit-1)
    return circuit
    
def build_qc(num_qubits):
    qc = QuantumCircuit(num_qubits)
    qft_rotations(qc, num_qubits)
    swap_registers(qc, num_qubits)
    return qc

def run_qm_circuit(qc, mps):
    # Write down the quantum circuit. This GHZ state given by the qmatcha library is the "best"
    # for MPS tensor networks, since it uses a linear connectivity
    
    #_ = GHZ_qiskit(qc)

    io_info = QCIO(inPATH="data1/in/", outPATH="data1/out/", initial_state=mps)

    observables = TNObservables()

    save_state = TNState2File(name="my_tn_state", formatting="U")
    observables += save_state

    results = run_simulation(qc, observables=observables, io_info=io_info)


    results.load_state()

    state = results.observables["tn_state_path"]

    print("-" * 30, "Observables results", "-" * 30)
    print(f"The state is saved in {state}, expected is data/out/my_tn_state.pklmps")
    print(f"Class of the saved TN result is {results.tn_state}")
    print("The resulting statevector is:")
    print_state(results.tn_state.to_statevector(qiskit_order=True).elem )
    print()

    comp_time = np.round(results.computational_time, 3)
    meas_time = np.round(results.observables.get("measurement_time", None), 3)
    memory = np.round(np.max(results.observables.get("memory", [0])), 4)
    print("-" * 30, "Runtime statistics", "-" * 30)
    print(f"Datetime of the simulation: {results.date_time}")
    print(f"Computational time: {comp_time} s")
    print(f"Measurement time: {meas_time} s")
    print(f"Maximum memory used: {memory} GB")
    print(
        f"Lower bound on the fidelity F of the state: {results.fidelity}, i.e.  {results.fidelity}≤F≤1"
    )
    return results 

def ket_encoding_zigzag(img):
    return np.ndarray.flatten(img)

def ket_encoding_halfing(img):
    pass

def ket_to_mps(ket):
    n =int(np.log2(len(ket)))
    conv_params = TNConvergenceParameters(max_bond_dimension=64)
    mps = MPS.from_statevector(ket.reshape([2]*n).reshape(-1, order="F"), conv_params=conv_params)
    return mps


def qft(block):
    ket = ket_encoding_zigzag(block)
    norm_value = np.sqrt(np.sum(ket**2))
    norm_ket = ket / norm_value
    num_qubits = np.log2(np.shape(ket)[0])
    qc = build_qc(int(num_qubits))
    mps = ket_to_mps(norm_ket)
    results = run_qm_circuit(qc, mps)

    return results.tn_state.to_statevector(qiskit_order=True).elem.real * norm_value
