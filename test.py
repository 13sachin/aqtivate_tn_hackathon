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
from scipy.fft import dct, fft, idct, ifft
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

def build_qc_inverse(num_qubits):
    qc = QuantumCircuit(num_qubits)
    qft_rotations(qc, num_qubits)
    swap_registers(qc, num_qubits)
    return qc.inverse()

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

    #print("-" * 30, "Observables results", "-" * 30)
    #print(f"The state is saved in {state}, expected is data/out/my_tn_state.pklmps")
    #print(f"Class of the saved TN result is {results.tn_state}")
    #print("The resulting statevector is:")
    #print_state(results.tn_state.to_statevector(qiskit_order=True).elem )
    #print()

    comp_time = np.round(results.computational_time, 3)
    meas_time = np.round(results.observables.get("measurement_time", None), 3)
    memory = np.round(np.max(results.observables.get("memory", [0])), 4)
    #print("-" * 30, "Runtime statistics", "-" * 30)
    #print(f"Datetime of the simulation: {results.date_time}")
    #print(f"Computational time: {comp_time} s")
    #print(f"Measurement time: {meas_time} s")
    #print(f"Maximum memory used: {memory} GB")
    #print(
    #    f"Lower bound on the fidelity F of the state: {results.fidelity}, i.e.  {results.fidelity}≤F≤1"
    #)
    return results 

def ket_encoding_vector(img):
    return np.ndarray.flatten(img)

def ket_encoding_zigzag(img):
    blocksize = int(np.sqrt(img.size))
    block = img.reshape(blocksize, blocksize)
    l=[]
    s=1
    for i in range(2*blocksize-1):
        if s==1:
            x=0
            y=i
            while y>-1:
                if x<blocksize:
                    if y<blocksize:
                        l.append(block[y,x])
                y-=1
                x+=1
                
        if s==-1:
            y=0
            x=i
            while x>-1:
                if y<blocksize:
                    if x<blocksize:
                        l.append(block[y,x])
                x-=1
                y+=1
        s*=-1
    l = np.array(l)
    return l

def ket_encoding_halfing(img):
    pass

def ket_to_mps(ket):
    n =int(np.log2(len(ket)))
    conv_params = TNConvergenceParameters(max_bond_dimension=64)
    #ket = ket.reshape([2]*n).reshape(-1, order="F")
    mps = MPS.from_statevector(ket, conv_params=conv_params)
    return mps

def qft_vector_encode(block):
    bsize = block.shape[0]
    ket = ket_encoding_vector(block).astype(complex)
    norm_value = np.sqrt(np.sum(ket**2))
    #print(norm_value)
    norm_ket = ket / norm_value
    num_qubits = np.log2(np.shape(ket)[0])
    qc = build_qc(int(num_qubits))
    mps = ket_to_mps(norm_ket)
    results = run_qm_circuit(qc, mps)
    results = results.tn_state.to_statevector(qiskit_order=False).elem * norm_value
    return results.reshape(bsize,bsize)

def qft_vector_decode(block):
    bsize = block.shape[0]
    ket = ket_encoding_vector(block).astype(complex)
    norm_value = np.sqrt(np.sum(ket**2))
    norm_ket = ket / norm_value
    num_qubits = np.log2(np.shape(ket)[0])
    qc = build_qc_inverse(int(num_qubits))
    mps = ket_to_mps(norm_ket)
    results = run_qm_circuit(qc, mps)
    results = results.tn_state.to_statevector(qiskit_order=False).elem.real * norm_value
    return results.reshape(bsize,bsize)


def qft_vector_strip(block):
    #bsize = block.shape[0]
    ket = ket_encoding_vector(block)
    norm_value = np.sqrt(np.sum(ket**2))
    norm_ket = ket / norm_value
    num_qubits = np.log2(np.shape(ket)[0])
    qc = build_qc(int(num_qubits))
    mps = ket_to_mps(norm_ket)
    results = run_qm_circuit(qc, mps)
    results = results.tn_state.to_statevector(qiskit_order=True).elem.real * norm_value
    return results#.reshape(bsize,bsize)


def qft_2_iter(block):
    bsize = block.shape[0]
    ket = []
    for j in range(bsize):
        kket = np.ndarray.flatten(qft_vector_strip(block[j]))
        ket.append(j)
    ket = np.array(ket).T
    ret = []
    for i in range(bsize):
        kket = np.ndarray.flatten(qft_vector_strip(ket[i]))
        ret.append(i)
    ret = np.array(ret).T
    return ret


def qft_zigzag(block):
    ket = ket_encoding_zigzag(block)
    norm_value = np.sqrt(np.sum(ket**2))
    norm_ket = ket / norm_value
    num_qubits = np.log2(np.shape(ket)[0])
    qc = build_qc(int(num_qubits))
    mps = ket_to_mps(norm_ket)
    results = run_qm_circuit(qc, mps)
    return results.tn_state.to_statevector(qiskit_order=True).elem.real * norm_value


#------------------------------------------------------------------------------------------------

#------------------- Functions for transformations ----------------------------------------------
def subdivide_matrix(matrix, block_size):
    # Get the shape of the matrix
    m, n = matrix.shape
    
    # Determine the number of blocks along each dimension
    blocks_per_row = m // block_size
    blocks_per_col = n // block_size
    
    # Create a list to hold the blocks
    blocks = []
    
    # Loop through the matrix to extract blocks
    for i in range(blocks_per_row):
        row_blocks = []
        for j in range(blocks_per_col):
            # Extract the block
            block = matrix[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size]
            row_blocks.append(block)
        blocks.append(row_blocks)
    
    return np.array(blocks)

def Fourier(block,type):
    if type == 'dct':
        FT = dct(block, norm="ortho")
        FT = dct(FT.T, norm="ortho").T
    elif type == 'qft_vector_encode':
        FT = qft_vector_encode(block)
    elif type == 'qft_vector_decode':
        FT = qft_vector_decode(block)    
    elif type == 'qft_vector_encode':
        bsize = block.shape[0]
        FT = fft(np.ndarray.flatten(block), norm="ortho")
        FT = FT.reshape((bsize,bsize))
    #elif type == 'qft_vector_decode':
    #    FT = qft_vector
    return FT

def QM(bsize):
    if bsize==8:
        return np.array([
            [16, 11, 10, 16, 24, 40, 51, 61],
            [12, 12, 14, 19, 26, 58, 60, 55],
            [14, 13, 16, 24, 40, 57, 69, 56],
            [14, 17, 22, 29, 51, 87, 80, 62],
            [18, 22, 37, 56, 68, 109, 103, 77],
            [24, 35, 55, 64, 81, 104, 113, 92],
            [49, 64, 78, 87, 103, 121, 120, 101],
            [72, 92, 95, 98, 112, 100, 103, 99]])
    if bsize==16:
        return np.array([
            [16, 11, 10, 16, 24, 40, 51, 61, 72, 80, 95, 100, 120, 130, 140, 150],
            [12, 12, 14, 19, 26, 58, 60, 55, 62, 78, 85, 95, 105, 115, 125, 135],
            [14, 13, 16, 24, 40, 57, 69, 56, 60, 70, 80, 90, 100, 110, 120, 130],
            [14, 17, 22, 29, 51, 87, 80, 62, 65, 75, 85, 95, 105, 115, 125, 135],
            [18, 22, 37, 56, 68, 109, 103, 77, 79, 85, 95, 105, 115, 125, 135, 145],
            [24, 35, 55, 64, 81, 104, 113, 92, 95, 100, 110, 120, 130, 140, 150, 160],
            [49, 64, 78, 87, 103, 121, 120, 101, 105, 110, 120, 130, 140, 150, 160, 170],
            [72, 92, 95, 98, 112, 100, 103, 99, 105, 115, 125, 135, 145, 155, 165, 175],
            [85, 95, 105, 110, 120, 125, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220],
            [95, 105, 115, 120, 130, 135, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230],
            [105, 115, 125, 130, 140, 145, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240],
            [115, 125, 135, 140, 150, 155, 160, 170, 180, 190, 200, 210, 220, 230, 240, 250],
            [125, 135, 145, 150, 160, 165, 170, 180, 190, 200, 210, 220, 230, 240, 250, 260],
            [135, 145, 155, 160, 170, 175, 180, 190, 200, 210, 220, 230, 240, 250, 260, 270],
            [145, 155, 165, 170, 180, 185, 190, 200, 210, 220, 230, 240, 250, 260, 270, 280],
            [155, 165, 175, 180, 190, 195, 200, 210, 220, 230, 240, 250, 260, 270, 280, 290]])

def zigzag(block):
    blocksize = int(np.sqrt(block.size))
    block = block.reshape(blocksize, blocksize)
    l=[]
    s=1
    for i in range(2*blocksize-1):
        if s==1:
            x=0
            y=i
            while y>-1:
                if x<blocksize:
                    if y<blocksize:
                        l.append(block[y,x])
                y-=1
                x+=1
                
        if s==-1:
            y=0
            x=i
            while x>-1:
                if y<blocksize:
                    if x<blocksize:
                        l.append(block[y,x])
                x-=1
                y+=1
        s*=-1
    l = np.array(l)
    return l

def zagzig(l):
    blocksize=int(np.sqrt(len(l)))
    block=np.zeros((blocksize,blocksize)).astype(np.cdouble)
    s=1
    r=0
    for i in range(2*blocksize-1):
        if s==1:
            x=0
            y=i
            while y>-1:
                if x<blocksize:
                    if y<blocksize:
                        block[y,x]=l[r]
                        r+=1
                y-=1
                x+=1

        if s==-1:
            y=0
            x=i
            while x>-1:
                if y<blocksize:
                    if x<blocksize:
                        block[y,x]=l[r]
                        r+=1
                x-=1
                y+=1
        s*=-1

    return block

def binary_string_to_bytes(binary_string):
    original_len = len(binary_string)
    padding_needed = (8 - len(binary_string) % 8) % 8
    binary_string_padded = binary_string + '0' * padding_needed

    byte_array = bytearray()
    for i in range(0, len(binary_string_padded), 8):
        byte_segment = binary_string_padded[i:i+8]
        byte_array.append(int(byte_segment, 2))
    return bytes(byte_array), original_len
    
def reassemble_matrix(blocks):
    # Get the number of blocks in each dimension
    num_blocks_row = len(blocks)
    num_blocks_col = len(blocks[0])
    
    # Get the shape of each block
    block_shape = blocks[0][0].shape
    block_height, block_width = block_shape
    
    # Determine the shape of the original matrix
    m = num_blocks_row * block_height
    n = num_blocks_col * block_width
    
    # Create an empty matrix to hold the reassembled matrix
    reassembled_matrix = np.zeros((m, n), dtype=blocks[0][0].dtype)
    
    # Loop through the blocks and place them in the correct position
    for i in range(num_blocks_row):
        for j in range(num_blocks_col):
            reassembled_matrix[i*block_height:(i+1)*block_height, j*block_width:(j+1)*block_width] = blocks[i][j]
    
    return reassembled_matrix
#------------------------------------------------------------------------------------------------

#-------------------- main code -----------------------------------------------------------------
if __name__ == "__main__":
    filename = "data/cameraman.bmp"

    # Load image
    if os.path.isfile(filename):
        image = Image.open(filename)
    else:
        raise IOError(f"File {filename} is not present.")

    # Convert image to numpy array
    original = np.array(image)
    bsize = 8
    height = original.shape[0]
    width = original.shape[1]
    qM = QM(bsize)

    freq = {}
    list_blocks = []
    #list_blocks_real = []
    #list_blocks_imag = []
    #dc_now = 0
    
    blocks = subdivide_matrix(original, bsize)
    #print(Fourier(blocks[0,0],'qft_vector_encode'))
    #print(Fourier(blocks[0,0],'fft_vector_encode'))
    #a=1/0
    for i in range(height//bsize):
        print(i)
        for j in range(width//bsize):
            tmp = Fourier(blocks[i,j],'qft_vector_encode')
            #if (i==0) and (j==0):
                #print(tmp)
                #tmp = Fourier(blocks[i,j],'qft_vector_encode')

            #tmpr = tmp.real
            #tmpi = tmp.imag
            #tmpr = np.round(tmp / qM).astype(int)
            #tmpi = np.round(tmpi / qM).astype(int)
            #tmpr = zigzag(tmpr)
            #tmpi = zigzag(tmpi)
            tmp = zigzag(tmp)
            #list_blocks_real.append(tmpr)
            #list_blocks_imag.append(tmpi)
            list_blocks.append(tmp)
            #rle_blocks.append(rle(tmp, dc_now))
            #dc_now = tmp[0]

    #np.savetxt("qft_new.csv", np.array(list_blocks), delimiter=",")
    
    decoded = []
    #list_blocks=list_blocks_real + 1j*list_blocks_imag
    print(list_blocks[0][0])
    for i, item in enumerate(np.array(list_blocks)):
        if (i%32 == 0):
            print(i)
        tmp = zagzig(item) #* qM
        tmp = Fourier(tmp,'qft_vector_decode')
        decoded.append(np.round(tmp.real).astype(np.uint8))
    
    decoded = np.array(decoded)
    decoded = decoded.ravel().reshape((height//bsize, width//bsize, bsize, bsize))
    decoded = reassemble_matrix(decoded)
    image = Image.fromarray(decoded)
    image.show()

    #huffman, node_tree, freq = create_huffman_code(freq)
    #encoding = encode(rle_blocks, huffman)

    #byte_data, byte_data_original_length = binary_string_to_bytes(encoding)
    #img_sizes = [width, height, bsize]
    #data = [huffman, img_sizes, byte_data, byte_data_original_length]
    #with open('compressed_data_fft_vector.bin', 'wb') as file:
    #    pickle.dump(data, file, pickle.HIGHEST_PROTOCOL)
#--------------------------------------------------------------------------------
