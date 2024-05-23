import numpy as np
from PIL import Image
from scipy.fft import idct, ifft
import pickle
import qft_test as qft

#------------------- Functions for Huffman decoding -------------------
def decode_huffman(encoded, huff_tree):
    decoded = ''
    tmp = ''
    tmp_list = list(huff_tree.values())
    for c in encoded:
        tmp += c
        if tmp in tmp_list:
            decoded += list(huff_tree.keys())[tmp_list.index(tmp)]
            tmp = ''
        
    return decoded
#----------------------------------------------------------------------

#------------------- Functions for RLE --------------------------------
def split_after_every_second_occurrence(s, char):
    result = []
    count = 0
    start = 0
    
    for i, c in enumerate(s):
        if c == char:
            count += 1
            if count == 2:
                result.append(s[start:i+1])
                start = i + 1
                count = 0
    
    # Add the remaining part of the string
    if start < len(s):
        result.append(s[start:])
    
    return result

def split_blocks(encoded):
    encoded = encoded.split('(0,0)')[:-1]
    blocks = []
    for item in encoded:
        tmp = []
        if item[:3] == '(0)':
            tmp.extend(['(0)'])
            item = item[3:]
            tmp.extend(split_after_every_second_occurrence(item, ')'))
        else:
            tmp.extend(split_after_every_second_occurrence(item, ')'))
        blocks.append(tmp) 
    return blocks

def decode_rle(encoded):
    blocks = split_blocks(encoded)
    dc_prev = 0
    first = True
    decoded = []
    for item in blocks:
        res = []
        for elem in item:
            if elem == '(0)':
                res.append(dc_prev)
                first = False
            elif elem != '(0)' and first:
                dc_prev += int(elem.split('(')[2][:-1])
                res.append(dc_prev)
                first = False
            else:
                elem_split = elem.split('(')
                nzeros = int(elem_split[1].split(',')[0])
                for i in range(nzeros):
                    res.append(0)
                res.append(int(elem_split[2][:-1]))
        first = True
        for i in range(bsize**2-len(res)):
            res.append(0)
        decoded.append(res)

    return decoded
#----------------------------------------------------------------------

#------------------- Functions for Transformations ---------------------------------------------------------------
def bytes_to_binary_string(byte_data, original_length):
    binary_string_padded = ''.join(format(byte, '08b') for byte in byte_data)
    return binary_string_padded[:original_length]

def zagzig(l):
    blocksize=int(np.sqrt(len(l)))
    block=np.zeros((blocksize,blocksize))
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

def Fourier(block,type):
    if type == 'dct':
        FT = idct(block.T, norm="ortho").T
        FT = idct(FT, norm="ortho")
    elif type == 'fft':
        FT = ifft(block, norm="ortho")
    elif type == 'qft_vector':
        FT = qft.qft_vector_decode(block)
        FT = FT.real
    return FT

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
#-----------------------------------------------------------------------------------------------------------------



#-------------------- main code -------------------------------------------------
with open('compressed_data.bin', 'rb') as file:
    data = pickle.load(file)

huff_tree = data[0]
width, height, bsize = data[1]
fourier_type = data[2]
encoded = bytes_to_binary_string(data[3], data[4])
filename = data[5]
filename = filename.split('\\')[1].split('.')[0]
qM = QM(bsize)

decoded = decode_huffman(encoded, huff_tree)
decoded = decode_rle(decoded)
for i, item in enumerate(decoded):
    tmp = zagzig(item) * qM
    tmp = Fourier(tmp, fourier_type)
    decoded[i] = np.round(tmp).astype(np.uint8)

decoded = np.array(decoded)
decoded = decoded.ravel().reshape((height//bsize, width//bsize, bsize, bsize))
decoded = reassemble_matrix(decoded)
image = Image.fromarray(decoded)
#image.show()
image.save(f'data_reconstructed\\{filename}_{bsize}_reconstructed.png')
#--------------------------------------------------------------------------------
