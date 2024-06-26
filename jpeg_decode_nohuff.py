import os
import numpy as np
from PIL import Image
from scipy.fft import idct
import pickle
from jpeg_encode import NodeTree

def bytes_to_binary_string(bytes_data):
    return ''.join(f'{byte:08b}' for byte in bytes_data)

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

def unblock(M, width, height, bsize):
    res = []
    for i in range(height//bsize):
        ires = []
        for j in range(width//bsize):
            ires.append(M[i*width//bsize+j])
        res.append(ires)
    res = np.block(res)
    return res

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

qM = np.array([[16, 11, 10, 16, 24, 40, 51, 61], [12, 12, 14, 19, 26, 58, 60, 55], [14, 13, 16, 24, 40, 57, 69, 56], [14, 17, 22, 29, 51, 87, 80, 62], [18, 22, 37, 56, 68, 109, 103, 77], [24, 35, 55, 64, 81, 104, 113, 92], [49, 64, 78, 87, 103, 121, 120, 101], [72, 92, 95, 98, 112, 100, 103, 99]])

with open('compressed_data.bin', 'rb') as file:
    data = pickle.load(file)

huff_tree = data[0]
width, height, bsize = data[1]
#encoded = bytes_to_binary_string(data[2])
encoded = data[2]

#decoded = decode_huffman(encoded, huff_tree)
decoded = decode_rle(encoded)
for i, item in enumerate(decoded):
    tmp = zagzig(item) * qM
    tmp = idct(tmp.T, norm="ortho").T
    tmp = idct(tmp, norm="ortho")
    decoded[i] = np.round(tmp).astype(int)

decoded = np.array(decoded)
decoded = decoded.ravel().reshape((height//bsize, width//bsize, bsize, bsize))
decoded = reassemble_matrix(decoded)
image = Image.fromarray(decoded)
image.show()
image.save('image_reconstructed.png')


