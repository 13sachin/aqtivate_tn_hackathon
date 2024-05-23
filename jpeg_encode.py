import sys
import os
import numpy as np
from PIL import Image
from scipy.fft import dct, fft
import pickle
import pdb
#import qft_test

#------------------- Functions for Huffman encoding -------------------
# Creating tree nodes
class NodeTree(object):

    def __init__(self, left=None, right=None):
        self.left = left
        self.right = right

    def children(self):
        return (self.left, self.right)

    def nodes(self):
        return (self.left, self.right)

    def __str__(self):
        return '%s_%s' % (self.left, self.right)

# Main function implementing huffman coding
def huffman_code_tree(node, left=True, binString=''):
    if type(node) is str:
        return {node: binString}
    (l, r) = node.children()
    d = dict()
    d.update(huffman_code_tree(l, True, binString + '0'))
    d.update(huffman_code_tree(r, False, binString + '1'))
    return d

def create_huffman_code(freq):
    freq = sorted(freq.items(), key=lambda x: x[1], reverse=True)

    nodes = freq

    while len(nodes) > 1:
        (key1, c1) = nodes[-1]
        (key2, c2) = nodes[-2]
        nodes = nodes[:-2]
        node = NodeTree(key1, key2)
        nodes.append((node, c1 + c2))

        nodes = sorted(nodes, key=lambda x: x[1], reverse=True)

    huffmanCode = huffman_code_tree(nodes[0][0])

    return huffmanCode, nodes, freq

def encode(blocks, huffman):
    encoding = ''
    for block in blocks:
        for item in block:
            encoding += huffman[item]
    return encoding
#----------------------------------------------------------------------


#------------------- Functions for RLE -------------------
def countBits(number):
    number = abs(number)
    return int((np.log(number) / np.log(2)) + 1); 

def addToDict(item):
    if item in freq:
        freq[item] += 1
    else:
        freq[item] = 1

def all_zeros_from_index(arr, index):
    return all(x == 0 for x in arr[index:])

def rle(b, dc):
    blocks = []
    start = b[0] - dc
    if start == 0:
        start1 = '(0)'
        addToDict(start1)
        blocks.append(start1)
    else:
        start1 = f'({countBits(start)})'
        start2 = f'({start})'
        addToDict(start1)
        addToDict(start2)
        blocks.append(start1)
        blocks.append(start2)
    nzeros = 0
    for i in range(1,b.size):
        if b[i] == 0:
            nzeros +=1
        else:
            tmp1 = f'({nzeros},{countBits(b[i])})'
            tmp2 = f'({b[i]})'
            addToDict(tmp1)
            addToDict(tmp2)
            blocks.append(tmp1)
            blocks.append(tmp2)
            nzeros = 0
        if all_zeros_from_index(b, i) or i == b.size-1:
            end = f'(0,0)'
            addToDict(end)
            blocks.append(end)
            return blocks

#---------------------------------------------------------

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
    elif type == 'fft':
        FT = fft(block, norm="ortho")
    elif type == 'qft_vector':
        FT = qft_vector(block)
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

def binary_string_to_bytes(binary_string):
    original_len = len(binary_string)
    padding_needed = (8 - len(binary_string) % 8) % 8
    binary_string_padded = binary_string + '0' * padding_needed

    byte_array = bytearray()
    for i in range(0, len(binary_string_padded), 8):
        byte_segment = binary_string_padded[i:i+8]
        byte_array.append(int(byte_segment, 2))
    return bytes(byte_array), original_len
#------------------------------------------------------------------------------------------------

#-------------------- main code -------------------------------------------------
args = [arg for arg in sys.argv[1:] if not arg.startswith("-")] #args[0] = filename, args[1] = Fouriertype, args[2] = bsize
#args = [0, 0, 0]
#args[0] = 'data\\couple.bmp'
#args[1] = 'dct'
#args[2] = 8

if __name__ == "__main__":
    filename = args[0]

    # Load image
    if os.path.isfile(filename):
        image = Image.open(filename)
    else:
        raise IOError(f"File {filename} is not present.")

    # Convert image to numpy array
    original = np.array(image)
    bsize = int(args[2])
    height = original.shape[0]
    width = original.shape[1]
    qM = QM(bsize)

    freq = {}
    rle_blocks = []
    dc_now = 0
    
    blocks = subdivide_matrix(original, bsize)
    for i in range(height//bsize):
        for j in range(width//bsize):
            tmp = Fourier(blocks[i,j], args[1])
            tmp = np.round(tmp / qM).astype(int)
            tmp = zigzag(tmp)
            rle_blocks.append(rle(tmp, dc_now))
            dc_now = tmp[0]

    huffman, node_tree, freq = create_huffman_code(freq)
    test = ''
    for block in rle_blocks:
        for item in block:
            test += item
    encoding = encode(rle_blocks, huffman)

    byte_data, byte_data_original_length = binary_string_to_bytes(encoding)
    img_sizes = [width, height, bsize]
    fourier_type = args[1]
    data = [huffman, img_sizes, fourier_type, byte_data, byte_data_original_length, filename]
    with open('compressed_data.bin', 'wb') as file:
        pickle.dump(data, file, pickle.HIGHEST_PROTOCOL)
#--------------------------------------------------------------------------------