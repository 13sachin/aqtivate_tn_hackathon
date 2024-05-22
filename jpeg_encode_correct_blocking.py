"""
Example to read and write a png file with python
"""

import os
import numpy as np
from PIL import Image
from scipy.fft import dct
import pickle

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

def block(image, bsize, IB, JB):
    width = image.shape[1]
    image = image.ravel()
    idx = IB*width//bsize+JB*bsize
    block = []
    for i in range(bsize):
        block.append(image[idx+i*width:idx+i*width+bsize])
    return np.array(block)

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

def countBits(number):
    number = abs(number)
    return int((np.log(number) / np.log(2)) + 1); 

def addToDict(item):
    if item in freq:
        freq[item] += 1
    else:
        freq[item] = 1

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
        if i == b.size-1 and b[i] == 0:
            end = f'(0,0)'
            addToDict(end)
            blocks.append(end)
    return blocks

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
    test = ''
    for block in blocks:
        for item in block:
            encoding += huffman[item]
            test += item
    return encoding
    #return test

def binary_string_to_bytes(binary_string):
    original_len = len(binary_string)
    padding_needed = (8 - len(binary_string) % 8) % 8
    binary_string_padded = binary_string + '0' * padding_needed

    byte_array = bytearray()
    for i in range(0, len(binary_string_padded), 8):
        byte_segment = binary_string_padded[i:i+8]
        byte_array.append(int(byte_segment, 2))
    return bytes(byte_array), original_len

def unblock(M, width, height, bsize):
    res = []
    for i in range(height//bsize):
        ires = []
        for j in range(width//bsize):
            ires.append(M[i*width//bsize+j])
        res.append(ires)
    res = np.block(res)
    return res

if __name__ == "__main__":
    filename = "image_compression.png"

    # Load image
    if os.path.isfile(filename):
        image = Image.open(filename)
    else:
        raise IOError(f"File {filename} is not present.")

    # Convert image to numpy array
    original = np.array(image)
    # Here you would do the modifications to the image
    bsize = 8
    height = original.shape[0]
    width = original.shape[1]
    qM = np.array([[16, 11, 10, 16, 24, 40, 51, 61], [12, 12, 14, 19, 26, 58, 60, 55], [14, 13, 16, 24, 40, 57, 69, 56], [14, 17, 22, 29, 51, 87, 80, 62], [18, 22, 37, 56, 68, 109, 103, 77], [24, 35, 55, 64, 81, 104, 113, 92], [49, 64, 78, 87, 103, 121, 120, 101], [72, 92, 95, 98, 112, 100, 103, 99]])

    freq = {}
    rle_blocks = []
    dc_now = 0
    
    blocks = subdivide_matrix(original, bsize)
    for i in range(height//bsize):
        for j in range(width//bsize):
            tmp = dct(blocks[i,j], norm="ortho")
            tmp = dct(tmp.T, norm="ortho").T
            tmp = np.round(tmp / qM).astype(int)
            tmp = zigzag(tmp)
            rle_blocks.append(rle(tmp, dc_now))
            dc_now = tmp[0]

    huffman, node_tree, freq = create_huffman_code(freq)
    encoding = encode(rle_blocks, huffman)

    #print(' Char | Huffman code ')
    #print('----------------------')
    #for (str, frequency) in freq:
    #    print(' %8s |%12s' % (str, huffman[str]))

    byte_data, byte_data_original_length = binary_string_to_bytes(encoding)
    #bytes_data = encoding
    img_sizes = [width, height, bsize]
    data = [huffman, img_sizes, byte_data, byte_data_original_length]
    with open('compressed_data.bin', 'wb') as file:
        pickle.dump(data, file, pickle.HIGHEST_PROTOCOL)
