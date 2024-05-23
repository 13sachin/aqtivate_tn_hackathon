def encode_position(row, col):
    # Start with an empty bit string
    bits = []

    # First bit: 0 for left 8x4 submatrix, 1 for right 8x4 submatrix
    bits.append((col >> 2) & 1)

    # Second bit: 0 for upper 4x4 submatrix within the 8x4 submatrix, 1 for lower 4x4 submatrix
    bits.append((row >> 2) & 1)

    # Third bit: 0 for left 4x2 submatrix within the 4x4 submatrix, 1 for right 4x2 submatrix
    bits.append((col >> 1) & 1)

    # Fourth bit: 0 for upper 2x2 submatrix within the 4x2 submatrix, 1 for lower 2x2 submatrix
    bits.append((row >> 1) & 1)

    # Fifth bit: 0 for left 2x1 submatrix within the 2x2 submatrix, 1 for right 2x1 submatrix
    bits.append(col & 1)

    # Sixth bit: 0 for upper 1x1 submatrix within the 2x1 submatrix, 1 for lower 1x1 submatrix
    bits.append(row & 1)

    # Pad with two zeros (since we have specified only 6 bits so far)
    bits.append(0)
    bits.append(0)

    # Reverse the bit order
    bits.reverse()
    
    # Convert bits to a single byte integer
    byte_value = 0
    for bit in bits:
        byte_value = (byte_value << 1) | bit
    
    return byte_value

def encode_position(row, col):
    # Start with an empty bit string
    bits = []

    # First bit: 0 for left 16x8 submatrix, 1 for right 16x8 submatrix
    bits.append((col >> 3) & 1)

    # Second bit: 0 for upper 8x8 submatrix within the 16x8 submatrix, 1 for lower 8x8 submatrix
    bits.append((row >> 3) & 1)

    # Third bit: 0 for left 8x4 submatrix within the 8x8 submatrix, 1 for right 8x4 submatrix
    bits.append((col >> 2) & 1)

    # Fourth bit: 0 for upper 4x4 submatrix within the 8x4 submatrix, 1 for lower 4x4 submatrix
    bits.append((row >> 2) & 1)

    # Fifth bit: 0 for left 4x2 submatrix within the 4x4 submatrix, 1 for right 4x2 submatrix
    bits.append((col >> 1) & 1)

    # Sixth bit: 0 for upper 2x2 submatrix within the 4x2 submatrix, 1 for lower 2x2 submatrix
    bits.append((row >> 1) & 1)

    # Seventh bit: 0 for left 2x1 submatrix within the 2x2 submatrix, 1 for right 2x1 submatrix
    bits.append(col & 1)

    # Eighth bit: 0 for upper 1x1 submatrix within the 2x1 submatrix, 1 for lower 1x1 submatrix
    bits.append(row & 1)
    
    # Reverse the bit order
    bits.reverse()

    # Convert bits to a single byte integer
    byte_value = 0
    for bit in bits:
        byte_value = (byte_value << 1) | bit
    
    return byte_value

def bit_encoding_matrix(bsize):
    if bsize == 8:
        # Generate the 8x8 matrix with encoded positions
        encoded_matrix = [[encode_position(row, col) for col in range(8)] for row in range(8)]
        return encoded_matrix
    elif bsize == 16:
        encoded_matrix = [[encode_position(row, col) for col in range(16)] for row in range(16)]
        return encoded_matrix
