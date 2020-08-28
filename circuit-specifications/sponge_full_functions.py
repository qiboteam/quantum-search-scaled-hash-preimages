from qibo.models import Circuit
from qibo import gates
import numpy as np

            
def n_mCNOT(controls, target, work):
    """Decomposition of a multi-controlled NOT gate with m qubits of work space.
    Args:
        controls (list): quantum register used as a control for the gate.
        target (int): qubit where the NOT gate is applied.
        work (list): quantum register used as work space.

    Returns:
        quantum gate generator for the multi-controlled NOT gate with m qubits of work space.
    """
    i = 0
    yield gates.TOFFOLI(controls[-1], work[-1], target)
    for i in range(1,len(controls)-2):
        yield gates.TOFFOLI(controls[-1-i], work[-1-i], work[-1-i+1])
    yield gates.TOFFOLI(controls[0], controls[1], work[-1-i])
    for i in reversed(range(1,len(controls)-2)):
        yield gates.TOFFOLI(controls[-1-i], work[-1-i], work[-1-i+1])
    yield gates.TOFFOLI(controls[-1], work[-1], target)
    for i in range(1,len(controls)-2):
        yield gates.TOFFOLI(controls[-1-i], work[-1-i], work[-1-i+1])
    yield gates.TOFFOLI(controls[0], controls[1], work[-1-i])
    for i in reversed(range(1,len(controls)-2)):
        yield gates.TOFFOLI(controls[-1-i], work[-1-i], work[-1-i+1])

        
def n_2CNOT(controls, target, work):
    """Decomposition up to Toffoli gates of a multi-controlled NOT gate with one work qubit.
    Args:
        controls (list): quantum register used as a control for the gate.
        target (int): qubit where the NOT gate is applied.
        work (int): qubit used as work space.

    Returns:
        quantum gate generator for the multi-controlled NOT gate with one work qubit.
    """
    m1 = int(((len(controls)+2)/2)+0.5)
    m2 = int(len(controls)+2-m1-1)
    yield n_mCNOT(controls[0:m1], work, controls[m1:len(controls)]+[target])
    yield n_mCNOT((controls+[work])[m1:m1+m2], target, controls[0:m1])
    yield n_mCNOT(controls[0:m1], work, controls[m1:len(controls)]+[target])
    yield n_mCNOT((controls+[work])[m1:m1+m2], target, controls[0:m1])
        

def adder_mod2n(a, b, x):
    """Quantum circuit for the adder modulo 2^n operation.
    Args:
        a (list): quantum register for the first number to be added.
        b (list): quantum register for the second number to be added, will be replaced by solution.
        x (int): ancillary qubit needed for the adder circuit.

    Returns:
        quantum gate generator that applies the quantum gates for addition modulo 2^n.
    """
    n = int(len(a))
    for i in range(n-2, -1, -1):
        yield gates.CNOT(a[i], b[i])
    yield gates.CNOT(a[n-2], x)
    yield gates.TOFFOLI(a[n-1], b[n-1], x)
    yield gates.CNOT(a[n-3], a[n-2])
    yield gates.TOFFOLI(x, b[n-2], a[n-2])
    yield gates.CNOT(a[n-4], a[n-3])
    for i in range(n-3, 1, -1):
        yield gates.TOFFOLI(a[i+1], b[i], a[i])
        yield gates.CNOT(a[i-2], a[i-1])
    yield gates.TOFFOLI(a[2], b[1], a[1])
    for i in range(n-2, 0, -1):
        yield gates.X(b[i])
    yield gates.CNOT(x, b[n-2])
    for i in range(n-3,  -1, -1):
        yield gates.CNOT(a[i+1], b[i])
    yield gates.TOFFOLI(a[2], b[1], a[1])
    for i in range(2, n-2):
        yield gates.TOFFOLI(a[i+1], b[i], a[i])
        yield gates.CNOT(a[i-2], a[i-1])
        yield gates.X(b[i-1])
    yield gates.TOFFOLI(x, b[n-2], a[n-2])
    yield gates.CNOT(a[n-4], a[n-3])
    yield gates.X(b[n-3])
    yield gates.TOFFOLI(a[n-1], b[n-1], x)
    yield gates.CNOT(a[n-3], a[n-2])
    yield gates.X(b[n-2])
    yield gates.CNOT(a[n-2], x)
    for i in range(n-1, -1, -1):
        yield gates.CNOT(a[i], b[i])
        
        
def r_adder_mod2n(a, b, x):
    """Reversed quantum circuit for the adder modulo 2^n operation.
    Args:
        a (list): quantum register for the first number to be added.
        b (list): quantum register for result of the addition.
        x (int): ancillary qubit needed for the adder circuit.

    Returns:
        quantum gate generator that applies the quantum gates for addition modulo 2^n in reverse.
    """
    n = int(len(a))
    for i in reversed(range(n-1, -1, -1)):
        yield gates.CNOT(a[i], b[i])
    yield gates.CNOT(a[n-2], x)
    yield gates.X(b[n-2])
    yield gates.CNOT(a[n-3], a[n-2])
    yield gates.TOFFOLI(a[n-1], b[n-1], x)
    yield gates.X(b[n-3])
    yield gates.CNOT(a[n-4], a[n-3])
    yield gates.TOFFOLI(x, b[n-2], a[n-2])
    for i in reversed(range(2, n-2)):
        yield gates.X(b[i-1])
        yield gates.CNOT(a[i-2], a[i-1])
        yield gates.TOFFOLI(a[i+1], b[i], a[i])
    yield gates.TOFFOLI(a[2], b[1], a[1])
    for i in reversed(range(n-3,  -1, -1)):
        yield gates.CNOT(a[i+1], b[i])
    yield gates.CNOT(x, b[n-2])
    for i in reversed(range(n-2, 0, -1)):
        yield gates.X(b[i])
    yield gates.TOFFOLI(a[2], b[1], a[1])
    for i in reversed(range(n-3, 1, -1)):
        yield gates.CNOT(a[i-2], a[i-1])
        yield gates.TOFFOLI(a[i+1], b[i], a[i])
    yield gates.CNOT(a[n-4], a[n-3])
    yield gates.TOFFOLI(x, b[n-2], a[n-2])
    yield gates.CNOT(a[n-3], a[n-2])
    yield gates.TOFFOLI(a[n-1], b[n-1], x)
    yield gates.CNOT(a[n-2], x)
    for i in reversed(range(n-2, -1, -1)):
        yield gates.CNOT(a[i], b[i])
        

def qr(a, b, c, d, x, rot):
    """Circuit for the quantum quarter round for the toy Chacha permutation.
    Args:
        a (list): quantum register of a site in the permutation matrix.
        b (list): quantum register of a site in the permutation matrix.
        c (list): quantum register of a site in the permutation matrix.
        d (list): quantum register of a site in the permutation matrix.
        x (int): ancillary qubit needed for the adder circuit.
        rot (list): characterization of the rotation part of the algorithm.

    Returns:
        quantum gate generator that applies the quantum gates for the Chacha quarter round.
    """
    n = int(len(a))
    yield adder_mod2n(b, a, x)
    for i in range(n):
        yield gates.CNOT(a[i], d[i])
    for i in range(rot[0]):
        d = d[1:] + [d[0]]
    yield adder_mod2n(d, c, x)
    for i in range(n):
        yield gates.CNOT(c[i], b[i])
    for i in range(rot[1]):
        b = b[1:] + [b[0]]
    yield adder_mod2n(b, a, x)
    for i in range(n):
        yield gates.CNOT(a[i], d[i])
    for i in range(rot[2]):
        d = d[1:] + [d[0]]
    yield adder_mod2n(d, c, x)
    for i in range(n):
        yield gates.CNOT(c[i], b[i])
    for i in range(rot[3]):
        b = b[1:] + [b[0]]

            
def r_qr(a, b, c, d, x, rot):
    """Reverse circuit for the quantum quarter round for the toy Chacha permutation.
    Args:
        a (list): quantum register of a site in the permutation matrix.
        b (list): quantum register of a site in the permutation matrix.
        c (list): quantum register of a site in the permutation matrix.
        d (list): quantum register of a site in the permutation matrix.
        x (int): ancillary qubit needed for the adder circuit.
        rot (list): characterization of the rotation part of the algorithm.

    Returns:
        quantum gate generator that applies the reversed quantum gates for the Chacha quarter round.
    """
    n = int(len(a))
    for i in range(rot[3]):
        b = [b[-1]] + b[:-1]
    for i in reversed(range(n)):
        yield gates.CNOT(c[i], b[i])
    yield r_adder_mod2n(d, c, x)
    for i in range(rot[2]):
        d = [d[-1]] + d[:-1]
    for i in reversed(range(n)):
        yield gates.CNOT(a[i], d[i])
    yield r_adder_mod2n(b, a, x)
    for i in range(rot[1]):
        b = [b[-1]] + b[:-1]
    for i in reversed(range(n)):
        yield gates.CNOT(c[i], b[i])
    yield r_adder_mod2n(d, c, x)
    for i in range(rot[0]):
        d = [d[-1]] + d[:-1]
    for i in reversed(range(n)):
        yield gates.CNOT(a[i], d[i])
    yield r_adder_mod2n(b, a, x)


def diffuser(q, work):
    """Generator that performs the inversion over the average step in Grover's search algorithm.
    Args:
        q (list): quantum register that encodes the problem.
        work (int): ancilliary qubit used for the multi-controlled gate.

    Returns:
        quantum gate generator that applies the diffusion step.
    """
    n = len(q)
    for i in range(n):
        yield gates.H(q[i])
        yield gates.X(q[i])
    yield gates.H(q[0])
    yield n_2CNOT(q[1:n], q[0], work)
    yield gates.H(q[0])
    for i in range(n):
        yield gates.X(q[i])
        yield gates.H(q[i])
        

def start_grover(q, ancilla):
    """Generator that performs the starting step in Grover's search algorithm.
    Args:
        q (list): quantum register that encodes the problem.
        ancilla (int): Grover ancillary qubit. 

    Returns:
        quantum gate generator for the first step of Grover.
    """
    n = len(q)
    yield gates.X(ancilla)
    yield gates.H(ancilla)
    for i in range(n):
        yield gates.H(q[i])
        
        
def create_qc(q):
    """Create the quantum circuit necessary to solve the problem. 
    Args:
        q (int): q (int): number of qubits of a site in the permutation matrix.

    Returns:
        A (list): quantum register of a site in the permutation matrix.
        B (list): quantum register of a site in the permutation matrix.
        C (list): quantum register of a site in the permutation matrix.
        D (list): quantum register of a site in the permutation matrix.
        x (int): ancillary qubit needed for the adder circuit.
        ancilla (int): Grover ancilla.
        circuit (Circuit): quantum circuit object for Grover's algorithm.
        qubits (int): total number of qubits in the system.
    """
    v = []
    k = 0
    for i in range(16):
        v.append([j+k*q for j in range(q)])
        k += 1    
    x = k*q
    ancilla = k*q+1
    qubits = k*q+2
    circuit = Circuit(qubits)
    return v, x, ancilla, circuit, qubits


def chacha_qr(q, A, B, C, D, rot):
    """Classical implementation of the Chacha quarter round
    Args:
        q (int): number of bits of a site in the permutation matrix.
        A (str): classical bitstring for a site of the permutation matrix.
        B (str): classical bitstring for a site of the permutation matrix.
        C (str): classical bitstring for a site of the permutation matrix.
        D (str): classical bitstring for a site of the permutation matrix.
        rot (list): characterization of the rotation part of the algorithm.
            
    Returns:
        A (str): updated classical bitstring for a site of the permutation matrix.
        B (str): updated classical bitstring for a site of the permutation matrix.
        C (str): updated classical bitstring for a site of the permutation matrix.
        D (str): updated classical bitstring for a site of the permutation matrix.
    """
    a = 0
    b = 0
    c = 0
    d = 0
    for i in range(q):
        a += int(A[i])*2**(q-1-i)
        b += int(B[i])*2**(q-1-i)
        c += int(C[i])*2**(q-1-i)
        d += int(D[i])*2**(q-1-i)
    a = (a + b) % (2**q)
    d = d ^ a
    A = "{0:0{bits}b}".format(a, bits=q)
    B = "{0:0{bits}b}".format(b, bits=q)
    C = "{0:0{bits}b}".format(c, bits=q)
    D = "{0:0{bits}b}".format(d, bits=q)
    for i in range(rot[0]):
            D = D[1:] + D[0]
    a = 0
    b = 0
    c = 0
    d = 0
    for i in range(q):
        a += int(A[i])*2**(q-1-i)
        b += int(B[i])*2**(q-1-i)
        c += int(C[i])*2**(q-1-i)
        d += int(D[i])*2**(q-1-i)
    c = (c + d) % (2**q)
    b = b ^ c
    A = "{0:0{bits}b}".format(a, bits=q)
    B = "{0:0{bits}b}".format(b, bits=q)
    C = "{0:0{bits}b}".format(c, bits=q)
    D = "{0:0{bits}b}".format(d, bits=q)
    for i in range(rot[1]):
            B = B[1:] + B[0]
            
    a = 0
    b = 0
    c = 0
    d = 0
    for i in range(q):
        a += int(A[i])*2**(q-1-i)
        b += int(B[i])*2**(q-1-i)
        c += int(C[i])*2**(q-1-i)
        d += int(D[i])*2**(q-1-i)
    a = (a + b) % (2**q)
    d = d ^ a
    A = "{0:0{bits}b}".format(a, bits=q)
    B = "{0:0{bits}b}".format(b, bits=q)
    C = "{0:0{bits}b}".format(c, bits=q)
    D = "{0:0{bits}b}".format(d, bits=q)
    for i in range(rot[2]):
            D = D[1:] + D[0]
    a = 0
    b = 0
    c = 0
    d = 0
    for i in range(q):
        a += int(A[i])*2**(q-1-i)
        b += int(B[i])*2**(q-1-i)
        c += int(C[i])*2**(q-1-i)
        d += int(D[i])*2**(q-1-i)
    c = (c + d) % (2**q)
    b = b ^ c
    A = "{0:0{bits}b}".format(a, bits=q)
    B = "{0:0{bits}b}".format(b, bits=q)
    C = "{0:0{bits}b}".format(c, bits=q)
    D = "{0:0{bits}b}".format(d, bits=q)
    for i in range(rot[3]):
            B = B[1:] + B[0]
    return A, B, C, D


def initial_step(q, iv, rot):
    """Perform the first step of the algorithm classically.
    Args:
        q (int): number of qubits of a site in the permutation matrix.
        iv (list): initial values that define the hash construction.
        rot (list): characterization of the rotation part of the algorithm.
            
    Returns:
        v (list): classical bitstring for the permutation matrix.
    """
    v = []
    for i in range(16):
        v.append("{0:0{bits}b}".format(iv[i], bits=q))
    for i in range(10):
        v[0], v[4], v[8], v[12] = chacha_qr(q, v[0], v[4], v[8], v[12], rot)
        v[1], v[5], v[9], v[13] = chacha_qr(q, v[1], v[5], v[9], v[13], rot)
        v[2], v[6], v[10], v[14] = chacha_qr(q, v[2], v[6], v[10], v[14], rot)
        v[3], v[7], v[11], v[15] = chacha_qr(q, v[3], v[7], v[11], v[15], rot)

        v[0], v[5], v[10], v[15] = chacha_qr(q, v[0], v[5], v[10], v[15], rot)
        v[1], v[6], v[11], v[12] = chacha_qr(q, v[1], v[6], v[11], v[12], rot)
        v[2], v[7], v[8], v[13] = chacha_qr(q, v[2], v[7], v[8], v[13], rot)
        v[3], v[4], v[9], v[14] = chacha_qr(q, v[3], v[4], v[9], v[14], rot)
    return v


def QhaQha(q, v, x, rot):
    """Circuit that performs the quantum Chacha permutation for the toy model.
    Args:
        q (int): number of qubits of a site in the permutation matrix.
        v (list): quantum register for the permutation matrix.
        x (int): ancillary qubit needed for the adder circuit.
        rot (list): characterization of the rotation part of the algorithm.
            
    Returns:
        generator that applies the ChaCha permutation as a quantum circuit
    """
    for i in range(10):
        yield qr(v[0], v[4], v[8], v[12], x, rot)
        for i in range(rot[0]+rot[2]):
            v[12] = v[12][1:] + [v[12][0]]
        for i in range(rot[1]+rot[3]):
            v[4] = v[4][1:] + [v[4][0]]
        yield qr(v[1], v[5], v[9], v[13], x, rot)
        for i in range(rot[0]+rot[2]):
            v[13] = v[13][1:] + [v[13][0]]
        for i in range(rot[1]+rot[3]):
            v[5] = v[5][1:] + [v[5][0]]
        yield qr(v[2], v[6], v[10], v[14], x, rot)
        for i in range(rot[0]+rot[2]):
            v[14] = v[14][1:] + [v[14][0]]
        for i in range(rot[1]+rot[3]):
            v[6] = v[6][1:] + [v[6][0]]
        yield qr(v[3], v[7], v[11], v[15], x, rot)
        for i in range(rot[0]+rot[2]):
            v[15] = v[15][1:] + [v[15][0]]
        for i in range(rot[1]+rot[3]):
            v[7] = v[7][1:] + [v[7][0]]
                 
        yield qr(v[0], v[5], v[10], v[15], x, rot)
        for i in range(rot[0]+rot[2]):
            v[15] = v[15][1:] + [v[15][0]]
        for i in range(rot[1]+rot[3]):
            v[5] = v[5][1:] + [v[5][0]]
        yield qr(v[1], v[6], v[11], v[12], x, rot)
        for i in range(rot[0]+rot[2]):
            v[12] = v[12][1:] + [v[12][0]]
        for i in range(rot[1]+rot[3]):
            v[6] = v[6][1:] + [v[6][0]]
        yield qr(v[2], v[7], v[8], v[13], x, rot)
        for i in range(rot[0]+rot[2]):
            v[13] = v[13][1:] + [v[13][0]]
        for i in range(rot[1]+rot[3]):
            v[7] = v[7][1:] + [v[7][0]]
        yield qr(v[3], v[4], v[9], v[14], x, rot)
        for i in range(rot[0]+rot[2]):
            v[14] = v[14][1:] + [v[14][0]]
        for i in range(rot[1]+rot[3]):
            v[4] = v[4][1:] + [v[4][0]]
                 
            
def r_QhaQha(q, v, x, rot):
    """Reversed circuit that performs the quantum Chacha permutation for the toy model.
    Args:
        q (int): number of qubits of a site in the permutation matrix.
        v (list): quantum register for the permutation matrix.
        x (int): ancillary qubit needed for the adder circuit.
        rot (list): characterization of the rotation part of the algorithm.
            
    Returns:
        generator that applies the reverse ChaCha permutation as a quantum circuit
    """
    for i in range(10):
        yield r_qr(v[3], v[4], v[9], v[14], x, rot)
        for i in range(rot[0]+rot[2]):
            v[14] = [v[14][-1]] + v[14][:-1]
        for i in range(rot[1]+rot[3]):
            v[4] = [v[4][-1]] + v[4][:-1]
        yield r_qr(v[2], v[7], v[8], v[13], x, rot)
        for i in range(rot[0]+rot[2]):
            v[13] = [v[13][-1]] + v[13][:-1]
        for i in range(rot[1]+rot[3]):
            v[7] = [v[7][-1]] + v[7][:-1]
        yield r_qr(v[1], v[6], v[11], v[12], x, rot)
        for i in range(rot[0]+rot[2]):
            v[12] = [v[12][-1]] + v[12][:-1]
        for i in range(rot[1]+rot[3]):
            v[6] = [v[6][-1]] + v[6][:-1]
        yield r_qr(v[0], v[5], v[10], v[15], x, rot)
        for i in range(rot[0]+rot[2]):
            v[15] = [v[15][-1]] + v[15][:-1]
        for i in range(rot[1]+rot[3]):
            v[5] = [v[5][-1]] + v[5][:-1]
        
        yield r_qr(v[3], v[7], v[11], v[15], x, rot)
        for i in range(rot[0]+rot[2]):
            v[15] = [v[15][-1]] + v[15][:-1]
        for i in range(rot[1]+rot[3]):
            v[7] = [v[7][-1]] + v[7][:-1]
        yield r_qr(v[2], v[6], v[10], v[14], x, rot)
        for i in range(rot[0]+rot[2]):
            v[14] = [v[14][-1]] + v[14][:-1]
        for i in range(rot[1]+rot[3]):
            v[6] = [v[6][-1]] + v[6][:-1]
        yield r_qr(v[1], v[5], v[9], v[13], x, rot)
        for i in range(rot[0]+rot[2]):
            v[13] = [v[13][-1]] + v[13][:-1]
        for i in range(rot[1]+rot[3]):
            v[5] = [v[5][-1]] + v[5][:-1]
        yield r_qr(v[0], v[4], v[8], v[12], x, rot)
        for i in range(rot[0]+rot[2]):
            v[12] = [v[12][-1]] + v[12][:-1]
        for i in range(rot[1]+rot[3]):
            v[4] = [v[4][-1]] + v[4][:-1]
            
            
def grover_step(q, c, circuit, v, x, ancilla, h, rot):
    """Add a full grover step to solve a Sponge Hash construction to a quantum circuit.
    Args:
        q (int): number of qubits of a site in the permutation matrix.
        c (list): classical register that contains the initial step
        circuit (Circuit): quantum circuit where the Grover step is added.
        v (list): quantum register for the permutation matrix.
        x (int): anciliary qubit for modular addition.
        ancilla (int): Grover ancilla.
        h (str): hash value that one wants to find preimages of.
        rot (list): characterization of the rotation part of the algorithm.
            
    Returns:
        circuit (Circuit): quantum circuit where the Grover step is added.
    """
    n = int(len(h))
    for j in range(16):
        for i in range(q):
            if int(c[j][i]) == 1:
                circuit.add(gates.X(v[j][i]))
    circuit.add(QhaQha(q, v, x, rot))
    for i in range(10):
        for i in range(rot[0]+rot[2]):
            v[12] = v[12][1:] + [v[12][0]]
        for i in range(rot[1]+rot[3]):
            v[4] = v[4][1:] + [v[4][0]]
        for i in range(rot[0]+rot[2]):
            v[13] = v[13][1:] + [v[13][0]]
        for i in range(rot[1]+rot[3]):
            v[5] = v[5][1:] + [v[5][0]]
        for i in range(rot[0]+rot[2]):
            v[14] = v[14][1:] + [v[14][0]]
        for i in range(rot[1]+rot[3]):
            v[6] = v[6][1:] + [v[6][0]]
        for i in range(rot[0]+rot[2]):
            v[15] = v[15][1:] + [v[15][0]]
        for i in range(rot[1]+rot[3]):
            v[7] = v[7][1:] + [v[7][0]]
        for i in range(rot[0]+rot[2]):
            v[15] = v[15][1:] + [v[15][0]]
        for i in range(rot[1]+rot[3]):
            v[5] = v[5][1:] + [v[5][0]]
        for i in range(rot[0]+rot[2]):
            v[12] = v[12][1:] + [v[12][0]]
        for i in range(rot[1]+rot[3]):
            v[6] = v[6][1:] + [v[6][0]]
        for i in range(rot[0]+rot[2]):
            v[13] = v[13][1:] + [v[13][0]]
        for i in range(rot[1]+rot[3]):
            v[7] = v[7][1:] + [v[7][0]]
        for i in range(rot[0]+rot[2]):
            v[14] = v[14][1:] + [v[14][0]]
        for i in range(rot[1]+rot[3]):
            v[4] = v[4][1:] + [v[4][0]]
    vh = []
    for i in range(8):
        vh += v[i]
    for i in range(n):
        if int(h[i]) != 1:
            circuit.add(gates.X(vh[i]))
    circuit.add(n_2CNOT(vh[:n], ancilla, x))
    for i in range(n):
        if int(h[i]) != 1:
            circuit.add(gates.X(vh[i]))
    circuit.add(r_QhaQha(q, v, x, rot))
    for i in range(10):
        for i in range(rot[0]+rot[2]):
            v[14] = [v[14][-1]] + v[14][:-1]
        for i in range(rot[1]+rot[3]):
            v[4] = [v[4][-1]] + v[4][:-1]
        for i in range(rot[0]+rot[2]):
            v[13] = [v[13][-1]] + v[13][:-1]
        for i in range(rot[1]+rot[3]):
            v[7] = [v[7][-1]] + v[7][:-1]
        for i in range(rot[0]+rot[2]):
            v[12] = [v[12][-1]] + v[12][:-1]
        for i in range(rot[1]+rot[3]):
            v[6] = [v[6][-1]] + v[6][:-1]
        for i in range(rot[0]+rot[2]):
            v[15] = [v[15][-1]] + v[15][:-1]
        for i in range(rot[1]+rot[3]):
            v[5] = [v[5][-1]] + v[5][:-1]
        for i in range(rot[0]+rot[2]):
            v[15] = [v[15][-1]] + v[15][:-1]
        for i in range(rot[1]+rot[3]):
            v[7] = [v[7][-1]] + v[7][:-1]
        for i in range(rot[0]+rot[2]):
            v[14] = [v[14][-1]] + v[14][:-1]
        for i in range(rot[1]+rot[3]):
            v[6] = [v[6][-1]] + v[6][:-1]
        for i in range(rot[0]+rot[2]):
            v[13] = [v[13][-1]] + v[13][:-1]
        for i in range(rot[1]+rot[3]):
            v[5] = [v[5][-1]] + v[5][:-1]
        for i in range(rot[0]+rot[2]):
            v[12] = [v[12][-1]] + v[12][:-1]
        for i in range(rot[1]+rot[3]):
            v[4] = [v[4][-1]] + v[4][:-1]
    for j in range(16):
        for i in range(q):
            if int(c[j][i]) == 1:
                circuit.add(gates.X(v[j][i]))
    circuit.add(diffuser(vh, x))
    return circuit


def grover_single(q, iv, rot, h, grover_it):
    """Run the full Grover's search algorithm to find a preimage of a hash function.
    Args:
        q (int): number of qubits of a site in the permutation matrix.
        iv (list): initial values that define the hash construction.
        rot (list): characterization of the rotation part of the algorithm.
        h (str): hash value that one wants to find preimages of.
        grover_it (int): number of Grover steps to be performed.
            
    Returns:
        result (dict): counts of the output generated by the algorithm
    """
    v, x, ancilla, circuit, qubits = create_qc(q)
    c = initial_step(q, iv, rot)
    for i in range(grover_it):
        circuit = grover_step(q, c, circuit, v, x, ancilla, h, rot)
    return circuit
