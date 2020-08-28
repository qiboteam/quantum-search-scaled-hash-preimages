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
        

def g(Va, Vb, Vc, Vd, x, y, anc, rot):
    """Basic component for the BLAKE construction.
    Args:
        Va (list): quantum register part of the permutation matrix v.
        Vb (list): quantum register part of the permutation matrix v.
        Vc (list): quantum register part of the permutation matrix v.
        Vd (list): quantum register part of the permutation matrix v.
        x (list): quantum register part of the message digest d.
        y (list): quantum register part of the message digest d.
        anc (int): ancilliary qubit for the modular addition circuit.
        rot (list): characterization of the rotations performed.
    
    Returns:
        generator of the quantum gates requires to apply the circuit.
    
    """
    n = int(len(Va))
    yield adder_mod2n(Vb, Va, anc)
    yield adder_mod2n(x, Va, anc)
    for i in range(n):
        yield gates.CNOT(Va[i], Vd[i])
    for i in range(rot[3]):
        Vd = [Vd[-1]] + Vd[:-1]
    yield adder_mod2n(Vd, Vc, anc)
    for i in range(n):
        yield gates.CNOT(Vc[i], Vb[i])
    for i in range(rot[2]):
        Vb = [Vb[-1]] + Vb[:-1]
    yield adder_mod2n(Vb, Va, anc)
    yield adder_mod2n(y, Va, anc)
    for i in range(n):
        yield gates.CNOT(Va[i], Vd[i])
    for i in range(rot[1]):
        Vd = [Vd[-1]] + Vd[:-1]
    yield adder_mod2n(Vd, Vc, anc)
    for i in range(n):
        yield gates.CNOT(Vc[i], Vb[i])
    for i in range(rot[0]):
        Vb = [Vb[-1]] + Vb[:-1]


def r_g(Va, Vb, Vc, Vd, x, y, anc, rot):
    """Reversed basic component for the BLAKE construction.
    Args:
        Va (list): quantum register part of the permutation matrix v.
        Vb (list): quantum register part of the permutation matrix v.
        Vc (list): quantum register part of the permutation matrix v.
        Vd (list): quantum register part of the permutation matrix v.
        x (list): quantum register part of the message digest d.
        y (list): quantum register part of the message digest d.
        anc (int): ancilliary qubit for the modular addition circuit.
        rot (list): characterization of the rotations performed.
    
    Returns:
        generator of the quantum gates requires to apply the circuit.
    
    """
    n = int(len(Va))
    for i in range(rot[0]):
        Vb = Vb[1:] + [Vb[0]]
    for i in reversed(range(n)):
        yield gates.CNOT(Vc[i], Vb[i])
    yield r_adder_mod2n(Vd, Vc, anc)
    for i in range(rot[1]):
        Vd = Vd[1:] + [Vd[0]]
    for i in reversed(range(n)):
        yield gates.CNOT(Va[i], Vd[i])
    yield r_adder_mod2n(y, Va, anc)
    yield r_adder_mod2n(Vb, Va, anc)
    for i in range(rot[2]):
        Vb = Vb[1:] + [Vb[0]]
    for i in reversed(range(n)):
        yield gates.CNOT(Vc[i], Vb[i])
    yield r_adder_mod2n(Vd, Vc, anc)
    for i in range(rot[3]):
        Vd = Vd[1:] + [Vd[0]]
    for i in reversed(range(n)):
        yield gates.CNOT(Va[i], Vd[i])
    yield r_adder_mod2n(x, Va, anc)
    yield r_adder_mod2n(Vb, Va, anc)

    
def blake(v, d, x, rot, rho):
    """BLAKE contruction given a rho value. 
    Args:
        v (list): quantum register for the permutation matrix v.
        d (list): quantum register for the message digest d.
        x (int): ancillary qubit used in the modular addition circuit.
        rot (list): characterization of the rotations performed.
        rho (int): number of repetitions of the main permutation g.
        
    Returns:
        generator that applies the full BLAKE cryptographic scheme.
    """
    permutations = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    [14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3],
    [11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4],
    [7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8],
    [9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13],
    [2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9],
    [12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11],
    [13, 11, 7 ,14 ,12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10],
    [6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5],
    [10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0]
    ]
    for i in range(0, 2*rho, 2):
        s = permutations[i//2 % len(permutations)]
        yield g(v[0], v[4], v[8], v[12], d[s[0]], d[s[1]], x, rot)
        for i in range(rot[0]+rot[2]):
            v[4] = [v[4][-1]] + v[4][:-1]
        for i in range(rot[3]+rot[1]):
            v[12] = [v[12][-1]] + v[12][:-1]
        yield g(v[1], v[5], v[9], v[13], d[s[2]], d[s[3]], x, rot)
        for i in range(rot[0]+rot[2]):
            v[5] = [v[5][-1]] + v[5][:-1]
        for i in range(rot[3]+rot[1]):
            v[13] = [v[13][-1]] + v[13][:-1]
        yield g(v[2], v[6], v[10], v[14], d[s[4]], d[s[5]], x, rot)
        for i in range(rot[0]+rot[2]):
            v[6] = [v[6][-1]] + v[6][:-1]
        for i in range(rot[3]+rot[1]):
            v[14] = [v[14][-1]] + v[14][:-1]
        yield g(v[3], v[7], v[11], v[15], d[s[6]], d[s[7]], x, rot)
        for i in range(rot[0]+rot[2]):
            v[7] = [v[7][-1]] + v[7][:-1]
        for i in range(rot[3]+rot[1]):
            v[15] = [v[15][-1]] + v[15][:-1]
                   
        yield g(v[0], v[5], v[10], v[15], d[s[8]], d[s[9]], x, rot)
        for i in range(rot[0]+rot[2]):
            v[5] = [v[5][-1]] + v[5][:-1]
        for i in range(rot[3]+rot[1]):
            v[15] = [v[15][-1]] + v[15][:-1]
        yield g(v[1], v[6], v[11], v[12], d[s[10]], d[s[11]], x, rot)
        for i in range(rot[0]+rot[2]):
            v[6] = [v[6][-1]] + v[6][:-1]
        for i in range(rot[3]+rot[1]):
            v[12] = [v[12][-1]] + v[12][:-1]
        yield g(v[2], v[7], v[8], v[13], d[s[12]], d[s[13]], x, rot)
        for i in range(rot[0]+rot[2]):
            v[7] = [v[7][-1]] + v[7][:-1]
        for i in range(rot[3]+rot[1]):
            v[13] = [v[13][-1]] + v[13][:-1]
        yield g(v[3], v[4], v[9], v[14], d[s[14]], d[s[15]], x, rot)
        for i in range(rot[0]+rot[2]):
            v[4] = [v[4][-1]] + v[4][:-1]
        for i in range(rot[3]+rot[1]):
            v[14] = [v[14][-1]] + v[14][:-1]

            
def r_blake(v, d, x, rot, rho):
    """Reversed BLAKE contruction given a rho value. 
    Args:
        v (list): quantum register for the permutation matrix v.
        d (list): quantum register for the message digest d.
        x (int): ancillary qubit used in the modular addition circuit.
        rot (list): characterization of the rotations performed.
        rho (int): number of repetitions of the main permutation g.
        
    Returns:
        generator that applies the full BLAKE cryptographic scheme in reverse.
    """
    permutations = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    [14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3],
    [11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4],
    [7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8],
    [9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13],
    [2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9],
    [12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11],
    [13, 11, 7 ,14 ,12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10],
    [6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5],
    [10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0]
    ]
    for i in reversed(range(0, 2*rho, 2)):
        s = permutations[i//2 % len(permutations)]
        yield r_g(v[3], v[4], v[9], v[14], d[s[14]], d[s[15]], x, rot)
        for i in range(rot[0]+rot[2]):
            v[4] = v[4][1:] + [v[4][0]]
        for i in range(rot[1]+rot[3]):
            v[14] = v[14][1:] + [v[14][0]]
        yield r_g(v[2], v[7], v[8], v[13], d[s[12]], d[s[13]], x, rot)
        for i in range(rot[0]+rot[2]):
            v[7] = v[7][1:] + [v[7][0]]
        for i in range(rot[1]+rot[3]):
            v[13] = v[13][1:] + [v[13][0]]
        yield r_g(v[1], v[6], v[11], v[12], d[s[10]], d[s[11]], x, rot)
        for i in range(rot[0]+rot[2]):
            v[6] = v[6][1:] + [v[6][0]]
        for i in range(rot[1]+rot[3]):
            v[12] = v[12][1:] + [v[12][0]]
        yield r_g(v[0], v[5], v[10], v[15], d[s[8]], d[s[9]], x, rot)
        for i in range(rot[0]+rot[2]):
            v[5] = v[5][1:] + [v[5][0]]
        for i in range(rot[1]+rot[3]):
            v[15] = v[15][1:] + [v[15][0]]
            
        yield r_g(v[3], v[7], v[11], v[15], d[s[6]], d[s[7]], x, rot)
        for i in range(rot[0]+rot[2]):
            v[7] = v[7][1:] + [v[7][0]]
        for i in range(rot[1]+rot[3]):
            v[15] = v[15][1:] + [v[15][0]]
        yield r_g(v[2], v[6], v[10], v[14], d[s[4]], d[s[5]], x, rot)
        for i in range(rot[0]+rot[2]):
            v[6] = v[6][1:] + [v[6][0]]
        for i in range(rot[1]+rot[3]):
            v[14] = v[14][1:] + [v[14][0]]
        yield r_g(v[1], v[5], v[9], v[13], d[s[2]], d[s[3]], x, rot)
        for i in range(rot[0]+rot[2]):
            v[5] = v[5][1:] + [v[5][0]]
        for i in range(rot[1]+rot[3]):
            v[13] = v[13][1:] + [v[13][0]]
        yield r_g(v[0], v[4], v[8], v[12], d[s[0]], d[s[1]], x, rot)
        for i in range(rot[0]+rot[2]):
            v[4] = v[4][1:] + [v[4][0]]
        for i in range(rot[1]+rot[3]):
            v[12] = v[12][1:] + [v[12][0]]
        
            
def init(iv, t, last=False):
    """Perform the first step of the algorithm classically.
    Args:
        iv (list): inital values that characterize the hash construction.
        t (int): constant that characterizes the initial step.
        last (bool): flag that checks if it is the last step.
            
    Returns:
        v (list): classical bitstrings of the permutation matrix.
    """
    v = [0 for i in range(16)]
    for i in range(8):
        v[i] = iv[i]
        v[i+8] = iv[i]
    v[12] = v[12] ^ (t % 2**64)   # Low word of the offset.
    if last == True:                # last block flag?
        v[14] = v[14] ^ (2**64-1) # v[14] ^ 0xFF..FF   # Invert all bits.
    return v
        

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
        

def create_qc(q):
    """Create the quantum circuit necessary to solve the problem. 
    Args:
        q (int): number of qubits of a site in the permutation matrix.

    Returns:
        v (list): quantum registers of a site in the permutation matrix.
        d (list): quantum registers for the message digest.
        x (int): ancillary qubit needed for the adder circuit.
        ancilla (int): Grover ancilla.
        circuit (Circuit): quantum circuit object for Grover's algorithm.
        qubits (int): total number of qubits in the system.
    """
    k = 0
    v = []
    d = []
    for i in range(16):
        v.append([j+k*q for j in range(q)])
        k += 1
    for i in range(16):
        d.append([j+k*q for j in range(q)])
        k += 1        
    x = k*q
    ancilla = k*q+1
    qubits = k*q+2
    circuit = Circuit(qubits)
    return v, d, x, ancilla, circuit, qubits


def grover_step(q, c, circuit, v, d, x, ancilla, h, rot, rho, iv):
    """Add a full grover step to solve a BLAKE construction to a quantum circuit.
    Args:
        q (int): number of qubits of a site in the permutation matrix.
        c (list): classical register that contains the initial step
        circuit (Circuit): quantum circuit where the Grover step is added.
        v (list): quantum register for the sites the permutation matrix.
        d (list): quantum register for the message digest.
        x (int): anciliary qubit for modular addition.
        ancilla (list): Grover ancilla.
        h (str): hash value that one wants to find preimages of.
        rot (list): characterization of the rotation part of the algorithm.
        rho (int): number of repetitions of the main permutation g.
        iv (list): inital values that characterize the hash construction.
            
    Returns:
        circuit (Circuit): quantum circuit where the Grover step is added.
    """
    n = int(len(h))
    for j in range(16):
        for i in range(q):
            if int("{0:0{bits}b}".format(c[j], bits=q)[i]) == 1:
                circuit.add(gates.X(v[j][i]))
    circuit.add(blake(v, d, x, rot, rho))
    for i in range(rho):
        for i in range(rot[0]+rot[2]):
            v[4] = [v[4][-1]] + v[4][:-1]
        for i in range(rot[3]+rot[1]):
            v[12] = [v[12][-1]] + v[12][:-1]
        for i in range(rot[0]+rot[2]):
            v[5] = [v[5][-1]] + v[5][:-1]
        for i in range(rot[3]+rot[1]):
            v[13] = [v[13][-1]] + v[13][:-1]
        for i in range(rot[0]+rot[2]):
            v[6] = [v[6][-1]] + v[6][:-1]
        for i in range(rot[3]+rot[1]):
            v[14] = [v[14][-1]] + v[14][:-1]
        for i in range(rot[0]+rot[2]):
            v[7] = [v[7][-1]] + v[7][:-1]
        for i in range(rot[3]+rot[1]):
            v[15] = [v[15][-1]] + v[15][:-1]
        for i in range(rot[0]+rot[2]):
            v[5] = [v[5][-1]] + v[5][:-1]
        for i in range(rot[3]+rot[1]):
            v[15] = [v[15][-1]] + v[15][:-1]
        for i in range(rot[0]+rot[2]):
            v[6] = [v[6][-1]] + v[6][:-1]
        for i in range(rot[3]+rot[1]):
            v[12] = [v[12][-1]] + v[12][:-1]
        for i in range(rot[0]+rot[2]):
            v[7] = [v[7][-1]] + v[7][:-1]
        for i in range(rot[3]+rot[1]):
            v[13] = [v[13][-1]] + v[13][:-1]
        for i in range(rot[0]+rot[2]):
            v[4] = [v[4][-1]] + v[4][:-1]
        for i in range(rot[3]+rot[1]):
            v[14] = [v[14][-1]] + v[14][:-1]
    vv1 = []
    vv2 = []
    for i in range(8):
        vv1 += v[i]
        vv2 += v[i+8]
    for i in range(n):
        circuit.add(gates.CNOT(vv2[i], vv1[i]))
    iivv = ''
    for i in range(8):
        iivv += "{0:0{bits}b}".format(iv[i], bits=512)
    for i in range(n):
        if iivv[i] == 1:
            circuit.add(gates.X(vv1[i]))
    for i in range(n):
        if int(h[i]) != 1:
            circuit.add(gates.X(vv1[i]))
    circuit.add(n_2CNOT(vv1[:n], ancilla, x))
    for i in range(n):
        if int(h[i]) != 1:
            circuit.add(gates.X(vv1[i]))
    for i in range(n):
        if iivv[i] == 1:
            circuit.add(gates.X(vv1[i]))
    for i in range(n):
        circuit.add(gates.CNOT(vv2[i], vv1[i]))
    circuit.add(r_blake(v, d, x, rot, rho))
    for i in range(rho):
        for i in range(rot[0]+rot[2]):
            v[4] = v[4][1:] + [v[4][0]]
        for i in range(rot[1]+rot[3]):
            v[14] = v[14][1:] + [v[14][0]]
        for i in range(rot[0]+rot[2]):
            v[7] = v[7][1:] + [v[7][0]]
        for i in range(rot[1]+rot[3]):
            v[13] = v[13][1:] + [v[13][0]]
        for i in range(rot[0]+rot[2]):
            v[6] = v[6][1:] + [v[6][0]]
        for i in range(rot[1]+rot[3]):
            v[12] = v[12][1:] + [v[12][0]]
        for i in range(rot[0]+rot[2]):
            v[5] = v[5][1:] + [v[5][0]]
        for i in range(rot[1]+rot[3]):
            v[15] = v[15][1:] + [v[15][0]]
        for i in range(rot[0]+rot[2]):
            v[7] = v[7][1:] + [v[7][0]]
        for i in range(rot[1]+rot[3]):
            v[15] = v[15][1:] + [v[15][0]]
        for i in range(rot[0]+rot[2]):
            v[6] = v[6][1:] + [v[6][0]]
        for i in range(rot[1]+rot[3]):
            v[14] = v[14][1:] + [v[14][0]]
        for i in range(rot[0]+rot[2]):
            v[5] = v[5][1:] + [v[5][0]]
        for i in range(rot[1]+rot[3]):
            v[13] = v[13][1:] + [v[13][0]]
        for i in range(rot[0]+rot[2]):
            v[4] = v[4][1:] + [v[4][0]]
        for i in range(rot[1]+rot[3]):
            v[12] = v[12][1:] + [v[12][0]]
    for j in range(16):
        for i in range(q):
            if int("{0:0{bits}b}".format(c[j], bits=q)[i]) == 1:
                circuit.add(gates.X(v[j][i]))
    dd = []
    for i in range(16):
        dd += d[i]
    circuit.add(diffuser(dd, x))
    return circuit


def grover_single(q, iv, rot, rho, h, t, grover_it):
    """Run the full Grover's search algorithm to find a preimage of a hash function.
    Args:
        q (int): number of qubits of a site in the permutation matrix.
        iv (list): constants that define the hash construction.
        rot (list): characterization of the rotation part of the algorithm.
        h (str): hash value that one wants to find preimages of.
        t (int): constant that characterizes the initial step.
        grover_it (int): number of Grover steps to be performed.
            
    Returns:
        result (dict): counts of the output generated by the algorithm
    """
    v, d, x, ancilla, circuit, qubits = create_qc(q)
    c = init(q, iv, t, last=True)
    for i in range(grover_it):
        circuit = grover_step(q, c, circuit, v, d, x, ancilla, h, rot, rho, iv)
    return circuit
