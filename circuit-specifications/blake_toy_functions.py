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


def g(a, b, x, y, anc, rot):
    """Basic component for the BLAKE construction.
    Args:
        a (list): quantum register part of the permutation matrix v.
        b (list): quantum register part of the permutation matrix v.
        x (list): quantum register part of the message digest d.
        y (list): quantum register part of the message digest d.
        anc (int): ancilliary qubit for the modular addition circuit.
        rot (list): characterization of the rotations performed.
    
    Returns:
        generator of the quantum gates requires to apply the circuit.
    """
    n = int(len(a))
    yield adder_mod2n(b, a, anc)
    yield adder_mod2n(x, a, anc)
    for i in range(n):
        yield gates.CNOT(a[i], b[i])
    for i in range(rot[1]):
        b = [b[-1]] + b[:-1]
    yield adder_mod2n(b, a, anc)
    yield adder_mod2n(y, a, anc)
    for i in range(n):
        yield gates.CNOT(a[i], b[i])
    for i in range(rot[0]):
        b = [b[-1]] + b[:-1]


def r_g(a, b, x, y, anc, rot):
    """Reversed basic component for the BLAKE construction.
    Args:
        a (list): quantum register part of the permutation matrix v.
        b (list): quantum register part of the permutation matrix v.
        x (list): quantum register part of the message digest d.
        y (list): quantum register part of the message digest d.
        anc (int): ancilliary qubit for the modular addition circuit.
        rot (list): characterization of the rotations performed.
    
    Returns:
        generator of the quantum gates requires to apply the circuit.
    """
    n = int(len(a))
    for i in range(rot[0]):
        b = b[1:] + [b[0]]
    for i in reversed(range(n)):
        yield gates.CNOT(a[i], b[i])
    yield r_adder_mod2n(y, a, anc)
    yield r_adder_mod2n(b, a, anc)
    for i in range(rot[1]):
        b = b[1:] + [b[0]]
    for i in reversed(range(n)):
        yield gates.CNOT(a[i], b[i])
    yield r_adder_mod2n(x, a, anc)
    yield r_adder_mod2n(b, a, anc)


def blake(v, d, x, rot, rho):
    """BLAKE contruction given a rho value. 
    Args:
        v (list): quantum register representing the permutation matrix.
        d (list): quantum register representing the message digest matrix.
        x (int): ancillary qubit used in the modular addition circuit.
        rot (list): characterization of the rotations performed.
        rho (int): number of repetitions of the main permutation g.
        
    Returns:
        generator that applies the BLAKE cryptographic scheme.
    """
    permutations = [
    [0,1,2,3],
    [1,0,2,3],
    [0,2,3,1],
    [3,1,2,0],
    ]
    for i in range(0, 2*rho, 2):
        s = permutations[i//2 % len(permutations)]
        yield g(v[0], v[2], d[s[0]], d[s[1]], x[0], rot)
        for i in range(sum(rot)):
            v[2] = [v[2][-1]] + v[2][:-1]
        yield g(v[1], v[3], d[s[2]], d[s[3]], x[1], rot)
        for i in range(sum(rot)):
            v[3] = [v[3][-1]] + v[3][:-1]
        yield g(v[0], v[3], d[s[0]], d[s[1]], x[0], rot)
        for i in range(sum(rot)):
            v[3] = [v[3][-1]] + v[3][:-1]
        yield g(v[1], v[2], d[s[2]], d[s[3]], x[1], rot)
        for i in range(sum(rot)):
            v[2] = [v[2][-1]] + v[2][:-1]


def r_blake(v, d, x, rot, rho):
    """Reversed BLAKE contruction given a rho value. 
    Args:
        v (list): quantum register representing the permutation matrix.
        d (list): quantum register representing the message digest matrix.
        x (int): ancillary qubit used in the modular addition circuit.
        rot (list): characterization of the rotations performed.
        rho (int): number of repetitions of the main permutation g.
        
    Returns:
        generator that applies the BLAKE cryptographic scheme in reverse.
    """
    permutations = [
    [0,1,2,3],
    [1,0,2,3],
    [0,2,3,1],
    [3,1,2,0],
    ]
    for i in reversed(range(0, 2*rho, 2)):
        s = permutations[i//2 % len(permutations)]
        yield r_g(v[1], v[2], d[s[2]], d[s[3]], x[0], rot)
        for i in range(sum(rot)):
            v[2] = v[2][1:] + [v[2][0]]
        yield r_g(v[0], v[3], d[s[0]], d[s[1]], x[1], rot)
        for i in range(sum(rot)):
            v[3] = v[3][1:] + [v[3][0]]
        yield r_g(v[1], v[3], d[s[2]], d[s[3]], x[0], rot)
        for i in range(sum(rot)):
            v[3] = v[3][1:] + [v[3][0]]
        yield r_g(v[0], v[2], d[s[0]], d[s[1]], x[1], rot)
        for i in range(sum(rot)):
            v[2] = v[2][1:] + [v[2][0]]


def init(q, iv0, iv1, t, last=False):
    """Perform the first step of the algorithm classically.
    Args:
        iv0 (int): initial value that characterizes the hash construction.
        iv1 (int): initial value that characterizes the hash construction.
        t (int): constant that characterizes the initial step.
        last (bool): flag that checks if it is the last step.
            
    Returns:
        v0 (str): classical bitstring for a site of the permutation matrix.
        v1 (str): classical bitstring for a site of the permutation matrix.
        v2 (str): classical bitstring for a site of the permutation matrix.
        v3 (str): classical bitstring for a site of the permutation matrix.
    """
    h0 = iv0^0x2
    h1 = iv1
    v0 = h0^t
    v1 = h1^t
    v2 = iv0
    if last == True:
        v2 ^= 0xF
    v3 = iv1
    v0 = "{0:0{bits}b}".format(v0, bits=q)
    v1 = "{0:0{bits}b}".format(v1, bits=q)
    v2 = "{0:0{bits}b}".format(v2, bits=q)
    v3 = "{0:0{bits}b}".format(v3, bits=q)
    return v0, v1, v2, v3


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
        v0 (list): quantum register of a site in the permutation matrix.
        v1 (list): quantum register of a site in the permutation matrix.
        v2 (list): quantum register of a site in the permutation matrix.
        v3 (list): quantum register of a site in the permutation matrix.
        d0 (list): quantum register of the message digest.
        d1 (list): quantum register of the message digest.
        d2 (list): quantum register of the message digest.
        d3 (list): quantum register of the message digest.
        x (int): ancillary qubit needed for the adder circuit.
        ancilla (int): Grover ancilla.
        circuit (Circuit): quantum circuit object for Grover's algorithm.
        qubits (int): total number of qubits in the system.
    """
    v0 = [i for i in range(q)]
    v1 = [i+q for i in range(q)]
    v2 = [i+2*q for i in range(q)]
    v3 = [i+3*q for i in range(q)]
    d0 = [i+4*q for i in range(q)]
    d1 = [i+5*q for i in range(q)]
    d2 = [i+6*q for i in range(q)]
    d3 = [i+7*q for i in range(q)]
    x = [8*q, 8*q+1]
    ancilla = 8*q+2
    qubits = 8*q+3
    circuit = Circuit(qubits)
    return v0, v1, v2, v3, d0, d1 ,d2, d3, x, ancilla, circuit, qubits


def grover_step(q, c, circuit, v, d, x, ancilla, h, rot, rho, iv):
    """Add a full grover step to solve a BLAKE construction to a quantum circuit.
    Args:
        q (int): number of qubits of a site in the permutation matrix.
        c (list): classical register that contains the initial step
        circuit (Circuit): quantum circuit where the Grover step is added.
        v (list): quantum register for the sites the permutation matrix.
        d (list): auxiliary quantum registers.
        x (int): anciliary qubit for modular addition.
        ancilla (list): Grover ancilla.
        h (str): hash value that one wants to find preimages of.
        rot (list): characterization of the rotation part of the algorithm.
        rho (int): number of repetitions of the main permutation g.
        iv (list): initial values that characterize the permutation.
            
    Returns:
        circuit (Circuit): quantum circuit where the Grover step is added.
    """
    n = int(len(h))
    for i in range(q):
        if int(c[0][i]) == 1:
            circuit.add(gates.X(v[0][i]))
        if int(c[1][i]) == 1:
            circuit.add(gates.X(v[1][i]))
        if int(c[2][i]) == 1:
            circuit.add(gates.X(v[2][i]))
        if int(c[3][i]) == 1:
            circuit.add(gates.X(v[3][i]))
    circuit.add(blake(v, d, x, rot, rho))
    for i in range(rho):
        for i in range(sum(rot)):
            v[2] = [v[2][-1]] + v[2][:-1]
        for i in range(sum(rot)):
            v[3] = [v[3][-1]] + v[3][:-1]
        for i in range(sum(rot)):
            v[3] = [v[3][-1]] + v[3][:-1]
        for i in range(sum(rot)):
            v[2] = [v[2][-1]] + v[2][:-1]
    for i in range(n):
        circuit.add(gates.CNOT((v[2]+v[3])[i], (v[0]+v[1])[i]))
    for i in range(n):
        if ("{0:0{bits}b}".format(iv[0], bits=q)+"{0:0{bits}b}".format(iv[1], bits=q))[i] == 1:
            circuit.add(gates.X((v[0]+v[1])[i]))
    for i in range(n):
        if int(h[i]) != 1:
            circuit.add(gates.X((v[0]+v[1])[i]))
    circuit.add(n_2CNOT((v[0]+v[1])[:n], ancilla, x[0]))
    for i in range(n):
        if int(h[i]) != 1:
            circuit.add(gates.X((v[0]+v[1])[i]))
    for i in range(n):
        if ("{0:0{bits}b}".format(iv[0], bits=q)+"{0:0{bits}b}".format(iv[1], bits=q))[i] == 1:
            circuit.add(gates.X((v[0]+v[1])[i]))
    for i in range(n):
        circuit.add(gates.CNOT((v[2]+v[3])[i], (v[0]+v[1])[i]))
    circuit.add(r_blake(v, d, x, rot, rho))
    for i in range(rho):
        for i in range(sum(rot)):
            v[2] = v[2][1:] + [v[2][0]]
        for i in range(sum(rot)):
            v[3] = v[3][1:] + [v[3][0]]
        for i in range(sum(rot)):
            v[3] = v[3][1:] + [v[3][0]]
        for i in range(sum(rot)):
            v[2] = v[2][1:] + [v[2][0]]
    for i in range(q):
        if int(c[0][i]) == 1:
            circuit.add(gates.X(v[0][i]))
        if int(c[1][i]) == 1:
            circuit.add(gates.X(v[1][i]))
        if int(c[2][i]) == 1:
            circuit.add(gates.X(v[2][i]))
        if int(c[3][i]) == 1:
            circuit.add(gates.X(v[3][i]))
    circuit.add(diffuser(d[0]+d[1]+d[2]+d[3], x[0]))
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
    v0, v1, v2, v3, d0, d1, d2, d3, x, ancilla, circuit, qubits = create_qc(q)
    c0, c1, c2, c3 = init(q, iv[0], iv[1], t, last=True)
    c = []
    c.append(c0)
    c.append(c1)
    c.append(c2)
    c.append(c3)
    v = []
    v.append(v0)
    v.append(v1)
    v.append(v2)
    v.append(v3)
    d = []
    d.append(d0)
    d.append(d1)
    d.append(d2)
    d.append(d3)
    for i in range(grover_it):
        circuit = grover_step(q, c, circuit, v, d, x, ancilla, h, rot, rho, iv)
    return circuit
