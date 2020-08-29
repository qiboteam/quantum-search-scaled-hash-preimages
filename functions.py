from qibo.models import Circuit
from qibo import gates
import numpy as np
import matplotlib.pyplot as plt


def error_gate(qubit, err, err_type='bitphaseflip'):
    """Gate that implements different types of Pauli errors.
    Args:
        qubit (int): qubit number where to apply error.
        err (float): error probability.
        err_type (str): type of error to simulate.
        
    Returns:
        generator with error gates if given by probability.
    """
    if err != 0:
        if err_type == 'bitflip':
            if np.random.random() <= err:
                yield gates.X(qubit)
        elif err_type == 'phaseflip':
            if np.random.random() <= err:
                yield gates.Z(qubit)
        elif err_type == 'bitphaseflip':
            if np.random.random() <= err:
                yield gates.X(qubit)
            if np.random.random() <= err:
                yield gates.Z(qubit)
            if np.random.random() <= err:
                yield gates.Y(qubit)


def n_mCNOT(controls, target, work, err):
    """Decomposition of a multi-controlled NOT gate with m qubits of work space.
    Args:
        controls (list): quantum register used as a control for the gate.
        target (int): qubit where the NOT gate is applied.
        work (list): quantum register used as work space.
        err (float): error probability.

    Returns:
        quantum gate generator for the multi-controlled NOT gate with m qubits of work space.
    """
    i = 0
    yield gates.TOFFOLI(controls[-1], work[-1], target)
    yield error_gate(controls[-1], err)
    yield error_gate(work[-1], err)
    yield error_gate(target, err)
    for i in range(1,len(controls)-2):
        yield gates.TOFFOLI(controls[-1-i], work[-1-i], work[-1-i+1])
        yield error_gate(controls[-1-i], err)
        yield error_gate(work[-1-i], err)
        yield error_gate(work[-1-i+1], err)
    yield gates.TOFFOLI(controls[0], controls[1], work[-1-i])
    yield error_gate(controls[0], err)
    yield error_gate(controls[1], err)
    yield error_gate(work[-1-i], err)
    for i in reversed(range(1,len(controls)-2)):
        yield gates.TOFFOLI(controls[-1-i], work[-1-i], work[-1-i+1])
        yield error_gate(controls[-1-i], err)
        yield error_gate(work[-1-i], err)
        yield error_gate(work[-1-i+1], err)
    yield gates.TOFFOLI(controls[-1], work[-1], target)
    yield error_gate(controls[-1], err)
    yield error_gate(work[-1], err)
    yield error_gate(target, err)
    for i in range(1,len(controls)-2):
        yield gates.TOFFOLI(controls[-1-i], work[-1-i], work[-1-i+1])
        yield error_gate(controls[-1-i], err)
        yield error_gate(work[-1-i], err)
        yield error_gate(work[-1-i+1], err)
    yield gates.TOFFOLI(controls[0], controls[1], work[-1-i])
    yield error_gate(controls[0], err)
    yield error_gate(controls[1], err)
    yield error_gate(work[-1-i], err)
    for i in reversed(range(1,len(controls)-2)):
        yield gates.TOFFOLI(controls[-1-i], work[-1-i], work[-1-i+1])
        yield error_gate(controls[-1-i], err)
        yield error_gate(work[-1-i], err)
        yield error_gate(work[-1-i+1], err)


def n_2CNOT(controls, target, work, err):
    """Decomposition up to Toffoli gates of a multi-controlled NOT gate with one work qubit.
    Args:
        controls (list): quantum register used as a control for the gate.
        target (int): qubit where the NOT gate is applied.
        work (int): qubit used as work space.
        err (float): error probability.

    Returns:
        quantum gate generator for the multi-controlled NOT gate with one work qubit.
    """
    m1 = int(((len(controls)+2)/2)+0.5)
    m2 = int(len(controls)+2-m1-1)
    yield n_mCNOT(controls[0:m1], work, controls[m1:len(controls)]+[target], err)
    yield n_mCNOT((controls+[work])[m1:m1+m2], target, controls[0:m1], err)
    yield n_mCNOT(controls[0:m1], work, controls[m1:len(controls)]+[target], err)
    yield n_mCNOT((controls+[work])[m1:m1+m2], target, controls[0:m1], err)


def adder_mod2n(a, b, x, err):
    """Quantum circuit for the adder modulo 2^n operation.
    Args:
        a (list): quantum register for the first number to be added.
        b (list): quantum register for the second number to be added, will be replaced by solution.
        x (int): ancillary qubit needed for the adder circuit.
        err (float): error probability.

    Returns:
        quantum gate generator that applies the quantum gates for addition modulo 2^n.
    """
    n = int(len(a))
    for i in range(n-2, -1, -1):
        yield gates.CNOT(a[i], b[i])
        yield error_gate(a[i], err)
        yield error_gate(b[i], err)
    yield gates.CNOT(a[n-2], x)
    yield error_gate(a[n-2], err)
    yield error_gate(x, err)
    yield gates.TOFFOLI(a[n-1], b[n-1], x)
    yield error_gate(a[n-1], err)
    yield error_gate(b[n-1], err)
    yield error_gate(x, err)
    yield gates.CNOT(a[n-3], a[n-2])
    yield error_gate(a[n-3], err)
    yield error_gate(a[n-2], err)
    yield gates.TOFFOLI(x, b[n-2], a[n-2])
    yield error_gate(x, err)
    yield error_gate(b[n-2], err)
    yield error_gate(a[n-2], err)
    yield gates.CNOT(a[n-4], a[n-3])
    yield error_gate(a[n-4], err)
    yield error_gate(a[n-3], err)
    for i in range(n-3, 1, -1):
        yield gates.TOFFOLI(a[i+1], b[i], a[i])
        yield error_gate(a[i+1], err)
        yield error_gate(b[i], err)
        yield error_gate(a[i], err)
        yield gates.CNOT(a[i-2], a[i-1])
        yield error_gate(a[i-2], err)
        yield error_gate(a[i-1], err)
    yield gates.TOFFOLI(a[2], b[1], a[1])
    yield error_gate(a[2], err)
    yield error_gate(b[1], err)
    yield error_gate(a[1], err)
    for i in range(n-2, 0, -1):
        yield gates.X(b[i])
        yield error_gate(b[i], err)
    yield gates.CNOT(x, b[n-2])
    yield error_gate(x, err)
    yield error_gate(b[n-2], err)
    for i in range(n-3,  -1, -1):
        yield gates.CNOT(a[i+1], b[i])
        yield error_gate(a[i+1], err)
        yield error_gate(b[i], err)
    yield gates.TOFFOLI(a[2], b[1], a[1])
    yield error_gate(a[2], err)
    yield error_gate(b[1], err)
    yield error_gate(a[1], err)
    for i in range(2, n-2):
        yield gates.TOFFOLI(a[i+1], b[i], a[i])
        yield error_gate(a[i+1], err)
        yield error_gate(b[i], err)
        yield error_gate(a[i], err)
        yield gates.CNOT(a[i-2], a[i-1])
        yield error_gate(a[i-2], err)
        yield error_gate(a[i-1], err)
        yield gates.X(b[i-1])
        yield error_gate(b[i-1], err)
    yield gates.TOFFOLI(x, b[n-2], a[n-2])
    yield error_gate(x, err)
    yield error_gate(b[n-2], err)
    yield error_gate(a[n-2], err)
    yield gates.CNOT(a[n-4], a[n-3])
    yield error_gate(a[n-4], err)
    yield error_gate(a[n-3], err)
    yield gates.X(b[n-3])
    yield error_gate(b[n-3], err)
    yield gates.TOFFOLI(a[n-1], b[n-1], x)
    yield error_gate(a[n-1], err)
    yield error_gate(b[n-1], err)
    yield error_gate(x, err)
    yield gates.CNOT(a[n-3], a[n-2])
    yield error_gate(a[n-3], err)
    yield error_gate(a[n-2], err)
    yield gates.X(b[n-2])
    yield error_gate(b[n-2], err)
    yield gates.CNOT(a[n-2], x)
    yield error_gate(a[n-2], err)
    yield error_gate(x, err)
    for i in range(n-1, -1, -1):
        yield gates.CNOT(a[i], b[i])
        yield error_gate(a[i], err)
        yield error_gate(b[i], err)


def r_adder_mod2n(a, b, x, err):
    """Reversed quantum circuit for the adder modulo 2^n operation.
    Args:
        a (list): quantum register for the first number to be added.
        b (list): quantum register for result of the addition.
        x (int): ancillary qubit needed for the adder circuit.
        err (float): error probability.

    Returns:
        quantum gate generator that applies the quantum gates for addition modulo 2^n in reverse.
    """
    n = int(len(a))
    for i in reversed(range(n-1, -1, -1)):
        yield gates.CNOT(a[i], b[i])
        yield error_gate(a[i], err)
        yield error_gate(b[i], err)
    yield gates.CNOT(a[n-2], x)
    yield error_gate(a[n-2], err)
    yield error_gate(x, err)
    yield gates.X(b[n-2])
    yield error_gate(b[n-2], err)
    yield gates.CNOT(a[n-3], a[n-2])
    yield error_gate(a[n-3], err)
    yield error_gate(a[n-2], err)
    yield gates.TOFFOLI(a[n-1], b[n-1], x)
    yield error_gate(a[n-1], err)
    yield error_gate(b[n-1], err)
    yield error_gate(x, err)
    yield gates.X(b[n-3])
    yield error_gate(b[n-3], err)
    yield gates.CNOT(a[n-4], a[n-3])
    yield error_gate(a[n-4], err)
    yield error_gate(a[n-3], err)
    yield gates.TOFFOLI(x, b[n-2], a[n-2])
    yield error_gate(x, err)
    yield error_gate(b[n-2], err)
    yield error_gate(a[n-2], err)
    for i in reversed(range(2, n-2)):
        yield gates.X(b[i-1])
        yield error_gate(b[i-1], err)
        yield gates.CNOT(a[i-2], a[i-1])
        yield error_gate(a[i-2], err)
        yield error_gate(a[i-1], err)
        yield gates.TOFFOLI(a[i+1], b[i], a[i])
        yield error_gate(a[i+1], err)
        yield error_gate(b[i], err)
        yield error_gate(a[i], err)
    yield gates.TOFFOLI(a[2], b[1], a[1])
    yield error_gate(a[2], err)
    yield error_gate(b[1], err)
    yield error_gate(a[1], err)
    for i in reversed(range(n-3,  -1, -1)):
        yield gates.CNOT(a[i+1], b[i])
        yield error_gate(a[i+1], err)
        yield error_gate(b[i], err)
    yield gates.CNOT(x, b[n-2])
    yield error_gate(x, err)
    yield error_gate(b[n-2], err)
    for i in reversed(range(n-2, 0, -1)):
        yield gates.X(b[i])
        yield error_gate(b[i], err)
    yield gates.TOFFOLI(a[2], b[1], a[1])
    yield error_gate(a[2], err)
    yield error_gate(b[1], err)
    yield error_gate(a[1], err)
    for i in reversed(range(n-3, 1, -1)):
        yield gates.CNOT(a[i-2], a[i-1])
        yield error_gate(a[i-2], err)
        yield error_gate(a[i-1], err)
        yield gates.TOFFOLI(a[i+1], b[i], a[i])
        yield error_gate(a[i+1], err)
        yield error_gate(b[i], err)
        yield error_gate(a[i], err)
    yield gates.CNOT(a[n-4], a[n-3])
    yield error_gate(a[n-4], err)
    yield error_gate(a[n-3], err)
    yield gates.TOFFOLI(x, b[n-2], a[n-2])
    yield error_gate(x, err)
    yield error_gate(b[n-2], err)
    yield error_gate(a[n-2], err)
    yield gates.CNOT(a[n-3], a[n-2])
    yield error_gate(a[n-3], err)
    yield error_gate(a[n-2], err)
    yield gates.TOFFOLI(a[n-1], b[n-1], x)
    yield error_gate(a[n-1], err)
    yield error_gate(b[n-1], err)
    yield error_gate(x, err)
    yield gates.CNOT(a[n-2], x)
    yield error_gate(a[n-2], err)
    yield error_gate(x, err)
    for i in reversed(range(n-2, -1, -1)):
        yield gates.CNOT(a[i], b[i])
        yield error_gate(a[i], err)
        yield error_gate(b[i], err)


def qr(a, b, x, rot, err):
    """Circuit for the quantum quarter round for the toy Chacha permutation.
    Args:
        a (list): quantum register of a site in the permutation matrix.
        b (list): quantum register of a site in the permutation matrix.
        x (int): ancillary qubit needed for the adder circuit.
        rot (list): characterization of the rotation part of the algorithm.
        err (float): error probability.

    Returns:
        quantum gate generator that applies the quantum gates for the Chacha quarter round.
    """
    n = int(len(a))
    for r in range(len(rot)):
        yield adder_mod2n(b, a, x, err)
        for i in range(n):
            yield gates.CNOT(a[i], b[i])
            yield error_gate(a[i], err)
            yield error_gate(b[i], err)
        for i in range(rot[r]):
            b = b[1:] + [b[0]]


def r_qr(a, b, x, rot, err):
    """Reverse circuit for the quantum quarter round for the toy Chacha permutation.
    Args:
        a (list): quantum register of a site in the permutation matrix.
        b (list): quantum register of a site in the permutation matrix.
        x (int): ancillary qubit needed for the adder circuit.
        rot (list): characterization of the rotation part of the algorithm.
        err (float): error probability.

    Returns:
        quantum gate generator that applies the reversed quantum gates for the Chacha quarter round.
    """
    n = int(len(a))
    for r in reversed(range(len(rot))):
        for i in range(rot[r]):
            b = [b[-1]] + b[:-1]
        for i in reversed(range(n)):
            yield gates.CNOT(a[i], b[i])
            yield error_gate(a[i], err)
            yield error_gate(b[i], err)
        yield r_adder_mod2n(b, a, x, err)


def diffuser(q, work, err):
    """Generator that performs the inversion over the average step in Grover's search algorithm.
    Args:
        q (list): quantum register that encodes the problem.
        work (int): ancilliary qubit used for the multi-controlled gate.
        err (float): error probability.

    Returns:
        quantum gate generator that applies the diffusion step.
    """
    n = len(q)
    for i in range(n):
        yield gates.H(q[i])
        yield error_gate(q[i], err)
        yield gates.X(q[i])
        yield error_gate(q[i], err)
    yield gates.H(q[0])
    yield error_gate(q[0], err)
    yield n_2CNOT(q[1:n], q[0], work, err)
    yield gates.H(q[0])
    yield error_gate(q[0], err)
    for i in range(n):
        yield gates.X(q[i])
        yield error_gate(q[i], err)
        yield gates.H(q[i])
        yield error_gate(q[i], err)


def start_grover(q, ancilla, err):
    """Generator that performs the starting step in Grover's search algorithm.
    Args:
        q (list): quantum register that encodes the problem.
        ancilla (int): Grover ancillary qubit.
        err (float): error probability.

    Returns:
        quantum gate generator for the first step of Grover.
    """
    n = len(q)
    yield gates.X(ancilla)
    yield error_gate(ancilla, err)
    yield gates.H(ancilla)
    yield error_gate(ancilla, err)
    for i in range(n):
        yield gates.H(q[i])
        yield error_gate(q[i], err)


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
    A = [i for i in range(q)]
    B = [i+q for i in range(q)]
    C = [i+2*q for i in range(q)]
    D = [i+3*q for i in range(q)]
    x = 4*q
    ancilla = 4*q+1
    qubits = 4*q+2
    circuit = Circuit(qubits)
    return A, B, C, D, x, ancilla, circuit, qubits


def chacha_qr(q, A, B, rot):
    """Classical implementation of the Chacha quarter round
    Args:
        q (int): number of bits of a site in the permutation matrix.
        A (str): classical bitstring for a site of the permutation matrix.
        B (str): classical bitstring for a site of the permutation matrix.
        rot (list): characterization of the rotation part of the algorithm.
            
    Returns:
        A (str): updated classical bitstring for a site of the permutation matrix.
        B (str): updated classical bitstring for a site of the permutation matrix.
    """
    for r in range(len(rot)):
        a = 0
        b = 0
        for i in range(q):
            a += int(A[i])*2**(q-1-i)
            b += int(B[i])*2**(q-1-i)
        a = (a + b) % (2**q)
        b = b ^ a
        A = "{0:0{bits}b}".format(a, bits=q)
        B = "{0:0{bits}b}".format(b, bits=q)
        for i in range(rot[r]):
                B = B[1:] + B[0]
    return A, B


def initial_step(q, constant_1, constant_2, rot):
    """Perform the first step of the algorithm classically.
    Args:
        q (int): number of qubits of a site in the permutation matrix.
        constant_1 (int): constant that defines the hash construction.
        constant_2 (int): constant that defines the hash construction.
        rot (list): characterization of the rotation part of the algorithm.
            
    Returns:
        a (str): classical bitstring for a site of the permutation matrix.
        b (str): classical bitstring for a site of the permutation matrix.
        c (str): classical bitstring for a site of the permutation matrix.
        d (str): classical bitstring for a site of the permutation matrix.
    """
    a = "{0:0{bits}b}".format(0, bits=q)
    b = "{0:0{bits}b}".format(0, bits=q)
    c = "{0:0{bits}b}".format(constant_1, bits=q)
    d = "{0:0{bits}b}".format(constant_2, bits=q)
    for i in range(10):
        a, c = chacha_qr(q, a, c, rot)
        b, d = chacha_qr(q, b, d, rot)
        a, d = chacha_qr(q, a, d, rot)
        b, c = chacha_qr(q, b, c, rot)
    return a, b, c, d


def QhaQha(q, A, B, C, D, x, rot, err):
    """Circuit that performs the quantum Chacha permutation for the toy model.
    Args:
        q (int): number of qubits of a site in the permutation matrix.
        A (list): quantum register of a site in the permutation matrix.
        B (list): quantum register of a site in the permutation matrix.
        C (list): quantum register of a site in the permutation matrix.
        D (list): quantum register of a site in the permutation matrix.
        x (int): ancillary qubit needed for the adder circuit.
        rot (list): characterization of the rotation part of the algorithm.
        err (float): error probability.
            
    Returns:
        generator that applies the ChaCha permutation as a quantum circuit
    """
    for i in range(10):
        yield qr(A, C, x, rot, err)
        for i in range(sum(rot)):
            C = C[1:] + [C[0]]
        yield qr(B, D, x, rot, err)
        for i in range(sum(rot)):
            D = D[1:] + [D[0]]
        yield qr(A, D, x, rot, err)
        for i in range(sum(rot)):
            D = D[1:] + [D[0]]
        yield qr(B, C, x, rot, err)
        for i in range(sum(rot)):
            C = C[1:] + [C[0]]


def r_QhaQha(q, A, B, C, D, x, rot, err):
    """Reversed circuit that performs the quantum Chacha permutation for the toy model.
    Args:
        q (int): number of qubits of a site in the permutation matrix.
        A (list): quantum register of a site in the permutation matrix.
        B (list): quantum register of a site in the permutation matrix.
        C (list): quantum register of a site in the permutation matrix.
        D (list): quantum register of a site in the permutation matrix.
        x (int): ancillary qubit needed for the adder circuit.
        rot (list): characterization of the rotation part of the algorithm.
        err (float): error probability.
            
    Returns:
        generator that applies the reverse ChaCha permutation as a quantum circuit
    """
    for i in range(10):
        yield r_qr(B, C, x, rot, err)
        for i in range(sum(rot)):
            C = [C[-1]] + C[:-1]
        yield r_qr(A, D, x, rot, err)
        for i in range(sum(rot)):
            D = [D[-1]] + D[:-1]
        yield r_qr(B, D, x, rot, err)
        for i in range(sum(rot)):
            D = [D[-1]] + D[:-1]
        yield r_qr(A, C, x, rot, err)
        for i in range(sum(rot)):
            C = [C[-1]] + C[:-1]   


def grover_step(q, c, circuit, A, B, C, D, x, ancilla, h, rot, err):
    """Add a full grover step to solve a Sponge Hash construction to a quantum circuit.
    Args:
        q (int): number of qubits of a site in the permutation matrix.
        c (list): classical register that contains the initial step
        circuit (Circuit): quantum circuit where the Grover step is added.
        A (list): quantum register of a site in the permutation matrix.
        B (list): quantum register of a site in the permutation matrix.
        C (list): quantum register of a site in the permutation matrix.
        D (list): quantum register of a site in the permutation matrix.
        h (str): hash value that one wants to find preimages of.
        rot (list): characterization of the rotation part of the algorithm.
        err (float): error probability.
            
    Returns:
        circuit (Circuit): quantum circuit where the Grover step is added.
    """
    n = int(len(h))
    for i in range(q):
        if int(c[0][i]) == 1:
            circuit.add(gates.X(A[i]))
            circuit.add(error_gate(A[i], err))
        if int(c[1][i]) == 1:
            circuit.add(gates.X(B[i]))
            circuit.add(error_gate(B[i], err))
        if int(c[2][i]) == 1:
            circuit.add(gates.X(C[i]))
            circuit.add(error_gate(C[i], err))
        if int(c[3][i]) == 1:
            circuit.add(gates.X(D[i]))
            circuit.add(error_gate(D[i], err))
    circuit.add(QhaQha(q, A, B, C, D, x, rot, err))
    for i in range(10):
        for i in range(sum(rot)):
            C = C[1:] + [C[0]]
        for i in range(sum(rot)):
            D = D[1:] + [D[0]]
        for i in range(sum(rot)):
            D = D[1:] + [D[0]]
        for i in range(sum(rot)):
            C = C[1:] + [C[0]]
    for i in range(n):
        if int(h[i]) != 1:
            circuit.add(gates.X((A+B)[i]))
            circuit.add(error_gate((A+B)[i], err))
    circuit.add(n_2CNOT((A+B)[:n], ancilla, x, err))
    for i in range(n):
        if int(h[i]) != 1:
            circuit.add(gates.X((A+B)[i]))
            circuit.add(error_gate((A+B)[i], err))
    circuit.add(r_QhaQha(q, A, B, C, D, x, rot, err))
    for i in range(10):
        for i in range(sum(rot)):
            C = [C[-1]] + C[:-1]
        for i in range(sum(rot)):
            D = [D[-1]] + D[:-1]
        for i in range(sum(rot)):
            D = [D[-1]] + D[:-1]
        for i in range(sum(rot)):
            C = [C[-1]] + C[:-1]
    for i in range(q):
        if int(c[0][i]) == 1:
            circuit.add(gates.X(A[i]))
            circuit.add(error_gate(A[i], err))
        if int(c[1][i]) == 1:
            circuit.add(gates.X(B[i]))
            circuit.add(error_gate(B[i], err))
        if int(c[2][i]) == 1:
            circuit.add(gates.X(C[i]))
            circuit.add(error_gate(C[i], err))
        if int(c[3][i]) == 1:
            circuit.add(gates.X(D[i]))
            circuit.add(error_gate(D[i], err))
    circuit.add(diffuser(A+B, x, err))
    return circuit


def check_hash(q, message, h, constant_1, constant_2, rot):
    """Check if a given output message is a preimage of a given hash value.
    Args:
        q (int): number of qubits of a site in the permutation matrix.
        message (str): output message that we want to check.
        h (str): hash value that one wants to find preimages of.
        constant_1 (int): constant that defines the hash construction.
        constant_2 (int): constant that defines the hash construction.
        rot (list): characterization of the rotation part of the algorithm.
            
    Returns:
        True of False if the message correspongs to a preimage.
    """
    n = int(len(h))
    m1 = 0
    m2 = 0
    for i in range(q):
        m1 += int(message[i])*2**(q-1-i)
        m2 += int(message[q+i])*2**(q-1-i)
    a = "{0:0{bits}b}".format(0, bits=q)
    b = "{0:0{bits}b}".format(0, bits=q)
    c = "{0:0{bits}b}".format(constant_1, bits=q)
    d = "{0:0{bits}b}".format(constant_2, bits=q)
    for i in range(10):
        a, c = chacha_qr(q, a, c, rot)
        b, d = chacha_qr(q, b, d, rot)
        a, d = chacha_qr(q, a, d, rot)
        b, c = chacha_qr(q, b, c, rot)
    A = 0
    B = 0
    for i in range(q):
        A += int(a[i])*2**(q-1-i)
        B += int(b[i])*2**(q-1-i)
    A = A ^ m1
    B = B ^ m2
    a = "{0:0{bits}b}".format(A, bits=q)
    b = "{0:0{bits}b}".format(B, bits=q)
    for i in range(10):
        a, c = chacha_qr(q, a, c, rot)
        b, d = chacha_qr(q, b, d, rot)
        a, d = chacha_qr(q, a, d, rot)
        b, c = chacha_qr(q, b, c, rot)
    output = (a+b)[:n]
    return h == output


def grover(q, constant_1, constant_2, rot, h, grover_it, nshots=100, err=0):
    """Run the full Grover's search algorithm to find a preimage of a hash function.
    Args:
        q (int): number of qubits of a site in the permutation matrix.
        constant_1 (int): constant that defines the hash construction.
        constant_2 (int): constant that defines the hash construction.
        rot (list): characterization of the rotation part of the algorithm.
        h (str): hash value that one wants to find preimages of.
        grover_it (int): number of Grover steps to be performed.
        nshots (int): number of shots to perform on the final state.
        err (float): error probability.
            
    Returns:
        result (dict): counts of the output generated by the algorithm
    """
    A, B, C, D, x, ancilla, circuit, qubits = create_qc(q)
    c1, c2, c3, c4 = initial_step(q, constant_1, constant_2, rot)
    c = []
    c.append(c1)
    c.append(c2)
    c.append(c3)
    c.append(c4)
    circuit.add(start_grover(A+B, ancilla, err))
    for i in range(grover_it):
        circuit = grover_step(q, c, circuit, A, B, C, D, x, ancilla, h, rot, err)
    circuit.add(gates.M(*(A+B), register_name='preimages'))
    result = circuit(nshots=nshots)
    return result.frequencies(binary=True)


def noisy_grover(q, constant_1, constant_2, rot, h, grover_it, err):
    """Run the full Grover's search algorithm with errors to find a preimage of a hash function.
    Args:
        q (int): number of qubits of a site in the permutation matrix.
        constant_1 (int): constant that defines the hash construction.
        constant_2 (int): constant that defines the hash construction.
        rot (list): characterization of the rotation part of the algorithm.
        h (str): hash value that one wants to find preimages of.
        grover_it (int): number of Grover steps to be performed.
        err (float): error probability.
            
    Returns:
        result (dict): counts of the output generated by the algorithm
    """
    A, B, C, D, x, ancilla, circuit, qubits = create_qc(q)
    c1, c2, c3, c4 = initial_step(q, constant_1, constant_2, rot)
    c = []
    c.append(c1)
    c.append(c2)
    c.append(c3)
    c.append(c4)
    circuit.add(start_grover(A+B, ancilla, err))
    for i in range(grover_it):
        circuit = grover_step(q, c, circuit, A, B, C, D, x, ancilla, h, rot, err)
    circuit.add(gates.M(*(A+B), register_name='AB'))
    result = circuit(nshots=1)
    return result.samples(binary=True, registers=False)


def grover_unknown_M(q, constant_1, constant_2, rot, h):
    """Run an iterative Grover's search algorithm to find a preimage of a hash function when the
           total number of solutions is unknown.
    Args:
        q (int): number of qubits of a site in the permutation matrix.
        constant_1 (int): constant that defines the hash construction.
        constant_2 (int): constant that defines the hash construction.
        rot (list): characterization of the rotation part of the algorithm.
        h (str): hash value that one wants to find preimages of.
            
    Returns:
        measured (str): output of a preimage for the given hash value.
        total_iterations: number of total Grover steps performed to encounter a solution.
    """
    k = 1
    lamda = 6/5
    total_iterations = 0
    while True:
        it = np.random.randint(k+1)
        if it != 0:
            total_iterations += it
            result = grover(q, constant_1, constant_2, rot, h, it, nshots=1)
            measured = result.most_common(1)[0][0]
            if check_hash(q, measured, h, constant_1, constant_2, rot):
                break
        k = min(lamda*k, np.sqrt(2**(2*q)))
    return measured, total_iterations


def plot_noise(bit_err, success_total, success_half_twice, h):
    """Plot the success probability of Grover's algorithm under increasing noise.
    Args:
        bit_err (list): error probabilities simulated.
        success_total (list): success after applying full Grover.
        success_half_twice (list): success after applying half Grover twice.
        h (str): target hash value.
        
    Returns:
        grover_bitphase_{}.png image with the success probability under increasing Pauli noise.
    """
    fig = plt.figure(figsize = (10,6))
    plt.xlabel('bitflip error (%)', fontsize=14)
    plt.ylabel('success probability', fontsize=14)
    plt.scatter(bit_err, success_total, label='Full Grover (8 steps once)', s=70, color='C0', marker='x')
    plt.scatter(bit_err, success_half_twice, label='Half Grover (4 steps twice)', s=70, color='C3', marker='x')
    plt.legend(fontsize=14)
    plt.xlim((-1e-4, 5e-3+1e-4)) 
    plt.savefig('grover_bitphase_{}.png'.format(h), dpi=300, bbox_inches='tight')
    plt.show()
