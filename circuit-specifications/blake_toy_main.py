#!/usr/bin/env python
import numpy as np
import blake_toy_functions as functions


def main():
    """Circuit summary of a single step of Grover's algorithm for the toy Blake cryptographic scheme.
    Returns:
        blake_toy_gate_list.txt: gates required to perform a Grover step in QASM format.
        summary of the qubits and gates requiered for a Grover step.
    """
    q = 4
    m = 8
    rot = [1, 2]
    t = 0x0
    iv0 = 0x8
    iv1 = 0xB
    iv = []
    iv.append(iv0)
    iv.append(iv1)
    rho = 12
    h_value = 163
    h = "{0:0{bits}b}".format(h_value, bits=8)
    grover_it = int(1)
    circuit = functions.grover_single(q, iv, rot, rho, h, t, grover_it)
    gate_list = open("blake_toy_gate_list.qasm","w")
    gate_list.write(circuit.to_qasm())
    print('Single Grover step for the toy BLAKE hash function built.\n')
    print('QASM file containing all gates created.\n')
    print('The circuit specifications are:\n')
    print('-'*30)
    print(circuit.summary)
    print('-'*30)


if __name__ == "__main__":
    main()
