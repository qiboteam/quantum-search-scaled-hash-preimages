#!/usr/bin/env python
import numpy as np
import sponge_full_functions as functions


def main():
    """Circuit summary of a single step of Grover's algorithm for the full Sponge cryptographic scheme.
    Returns:
        sponge_full_gate_list.txt: gates required to perform a Grover step in QASM format.
        summary of the qubits and gates requiered for a Grover step.
    """
    q = 32
    m = 8
    rot = [16, 12, 8, 7]
    state = [0]*16
    state[:4] = (1634760805, 857760878, 2036477234, 1797285236)
    state[4:12] = (0x03020100, 0x07060504, 0x0b0a0908, 0x0f0e0d0c, 0x13121110, 0x17161514, 0x1b1a1918, 0x1f1e1d1c)
    state[12] = 0x00000001
    state[13:16] = (0x09000000, 0x4a000000, 0x00000000)
    h_value = 0xBB67AE8584CA2349072823509864583347593284750294239840239429348123
    h = "{0:0{bits}b}".format(h_value, bits=256)
    grover_it = int(1)
    circuit = functions.grover_single(q, state, rot, h, grover_it)
    gate_list = open("sponge_full_gate_list.qasm","w")
    gate_list.write(circuit.to_qasm())
    print('Single Grover step for the toy sponge hash function built.\n')
    print('QASM file containing all gates created.\n')
    print('The circuit specifications are:\n')
    print('-'*30)
    print(circuit.summary)
    print('-'*30)


if __name__ == "__main__":
    main()
