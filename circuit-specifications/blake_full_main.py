#!/usr/bin/env python
import numpy as np
import blake_full_functions as functions
import argparse


def main():
    """Circuit summary of a single step of Grover's algorithm for the full Blake2 cryptographic scheme.
    Returns:
        blake_full_gate_list.txt: gates required to perform a Grover step in QASM format.
        summary of the qubits and gates requiered for a Grover step.
    """
    q = 64
    rot = [32, 24, 16, 63]
    t = 0x0
    iv = [
    0x6A09E667F3BCC908, 
    0xBB67AE8584CAA73B,
    0x3C6EF372FE94F82B, 
    0xA54FF53A5F1D36F1,
    0x510E527FADE682D1,
    0x9B05688C2B3E6C1F,
    0x1F83D9ABFB41BD6B, 
    0x5BE0CD19137E2179
    ]
    rho = 12
    h_value = 0xBB67AE8584CAA73B2423439478347223490728728238409834089120398401938401283409723509159823764528392347593284750294239840239429348123
    h = "{0:0{bits}b}".format(h_value, bits=512)
    grover_it = int(1)
    circuit = functions.grover_single(q, iv, rot, rho, h, t, grover_it)
    gate_list = open("blake_full_gate_list.qasm","w")
    gate_list.write(circuit.to_qasm())
    print('Built a single Grover step for the toy BLAKE hash function.\n')
    print('QASM file containing all gates created\n')
    print('The circuit specifications are:\n')
    print('-'*30)
    print(circuit.summary())
    print('-'*30)


if __name__ == "__main__":
    main()
