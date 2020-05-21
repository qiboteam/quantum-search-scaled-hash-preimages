[![DOI](https://zenodo.org/badge/265590690.svg)](https://zenodo.org/badge/latestdoi/265590690)
# Quantum Search for Scaled Hash Function Preimages
### OpenQASM 2.0 files

This repository contains examples for the Grover search implementations of scaled down versions of Chacha based Sponge hash functions and Blake2 construction.

The OpenQASM convertion has been possible thanks to the Qibo quantum simulation language.

## Toy Sponge Hash Grover circuit

Circuit that implements the full Grover's search algorithm for a hash value of: 10100011 (163). This particular value has 2 preimages, and the algorithm is constructed to have the exact number of Grover steps in order to find solution with unit probability. 

The circuit code is in [qasm-sponge-grover-circuit.qasm](https://github.com/Quantum-TII/quantum-search-scaled-hash-preimages/blob/layout/qasm-sponge-grover-circuit.qasm).

## Toy Blake2 Hash Grover step

Circuit that implements the initialization and a Grover step of the scaled down Blake2 Hash construction. This algorithm requires more qubits than normally simulable by classical means. This circuit is an illustration of what a single Grover step for this construction would look like. Hash value to find preimages to: 10100011 (163). 

The circuit code is in [qasm-blake2-grover-step.qasm](https://github.com/Quantum-TII/quantum-search-scaled-hash-preimages/blob/layout/qasm-blake2-grover-step.qasm).
