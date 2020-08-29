[![DOI](https://zenodo.org/badge/265590690.svg)](https://zenodo.org/badge/latestdoi/265590690)
# Quantum Search for Scaled Hash Function Preimages

## Circuit specifications

The following scripts return a circuit analysis and gate count for both toy versions and real implementations of the Sponge Hash and Blake2 constructions presented in the paper. They also create a QASM file with the gates needed to apply a single Grover step for the designated Hash function.

The scripts to run are:
- `sponge_toy_main.py`
- `sponge_full_main.py` 
- `blake_toy_main.py` 
- `blake_full_main.py` 

Running the `main` script for each different version of the Hash implemlementation returns:
- A summary of the circuit needed to implement a Grover step.
  - Total number of gates. Specified as here as `Circuit depth` as classically one cannot apply quantum gates in parallel.
  - Total number of qubits used.
  - Breakdown of the type and number of basic gates used in the circuit.
- `{}_{}_gate_list.qasm`: QASM file containing all gates required for the implementation of a single Grover step.
