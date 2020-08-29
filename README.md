[![DOI](https://zenodo.org/badge/265590690.svg)](https://zenodo.org/badge/latestdoi/265590690)
# Quantum Search for Scaled Hash Function Preimages

This repository contains code to reproduce the result presented in the paper: Quantum Search for Scaled Hash Function Preimages.

The scripts presented use [Qibo](https://github.com/Quantum-TII/qibo), a Python library for classical simulation of quantum circuits. In order to properly run the scripts, please start by installing Qibo to your Python enviroment with `pip install qibo`.

Run the `main.py` file to find a preimage for hash value `10100011` without knowing the total amount of preimages. This code is based on the Toy Sponge Hash implementation described in the paper.

The following arguments can alter the program output:

- `--hash` (int) hash value whose preimages will be found. Has to be represented by less than 8 bits, and is `163` by default.
- `--bits` (int) minimum amount of bits to use when defining the hash value. Default value is 7.
- `--collisions` (int) if known, the number of preimages of target hash, to perform direct Grover over the iterative method. Hash `163` has 2 collisions, but is set to `None` by default.
- `--noise` perform a noisy simulation with Pauli errors after every gate and return a graph of success probability for increasing error.

The program returns:

- The target hash value used in the program.
- If as successful solution has been found.
  - **If number of collisions is not given:** a preimage of the function.
  - **If number of collisions is given:** all preimages of the function.
-  Total Grover iterations taken to find a primage.

If `--noise` was given:

- `grover_bitphase.png` image detailing the success probability of runing the full, or half Grover algorithm under increasing amounts of noise.

Circuit specifications and gate counts for a single Grover step for both toy models and real implementations of the Sponge Hash construction as well as the Blake2 hash construction are found in the **circuit-specifications** folder.
