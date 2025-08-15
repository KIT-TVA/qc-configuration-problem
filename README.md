<p align="center">
<img src="https://user-images.githubusercontent.com/263321/180779012-f2cad23b-0e27-4b78-a2e6-00426cf38e5f.png" alt="Quantum Banner">
</p>

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)

# Addressing the Configuration Problem via Quantum Computing

Repository containing data and code for Q-SE 2024 submission: Quantum Solution for Configuration Selection and Prioritization

## Installation

Install [python](https://www.python.org/downloads/) 3.9. Ideally, you should use a virtual environment for this project.
Clone (or download) this repository and install the [requirements](requirements.txt) via pip:

```
pip install -r requirements.txt
```

## Quantum Approximate Optimization Algorithm (QAOA)

This repository contains code and data for applying QAOA to different instances 
and solving configuration selection and prioritization problems.

There are multiple Jupyter notebooks in this repository (all inside the `configproblem/qaoa` directory):

- [qaoa_mincost_sat.ipynb](configproblem/qaoa/qaoa_mincost_sat.py) - QAOA for Minimum Cost 2-SAT explaining the algorithm for QUBO formulation
- [qaoa_mincost_k_sat.ipynb](configproblem/qaoa/qaoa_mincost_k_sat.ipynb) - QAOA for Minimum Cost k-SAT explaining the algorithm for more general PUBO formulation
- [qaoa_instance_evaluation.ipynb](configproblem/qaoa/qaoa_instance_evaluation.ipynb) - Generating different plots and histograms for evaluating results of QAOA

The first two notebooks contain detailed explanations of the algorithm and examples.

The accompanying python file of the first two notebooks
([qaoa_mincost_sat.py](configproblem/qaoa/qaoa_mincost_sat.py) and [qaoa_mincost_k_sat.py](configproblem/qaoa/qaoa_mincost_k_sat.py))
contains the different implementations of the phase seperating operator $U_C$ for the QUBO / PUBO formulation.

Additionally, you can use [evaluation.py](configproblem/qaoa/evaluation.py) 
to run the QAOA algorithm on a set of DIMACS files and save and process the results.

## Grover's Algorithm

This repository contains code and data for applying Grover's Algorithm to retrieve uniform random configuration samples.
See the Jupyter notebook [grover_sat.ipynb](configproblem/grover_sat.ipynb).

## Usage

1. Navigate to the root directory of this repository inside a terminal.
2. If you have a virtual environment set up, activate it.

### Notebooks

3. Run the `jupyter notebook` command. A browser window should open up.
4. In the browser window, navigate to the `configproblem` directory for the notebook about Grover's algorithm 
   or to the `configproblem/qaoa`directory for the notebooks about QAOA
   and open the notebook you want to run.

### Evaluation of QAOA using DIMACS files

Run `python -m configproblem.qaoa.evaluation <args>` with the following arguments:
   - `--first <index>` - Index of the first file to evaluate
   - `--last <index>` - Index of the last file to evaluate
   - `--skip-quso` - Skip running QAOA for the QUSO/QUBO formulation
   - `--skip-puso` - Skip running QAOA for the PUSO/PUBO formulation
   - `--save-individual-results` - Save the results of each file in a separate file 
      (so you can terminate the program before it finishes)
   - `-f <filename>` - Process the results of a previous run (saved in a file)
   - `--linux` - Use this flag to use Linux file paths (default is Windows)

DIMACS files have to be placed in the `benchmarks/qaoa-feature-models` directory
and have to be named `feature_model_<index>.dimacs` where `<index>` is the index of the file.

The files currently in the directory are the ones we used for our evaluation.

Results will be saved in `benchmarks/qaoa-feature-models/results`.
    

## Other repositories

Here is a list of tools that we used during our evaluation:

- [FeatureIDE](https://github.com/FeatureIDE/FeatureIDE) for generating feature models.
- [Qiskit](https://github.com/Qiskit) for creating and simulating quantum circuits.
- [Qubovert](https://github.com/jtiosue/qubovert) for conversion from and to QUBO and PUBO models.

## Citation

When publishing articles or otherwise writing about this work, please cite the following:

```bibtex
@inproceedings{10.1145/3643667.3648221,
author = {Ammermann, Joshua and Brenneisen, Fabian Jakob and Bittner, Tim and Schaefer, Ina},
title = {Quantum Solution for Configuration Selection and Prioritization},
year = {2024},
isbn = {9798400705700},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3643667.3648221},
doi = {10.1145/3643667.3648221},
abstract = {The analyses of highly configurable systems, as applied in software or automotive domains, yield hard problems due to the exponentially increasing number of possible product configurations. Current research identified that such combinatorial optimization problems, e.g. configuration selection and prioritization, are ideal targets for expected exponential quantum speedups. However, empirical evidence about the applicability of quantum computing to these problems is still missing. In this paper, we investigate how the constraint satisfaction and optimization problems of configuration selection and prioritization can be addressed using quantum computing. We propose a method to transform the configuration selection and prioritization problems encoded in attributed feature models into a quantum mechanical formulation suitable for optimization problems. We provide a Python library to automatically perform this transformation and apply the Quantum Approximate Optimization Algorithm (QAOA), such that configuration selection and prioritization are solved with quantum computers. Our approach is evaluated regarding feasibility, solution quality, and scalability. We show that QAOA obtains good results regarding configuration selection, but for configuration prioritization, the approach needs further improvement.},
booktitle = {Proceedings of the 5th ACM/IEEE International Workshop on Quantum Software Engineering},
pages = {21–28},
numpages = {8},
keywords = {configuration selection, configuration prioritization, QAOA},
location = {Lisbon, Portugal},
series = {Q-SE 2024}
}
```
