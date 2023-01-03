"""Script to quickly aggregate information about feature models in CNF file format"""
import argparse, platform
from os import walk
from os.path import abspath, relpath, join, isfile
import json, csv, math

import util.dimacs_reader as reader
from util.cnf import CNF
from util.timeout_try import start_timeout, TimeoutError, reset_timer
from SolvingSATWithGrover import create_ksat_grover, collect_circuit_info

from qiskit.circuit.exceptions import CircuitError

args = None
# defube verbose print function that can silence print statements based on command line flags
vprint = print

def parse_arguments():
    parser = argparse.ArgumentParser(description="Enter a directory of CNF files to get analytics")
    parser.add_argument('input')
    parser.add_argument('-o', '--output_directory', default=".")
    parser.add_argument('-n', '--output_filename', default="cnfinfo")
    parser.add_argument('-q', '--quiet', action="store_true")
    parser.add_argument('--max_width', default=1e3) # maximum width below which detailed analyses are carried out
    parser.add_argument('--max_k', default=30) # limit for the circuit generation
    parser.add_argument('--circuit_timeout', default=60) # number of seconds before creating and simulating a single circuit times out and is skipped
    parser.add_argument('--csv', action="store_true")
    parser.add_argument('--ssat', action="store_true")
    
    global args 
    args = parser.parse_args()

    # check that the ssat argument is only issued on linux!
    if args.ssat:
        if platform.system() == "Linux":
            # Check availability of pyapproxmc
            init_sharpsat()
        else:
            raise RuntimeError("Sharp SAT model counting is only supported on Linux systems")

    # init verbose print
    global vprint
    vprint = print if not args.quiet else lambda *a, **k: None    

def init_sharpsat():
    try:
        import pyapproxmc
    except ImportError:
        print("\n\npyapproxmc not found, install it using `pip install pyapproxmc` on Linux distributions!")
        exit(-1)


def main():
    if isfile(args.input):
        analytics = {args.input: analyze_cnf(abspath(args.input))}
    else:
        analytics = analyze_dirs(args.input)
    output(analytics, args.output_directory, args.output_filename)
    if args.csv:
        output_csv(analytics, args.output_directory, args.output_filename)


def analyze_dirs(path):
    # start the analysis with a dummy to ensure all columns are present and methods are working
    dummy_path = abspath(join(__file__, '../../benchmarks/car.dimacs'))
    analytics = {"DUMMY_CAR": analyze_cnf(dummy_path)}
    for dirpath, dirs, files in walk(path):
        for file in files:
            file_abspath = join(abspath(dirpath), file)
            file_key = relpath(file_abspath, abspath(path))
            try:
                vprint("Analyzing: " + file_key)
                file_info = analyze_cnf(file_abspath)
                analytics[file_key] = file_info
            except KeyboardInterrupt:
                return analytics

    return analytics


def analyze_cnf(cnf_file):
    rd = reader.DimacsReader()
    exp_qubits = -1
    
    try:
        vprint("...Reading CNF")
        rd.fromFile(cnf_file)
        exp_qubits = 1 + int(rd.nFeatures) + int(rd.nClauses)
    except Exception:
        return {"_" : "CNF Parser Error"} # ignore errors in cnf parser

    if exp_qubits > args.max_width:
        return {'nFeatures': rd.nFeatures, 'nClauses': rd.nClauses, 'expQbits': exp_qubits}

    # transform from DimacsReader to simplified problem instance
    vprint("...Transforming Problem")
    problem = CNF().from_dimacs(rd).to_problem()
    
    n_solutions = -1
    solutions_percentage = -1
    k = -1

    # count solutions if argument is issued. Note that OS check is done directly after parsing arguments
    if args.ssat:
        try:
            vprint("...Counting Solutions", end="")
            # invoke sharp sat for solution counting, GANAK is probably the best choice
            n_solutions = count_solutions(problem)
            # derive k from n_solutions 
            solutions_percentage = round((n_solutions / (2**rd.nFeatures)) * 10000) / 100
            k = max(1, math.floor((math.pi / 4) * math.sqrt((2**rd.nFeatures)/n_solutions)))
            vprint(f": {n_solutions} ==> k = {k}")
        except Exception:
            vprint("Failed") # do not calculate solutions

    # limit k for circuit purposes but store the calculated value
    real_k = k
    if k > args.max_k or k < 1: 
        real_k = f"1 ({k})"
        k = 1

    # attempt circuit creation and simulation
    try:
        start_timeout(args.circuit_timeout)
        vprint("...Create Quantum Circuit")
        quantum_circuit, _ = create_ksat_grover(problem, k)

        vprint("...Collect Circuit Info")
        statevector_info = collect_circuit_info(quantum_circuit)
        # falcon_info = collect_circuit_info(quantum_circuit, TODO Falconr8)
    except TimeoutError:
        vprint("! Collecting info timed out !")
        statevector_info = {'width':-1, 'depth':-1}
    except CircuitError:
        vprint("! Creating the quantum circuit failed !")
        statevector_info = {'width':-1, 'depth':-1}
    finally:
        reset_timer()

    return {
        'nFeatures': rd.nFeatures, 
        'nClauses': rd.nClauses,
        'expQbits': exp_qubits,
        'nSolutions': n_solutions,
        'percentSolutions': solutions_percentage,
        'estimatedK': real_k,
        'StateVectorWidth': statevector_info['width'],
        'StateVectorDepth': statevector_info['depth'],
        # 'FalconWidth': falcon_info['width'],
        # 'FalconDepth': falcon_info['depth'],
    }


def count_solutions(problem):
    """
        Use approximate model counter to count solutions for the problem
    """
    import pyapproxmc
    c = pyapproxmc.Counter(epsilon=0.9, delta=0.5)
    for clause in problem:  # [(symbol, negated), ...]
        cl = []
        for variable in clause:  # (symbol, negated)
            val = variable[0] + 1
            if variable[1]:
                val *= -1
            cl.append(val)
        c.add_clause(cl)

    count = c.count() # (factor, two's-exponent)
    return count[0] * (2**count[1])


def output(analytics, output_dir='.', output_name='cnfinfo'):
    with open(join(abspath(output_dir), output_name + '.json'), "w") as f:
        json.dump(analytics, f)


def output_csv(analytics, output_dir='.', output_name='cnfinfo'):
    with open(join(abspath(output_dir), output_name + '.csv'), "w") as f:
        writer = csv.writer(f)
        # headline
        first_keys = next(iter(analytics.values()))
        writer.writerow(["Path"] + list(first_keys))
        # content
        for entry, value in analytics.items():
            writer.writerow([entry] + list(value.values()))


if __name__=="__main__":
    parse_arguments()
    main()
