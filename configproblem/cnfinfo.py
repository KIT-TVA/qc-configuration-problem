"""Script to quickly aggregate information about feature models in CNF file format"""
import argparse
from os import walk
from os.path import abspath, relpath, join, isfile
import json, csv

import util.dimacs_reader as reader
from util.cnf import CNF
from SolvingSATWithGrover import create_ksat_grover, collect_circuit_info

def parse_arguments():
    parser = argparse.ArgumentParser(description="Enter a directory of CNF files to get analytics")
    parser.add_argument('input')
    parser.add_argument('-o', '--output_directory', default=".")
    parser.add_argument('-n', '--output_filename', default="cnfinfo")
    parser.add_argument('--csv', action="store_true")
    parser.add_argument('--ssat', action="store_true")
    return parser.parse_args()


def main(args):
    if isfile(args.input):
        analytics = {args.input: analyze_cnf(abspath(args.input), args.ssat)}
    else:
        analytics = analyze_dirs(args.input, args.ssat)
    output(analytics, args.output_directory, args.output_filename)
    if args.csv:
        output_csv(analytics, args.output_directory, args.output_filename)


def analyze_dirs(path, enable_sharpsat):
    analytics = {}
    for dirpath, dirs, files in walk(path):
        # print("Current directory:", abspath(dirpath))
        # print("Directories:", dirs)
        # print("Files:", files)
        for file in files:
            file_abspath = join(abspath(dirpath), file)
            file_key = relpath(file_abspath, abspath(path))
            file_info = analyze_cnf(file_abspath, enable_sharpsat)
            analytics[file_key] = file_info
            return analytics

    return analytics


def analyze_cnf(cnf_file, enable_sharpsat):
    rd = reader.DimacsReader()
    exp_qubits = -1
    
    print(cnf_file)
    try:
        print("...Reading CNF")
        rd.fromFile(cnf_file)
        print(rd.features)
        exp_qubits = 1 + int(rd.nFeatures) + int(rd.nClauses)
    except KeyboardInterrupt:
        return {} # ignore errors in cnf parser
    
    n_solutions = -1
    k = 1
    if enable_sharpsat:
        try:
            print("...Counting Solutions")
            # TODO invoke sharp sat for solution counting, GANAK is probably the best choice
            # TODO derive k from n_solutions
            pass 
        except Exception:
            pass # do not calculate solutions

    # transform from DimacsReader to simplified problem instance
    print("...Transforming Problem")
    problem = CNF().from_dimacs(rd).to_problem()
    print(problem)
    print("...Create Quantum Circuit")
    quantum_circuit = create_ksat_grover(problem, k)
    
    print("...Collect Circuit Info")
    statevector_info = collect_circuit_info(quantum_circuit)
    # falcon_info = collect_circuit_info(quantum_circuit, TODO Falconr8)

    return {
        'nFeatures': rd.nFeatures, 
        'nClauses': rd.nClauses,
        'expQbits': exp_qubits,
        'nSolutions': n_solutions,
        'StateVectorWidth': statevector_info['width'],
        'StateVectorDepth': statevector_info['depth'],
        # 'FalconWidth': falcon_info['width'],
        # 'FalconDepth': falcon_info['depth'],
        }


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
            # TODO unpack qiskit data once its there



if __name__=="__main__":
    args = parse_arguments()
    main(args)
