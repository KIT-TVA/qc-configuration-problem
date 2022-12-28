"""Script to quickly aggregate information about feature models in CNF file format"""
import argparse
from os import walk
from os.path import abspath, relpath, join
import json

import reader

def parse_arguments():
    parser = argparse.ArgumentParser(description="Enter a directory of CNF files to get analytics")
    parser.add_argument('input_directory')
    parser.add_argument('-o', '--output_directory', default=".")
    parser.add_argument('-n', '--output_filename', default="cnfinfo")
    parser.add_argument('--csv', action="store_true")
    return parser.parse_args()


def main(args):
    analytics = walk_dir(args.input_directory)
    output(analytics, args.output_directory, args.output_filename)
    if args.csv:
        output_csv(analytics, args.output_directory, args.output_filename)


def walk_dir(path):
    analytics = {}
    for dirpath, dirs, files in walk(path):
        # print("Current directory:", abspath(dirpath))
        # print("Directories:", dirs)
        # print("Files:", files)
        for file in files:
            file_abspath = join(abspath(dirpath), file)
            file_key = relpath(file_abspath, abspath(path))
            file_info = analyze_cnf(file_abspath)
            analytics[file_key] = file_info
            # return analytics

    return analytics


def analyze_cnf(cnf_file):
    rd = reader.DimacsReader()
    print(cnf_file)
    try:
        rd.fromFile(cnf_file)
    except Exception:
        pass  # ignore errors in cnf parser

    return {
        'nFeatures': rd.nFeatures, 
        'nClauses': rd.nClauses
        }


def output(analytics, output_dir='.', output_name='cnfinfo'):
    with open(join(abspath(output_dir), output_name + '.json'), "w") as f:
        json.dump(analytics, f)


def output_csv(analytics, output_dir='.', output_name='cnfinfo'):
    with open(join(abspath(output_dir), output_name + '.csv'), "w") as f:
        # headline
        f.write("Path,nFeatures,nClauses\n")
        for entry, value in analytics.items():
            f.write(f"{entry},{value['nFeatures']},{value['nClauses']}\n")



if __name__=="__main__":
    args = parse_arguments()
    main(args)
