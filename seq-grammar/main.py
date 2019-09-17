#!/usr/bin/env python

'''
Gifford Lab - seq-grammar

@author: Konstantin Krismer
'''

import sys
import getopt
from simulator import Simulator

def parse_config_file(file_name):
    with open(file_name) as f:
        config = f.read()
    return config

def main(argv):
    usage = "parse_hic_file.py -c <config file> -o <output directory>"
    config_file = ""
    output_dir = ""
    try:
        opts, _ = getopt.getopt(argv, "hc:o:", ["help", "configfile=", "outputfile="])
        if not opts:
            print('No options supplied')
            print(usage)
            sys.exit(2)
            
    except getopt.GetoptError as err:
        print(err)
        print(usage)
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print(usage)
            sys.exit()
        elif opt in ("-c", "--configfile"):
            config_file = arg
        elif opt in ("-o", "--outputfile"):
            output_file = arg
        else:
            print(usage)
            sys.exit(2)

    if config_file != "" and output_file != "":
        config = parse_config_file(config_file)
        simulator = Simulator(config, output_dir)
        simulator.simulate_data()
    else:
        print(usage)
        sys.exit(2)

if __name__ == "__main__":
    main(sys.argv[1:])
