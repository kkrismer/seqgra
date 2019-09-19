#!/usr/bin/env python

'''
Gifford Lab - seq-grammar

@author: Konstantin Krismer
'''

import sys
import getopt

from seqgra.parser.parser import Parser
from seqgra.parser.xmlparser import XMLParser
from seqgra.simulator import Simulator

def parse_config_file(file_name: str) -> str:
    with open(file_name) as f:
        config: str = f.read()
    return config

def main(argv) -> None:
    usage: str = "parse_hic_file.py -c <config file> -o <output directory>"
    config_file: str = ""
    output_dir: str = ""
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
        parser: Parser = XMLParser(config)
        simulator = Simulator(parser)
        simulator.simulate_data(output_dir)
    else:
        print(usage)
        sys.exit(2)

if __name__ == "__main__":
    main(sys.argv[1:])
