#!/bin/python3
# Script Name: parse_mhc.py
# Description: Parse MHC alleles for each sample
# Author: Kohan

import argparse


def ArgumentParser(args=None):
    parser = argparse.ArgumentParser(prog='Parse MHC alleles from results of HLA typing',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # required arguments
    parser.add_argument('in_file', type=str, help='Path to the HLA-HD result')
    parser.add_argument('out_file', type=str, help='Output path')
    
    return parser

# output of polysolver
def ParsePolysolver(file, genes=['A','B','C']):
    # read lines
    with open(file, 'r') as f:
        lines = [line.strip() for line in f.readlines()]
    
    # parse alleles
    alleles = dict()
    for line in lines:
        line = line.split('_')
        gene = line[1].upper()
        if gene not in genes:
            continue
        allele = f'{gene}*{line[2]}:{line[3]}'
        if f'{gene}_1' in alleles:
            alleles[f'{gene}_2'] = allele
        else:
            alleles[f'{gene}_1'] = allele
    
    return alleles


# output of HLA-HD ("-" means homozygous)
def ParseHLAHD(file, genes=['A', 'B', 'C', 'DRB1', 'DQA1', 'DQB1', 'DPA1', 'DPB1']):
    # read lines
    with open(file, 'r') as f:
        lines = [line.strip() for line in f.readlines()]

    # parse alleles
    alleles = dict()
    for line in lines:
        line = line.split('\t')
        gene = line[0]

        # check gene
        if gene not in genes:
            continue

        # allele 1
        candidate = line[1]
        if candidate == 'Not typed':
            a1 = ''
        else:
            codes = candidate.split('*')[1].split(':')
            a1 = f'{gene}*{codes[0]}:{codes[1]}'

        # allele 2
        candidate = line[2]
        if candidate == 'Not typed':
            a2 = ''
        elif candidate == '-':
            a2 = a1
        else:
            codes = candidate.split('*')[1].split(':')
            a2 = f'{gene}*{codes[0]}:{codes[1]}'
            
        alleles[f'{gene}_1'] = a1
        alleles[f'{gene}_2'] = a2

    return alleles


def ParseAlleleList(file, genes=['A', 'B', 'C', 'DRB1', 'DQA1', 'DQB1', 'DPA1', 'DPB1']):
    # read file
    with open(file, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    
    # group alleles by gene
    alleles = dict()
    for line in lines:
        gene = line.split('*')[0]
        if gene in genes:
            if alleles.get(f'{gene}_1'):
                alleles[f'{gene}_2'] = line
            else:
                alleles[f'{gene}_1'] = line
    
    # duplicate homozygous
    for gene in genes:
        if (not alleles.get(f'{gene}_2')) and alleles.get(f'{gene}_1'):
            alleles[f'{gene}_2'] = alleles[f'{gene}_1']
    
    return alleles


def Main(input_file, out_file, genes=['A', 'B', 'C', 'DRB1', 'DQA1', 'DQB1', 'DPA1', 'DPB1']):
    with open(input_file, 'r') as f:
        lines = f.readlines()
    
    if len(lines[0].split()) > 1:
        alleles = ParseHLAHD(input_file, genes=genes)
    else:
        alleles = ParseAlleleList(input_file, genes=genes)

    with open(out_file, 'w') as f:
        for gene in genes:
            if f'{gene}_1' in alleles:
                a1 = alleles[f'{gene}_1']
                f.write(f'{a1}\n')
            if f'{gene}_2' in alleles:
                a2 = alleles[f'{gene}_2']
                f.write(f'{a2}\n')


if __name__=='__main__':
    args = ArgumentParser().parse_args()
    Main(args.in_file, args.out_file)