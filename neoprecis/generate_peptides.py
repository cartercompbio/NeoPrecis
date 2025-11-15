#!/bin/python3
# Script Name: generate_peptides.py
# Description: Generate mutated peptide and aligned wild-type peptide
# Author: Kohan

import sys
import os
import argparse
import gzip
import re
import pandas as pd

# Support both package import and direct script execution
try:
    from .api import ReadVCF, ReadMAF, PepGen, EpiGen, WritePeptideTXT
except ImportError:
    # Fallback for direct execution (python neoprecis/generate_peptides.py)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from api import ReadVCF, ReadMAF, PepGen, EpiGen, WritePeptideTXT


def ArgumentParser(args=None):
    parser = argparse.ArgumentParser(prog='Generate mutated peptide and aligned wild-type peptide',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # required arguments
    parser.add_argument('mut_file', type=str, help='Path to the VEP-annotated VCF or MAF file')
    parser.add_argument('out_prefix', type=str, help='Output prefix; ensure the output directory exists')
    
    # optional arguments
    parser.add_argument('--mhci_pept_lens', type=str, default='8,9,10,11', help='A list of MHC-I peptide lengths separated by ","')
    parser.add_argument('--mhcii_pept_lens', type=str, default='15', help='A list of MHC-II peptide lengths separated by ","')
    parser.add_argument('--cdna_file', type=str,
                        default='../data/Homo_sapiens.GRCh37.cdna.all.fa',
                        help='Path to the cDNA reference sequence file (FASTA)')
    parser.add_argument('--cds_file', type=str,
                        default='../data/Homo_sapiens.GRCh37.cds.all.fa',
                        help='Path to the CDS reference sequence file (FASTA)')
    
    return parser


# assign mutation ID for each row
def AssignMutID(row):
    hgvsp = str(row['HGVSp'])
    if re.match(r'.*:p\.', hgvsp):  # Match any chars followed by ':p.'
        mut = hgvsp.split(':')[1]
    elif re.match(r'^p\.', hgvsp):  # Match strings starting with 'p.'
        mut = hgvsp
    else:
        mut_aa = row['Amino_acids'].split('/')
        pos = row['Protein_position'].split('/')[0]  # if length provided (position/length)
        if len(mut_aa) == 2:
            mut = f"p.{mut_aa[0]}{pos}{mut_aa[1]}"
        else:
            mut = f"p.{pos}{mut_aa[0]}"
    return f'{row["SYMBOL"]}:{mut}'


def Main(mut_file, out_prefix, cdna_file, cds_file,
         mhci_pept_lens=[8,9,10,11], mhcii_pept_lens=[15]):
    ### inputs
    # load mutation file (auto-detect format)
    if mut_file.endswith('.gz'):
        with gzip.open(mut_file, 'rt') as f:
            first_line = f.readline()
    else:
        with open(mut_file, 'r') as f:
            first_line = f.readline()
    if first_line.startswith('##'):
        mut_df = ReadVCF(mut_file)
    else:
        mut_df = ReadMAF(mut_file)
    assert all(x in mut_df.columns for x in ['#CHROM',
                                             'POS',
                                             'REF',
                                             'ALT',
                                             'FILTER',
                                             'Transcript',
                                             'Consequence',
                                             'cDNA_position',
                                             'CDS_position',
                                             'Protein_position',
                                             'Amino_acids',
                                             'Codons',
                                             'SYMBOL',
                                             'HGVSc',
                                             'HGVSp'])
    # filter mutations
    if 'CANONICAL' in mut_df.columns:
        mut_df = mut_df[mut_df['CANONICAL']=='YES'] # keep canonical transcripts
    else:
        print('Warning: No CANONICAL column. Might have multiple annotations.')
    mut_df = mut_df[mut_df['FILTER']=='PASS'] # keep qualified mutations
    allowed_consequences = ['frameshift_variant', 'stop_lost', 'inframe_insertion', 'inframe_deletion', 'missense_variant', 'protein_altering_variant']
    mut_df = mut_df[mut_df['Consequence'].astype(str).str.contains('|'.join(allowed_consequences))] # keep nonsynonymous mutations
    mut_df = mut_df[~(mut_df['Codons']=='')] # drop mutations without codon changes
    if mut_df.shape[0] == 0:
        print('No non-synonymous mutation')
        sys.exit(1)
    # keep one transcript fo each mutation
    mut_df = mut_df.drop_duplicates(subset=['#CHROM', 'POS', 'REF', 'ALT'], keep='first')
    # check FASTA reference
    if not (os.path.isfile(cdna_file) and os.path.isfile(cds_file)):
        print('cDNA and CDS files are required if WT and MT columns not available')
        sys.exit(1)
    # add mutation ID
    mut_df['Mutation_ID'] = mut_df.apply(lambda row: AssignMutID(row), axis=1)
    
    ### WT and MT peptides
    max_len = max(mhci_pept_lens+mhcii_pept_lens)
    pepgen = PepGen(cdna_file, cds_file)
    wt_pept_list, mt_pept_list = list(), list()
    for idx, mutation in mut_df.iterrows():
        wt_pept, mt_pept = pepgen(mutation['Transcript'], mutation['CDS_position'], mutation['Codons'], mutation['Consequence'], length=max_len)
        wt_pept_list.append(wt_pept)
        mt_pept_list.append(mt_pept)
    mut_df['WT_seq'] = wt_pept_list
    mut_df['MT_seq'] = mt_pept_list
    mut_df = mut_df.reset_index(names=['original_idx'])
    mut_df.to_csv(f'{out_prefix}.mut.csv')

    ### WT and MT epitopes
    mhci_pepts, mhcii_pepts = set(), set()
    pept_lens = sorted(list(set(mhci_pept_lens) | set(mhcii_pept_lens)))
    for idx, mutation in mut_df.iterrows():
        print(f'Generate epitopes from WT `{mutation["WT_seq"]}` and MT `{mutation["MT_seq"]}`')
        # generate epitopes
        epigen = EpiGen(mutation['WT_seq'], mutation['MT_seq'], mutation['Consequence'])
        epi_df = pd.DataFrame()
        for length in pept_lens:
            tmp_df = epigen(length)
            epi_df = pd.concat([epi_df, tmp_df], axis=0)
        if epi_df.shape[0] == 0:
            print('Skip: no distinct mutated peptides')
            continue
        epi_df.to_csv(f'{out_prefix}.peptide{idx}.csv', index=False)
        # output epitopes for MHC-I prediction
        pepts = epigen.get_peptides(epi_df, mhci_pept_lens)
        mhci_pepts = mhci_pepts | set(pepts)
        # output epitopes for MHC-II prediction
        pepts = epigen.get_peptides(epi_df, mhcii_pept_lens)
        mhcii_pepts = mhcii_pepts | set(pepts)
        print('------------------------------------------------')
    WritePeptideTXT(list(mhci_pepts), f'{out_prefix}.peptide.mhci.txt')
    WritePeptideTXT(list(mhcii_pepts), f'{out_prefix}.peptide.mhcii.txt')


if __name__=='__main__':
    args = ArgumentParser().parse_args()
    mhci_pept_lens = [int(i) for i in args.mhci_pept_lens.split(',')]
    mhcii_pept_lens = [int(i) for i in args.mhcii_pept_lens.split(',')]
    Main(args.mut_file, args.out_prefix, args.cdna_file, args.cds_file, mhci_pept_lens=mhci_pept_lens, mhcii_pept_lens=mhcii_pept_lens)
