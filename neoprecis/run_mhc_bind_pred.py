#!/bin/python3
# Script Name: run_mhc_bind_pred.py
# Description: Run MHC binding prediction
# Author: Kohan

import sys
import os
import argparse
import pandas as pd

# Support both package import and direct script execution
try:
    from .api import MHC, LoadAllowedAlleles, MHCIAlleleTransform, MHCIIAlleleTransform
    from .api import RunNetMHCpan, RunMixMHCpred, ReadNetMHCpan, ReadMixMHCpred
except ImportError:
    # Fallback for direct execution (python neoprecis/run_mhc_bind_pred.py)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from api import MHC, LoadAllowedAlleles, MHCIAlleleTransform, MHCIIAlleleTransform
    from api import RunNetMHCpan, RunMixMHCpred, ReadNetMHCpan, ReadMixMHCpred


def ArgumentParser(args=None):
    parser = argparse.ArgumentParser(prog='Run MHC binding prediction')
    # required
    parser.add_argument('mhc_file', type=str, help='MHC allele TXT file')
    parser.add_argument('peptide_file', type=str, help='Peptide file')
    parser.add_argument('out_prefix', type=str, help='Output prefix')
    # optional
    parser.add_argument('--mhc_class', type=str, default='I', help='MHC class (I or II)')
    parser.add_argument('--predictor', type=str, default='netMHCpan', help='netMHCpan or mixMHCpred')
    parser.add_argument('--exec_file', type=str, default='netMHCpan-4.1/netMHCpan', help='executable file')
    return parser


# identify unpredicted peptides
# input: peptides, alleles, prev_df (previous predicted df)
# output: unpredicted peptides (list)
def IdentifyNewPeptides(peptides: list, alleles: list, prev_df: pd.DataFrame):
    exist_pairs = set(prev_df.groupby(['Peptide', 'MHC']).size().index)
    require_pairs = set([(peptide, allele) for peptide in peptides for allele in alleles])
    last_pairs = require_pairs - exist_pairs
    last_peptides = list(set([i[0] for i in list(last_pairs)]))
    return last_peptides
        

# save peptides to a txt file where each row is a peptide
def SavePeptides(peptides: list, file: str):
    with open(file, 'w') as f:
        for peptide in peptides:
            f.write(f'{peptide}\n')


# rename previous predicted files if exist
# prefix.csv -> prefix.n.csv (if prefix.n-1.csv exists; n starts from 1)
def RenamePredFiles(file: str):
    prefix = '.'.join(file.split('.')[:-1])
    n = 1
    while os.path.isfile(f'{prefix}.{n}.csv'):
        n += 1
    os.rename(file, f'{prefix}.{n}.csv')


# run MHC binding prediction
# input: mhc_class ('i' or 'ii'), predictor ('netmhcpan' or 'mixmhcpred'),
#        exec_path (binding predictor exec file), alleles, peptide_file, out_file
# allele name has to match the predictor's valid name
def RunMHCPred(mhc_class: str, predictor: str, exec_path: str, alleles: list, peptide_file: str, out_file: str):
    # netmhcpan
    if predictor == 'netmhcpan':
        RunNetMHCpan(','.join(alleles), peptide_file, out_file, exec_path)
    
    # mixmhcpred
    else:
        context_cmd = True if mhc_class == 'ii' else False
        sep = ' ' if mhc_class == 'ii' else ','
        RunMixMHCpred(
            sep.join(alleles),
            peptide_file,
            out_file,
            context_cmd=context_cmd,
            exe_path=exec_path
        )


# read MHC prediction file
# input: mhc_class ('i' or 'ii'), predictor ('netmhcpan' or 'mixmhcpred'),
#        mhc (allele object), pred_file (binding prediction file)
# output: pandas DF
def ReadMHCPred(mhc_class: str, mhc: MHC, predictor: str, pred_file: str):
    # alelle rename dict
    mhc_name_transform_dict = mhc.get_transform_dict(mhc.name_dict[f'MHC-{mhc_class.upper()}'], From=predictor, To='standard')
    # read prediction file
    if predictor == 'netmhcpan': # netmhcpan
        df = ReadNetMHCpan(pred_file)
    else: # mixmhcpred
        df = ReadMixMHCpred(pred_file)
    if (predictor == 'netmhcpan') and (mhc_class == 'i'):
        df['MHC'] = df['MHC'].apply(lambda x: x[4:])
    else:
        df['MHC'] = df['MHC'].apply(lambda x: mhc_name_transform_dict[x])
    return df


def Main(args):
    mhc_class = args.mhc_class.lower()
    predictor = args.predictor.lower()

    ### MHC alleles
    exec_dir = os.path.dirname(args.exec_file)
    # allowed alleles
    allowed_alleles = LoadAllowedAlleles(mhc_class, predictor, exec_dir)
    # target alleles
    mhc = MHC(args.mhc_file)
    alleles = mhc.name_dict[f'MHC-{mhc_class.upper()}'][predictor] # predictor's allele names
    print('Target alleles:', alleles)
    # check if targets in allowed list
    alleles = list(set(alleles)) # unique
    alleles = [allele for allele in alleles if allele in allowed_alleles]
    if mhc_class == 'i':
        std_alleles = [MHCIAlleleTransform(allele, From=predictor, To='standard') for allele in alleles]
    else:
        std_alleles = [MHCIIAlleleTransform(allele, From=predictor, To='standard') for allele in alleles]
    print('Allowed alleles:', alleles)
    # exit if no alleles
    if len(alleles) == 0:
        print('Exit due to no alleles')
        sys.exit(1)

    ### Peptides
    with open(args.peptide_file, 'r') as f:
        peptides = [line.strip() for line in f.readlines()]
    if len(peptides) == 0:
        print('Exit due to no peptides')
        sys.exit(1)
    print('#Peptides =', len(peptides))

    ### Prediction
    pred_file = f'{args.out_prefix}.netmhcpan.mhc{mhc_class}'   # prediction file
    out_file = f'{args.out_prefix}.pred.mhc{mhc_class}.csv'     # processed prediction file
    tmp_peptide_file = f'{args.out_prefix}.peptide.left.txt'    # temp peptide file
    # check if output exists and identify unpredicted peptides
    if os.path.isfile(out_file):
        prev_df = pd.read_csv(out_file)                                     # previous prediction file
        left_peptides = IdentifyNewPeptides(peptides, std_alleles, prev_df) # identify unpredicted peptides
        print(f'Previous predictions found; {len(left_peptides)} peptides left for prediction')
    else:
        prev_df = pd.DataFrame()
        left_peptides = peptides
    # save unpredicted peptides
    if len(left_peptides) == 0:
        print('Exit due to no unpredicted peptides')
        sys.exit(1)
    else:
        SavePeptides(left_peptides, tmp_peptide_file) # save left peptides to temp peptide file
    # predict
    print('Running MHC-peptide binding prediction ...')
    RunMHCPred(mhc_class, predictor, args.exec_file, alleles, tmp_peptide_file, pred_file) # run binding prediction

    ### Parse output
    df = ReadMHCPred(mhc_class, mhc, predictor, pred_file)
    df = pd.concat([prev_df, df], axis=0, ignore_index=True)
    df = df[df['Peptide'].isin(peptides)].reset_index(drop=True)
    df = df[df['MHC'].isin(std_alleles)].reset_index(drop=True)
    df.to_csv(out_file, index=False)
    print('#Total prediction pairs =', df.shape[0])

    ### remove temp files
    os.remove(tmp_peptide_file) # remove temp peptide file
    os.remove(pred_file) # remove binding prediction file
    print('Finished!!!')


if __name__=='__main__':
    args = ArgumentParser().parse_args()
    Main(args)