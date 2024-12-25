#!/bin/python3
# Script Name: annotate_abundance.py
# Description: Append abundance-related metrics, including DNA_AF, RNA_AF, RNA_EXP
# Author: Kohan

import sys, os, argparse
import pandas as pd
from api import *


def ArgumentParser(args=None):
    parser = argparse.ArgumentParser(prog='Append abundance-related metrics',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    # required arguments
    parser.add_argument('mut_file', type=str, help='Path to the mutation CSV file, output from generate_peptides.py')
    parser.add_argument('out_file', type=str, help='Output CSV file')

    # optional arguments
    parser.add_argument('--tumor_colname', type=str, default=None, help='Column name of tumor sample (derived from VCF); highly recommend to specify it')
    parser.add_argument('--rna_rsem_file', type=str, default=None, help='Path to the RSEM expression file (.genes.result)')
    parser.add_argument('--rna_af_file', type=str, default=None, help='Path to the RNA AF file (.readcount.parsed.tsv) derived from bam-readcount')

    return parser


def AddDNAAF(mut_df, colname):
    mut_df['DNA_AF'] = mut_df[colname].apply(lambda x: float(x.split(':')[2].split(',')[0]))
    return mut_df


def AddRNAEXP(mut_df, exp_file):
    exp_df = pd.read_csv(exp_file, sep='\t')

    # add quartile
    exp_df['QRT'] = 1
    for i, q in enumerate([0.25, 0.5, 0.75]):
        thrs = exp_df[exp_df['TPM']>0]['TPM'].quantile(q)
        exp_df.loc[exp_df['TPM']>thrs, 'QRT'] = i+2

    # transcript to exp and qrt dict
    exp_dict, qrt_dict = dict(), dict()
    for idx, row in exp_df.iterrows():
        gene = row['gene_id'].split('.')[0]
        exp_dict[gene] = row['TPM']
        qrt_dict[gene] = row['QRT']
    
    # annotate
    mut_df['RNA_EXP'] = mut_df['Gene'].apply(lambda x: exp_dict.get(x, 0))
    mut_df['RNA_EXP_QRT'] = mut_df['Gene'].apply(lambda x: qrt_dict.get(x, 1))

    return mut_df


# merge single substitutions based on chr, pos, ref, alt
# merge indels based on chr, pos (if duplicated, choose the max VAF one)
def AddRNAAF(mut_df, rna_af_file):
    rna_df = pd.read_csv(rna_af_file, sep='\t')
    rna_df['ref_length'] = rna_df['ref'].apply(lambda x: len(x)) # ref length
    rna_df['base_length'] = rna_df['base'].apply(lambda x: len(x)) # alt length
    snv_rna_df = rna_df[(rna_df['ref_length']==1) & (rna_df['base_length']==1)] # SNV
    indel_rna_df = rna_df[~((rna_df['ref_length']==1) & (rna_df['base_length']==1))] # INDEL

    # merge INDELs
    indel_rna_df = indel_rna_df.sort_values(by=['chrom', 'position', 'vaf']).drop_duplicates(subset=['chrom', 'position'])
    mut_df = mut_df.set_index(['#CHROM', 'POS'])
    indel_rna_df = indel_rna_df.set_index(['chrom', 'position'])
    mut_df['RNA_DEPTH_INDEL'] = indel_rna_df['depth'] # add read depth for indels
    mut_df['RNA_AF_INDEL'] = indel_rna_df['vaf'] # add VAF for indels
    mut_df = mut_df.reset_index()

    # merge SNVs
    snv_rna_df = snv_rna_df.sort_values(by=['chrom', 'position', 'ref', 'base', 'vaf']).drop_duplicates(subset=['chrom', 'position', 'ref', 'base'])
    mut_df = mut_df.set_index(['#CHROM', 'POS', 'REF', 'ALT'])
    snv_rna_df = snv_rna_df.set_index(['chrom', 'position', 'ref', 'base'])
    mut_df['RNA_DEPTH_SNV'] = snv_rna_df['depth'] # add read depth for SNVs
    mut_df['RNA_AF_SNV'] = snv_rna_df['vaf'] # add VAF for SNVs
    mut_df = mut_df.reset_index()

    # Combine SNV and INDEL columns into general RNA_DEPTH and RNA_AF
    mut_df['RNA_DEPTH'] = mut_df['RNA_DEPTH_SNV'].combine_first(mut_df['RNA_DEPTH_INDEL'])
    mut_df['RNA_AF'] = mut_df['RNA_AF_SNV'].combine_first(mut_df['RNA_AF_INDEL'])

    # Drop intermediate columns
    mut_df.drop(columns=['RNA_DEPTH_SNV', 'RNA_DEPTH_INDEL', 'RNA_AF_SNV', 'RNA_AF_INDEL'], inplace=True)

    return mut_df


if __name__=='__main__':
    args = ArgumentParser().parse_args()
    mut_df = pd.read_csv(args.mut_file, index_col=0)

    # DNA AF
    if args.tumor_colname:
        tumor_colname = args.tumor_colname
    else:
        tumor_colname = [col for col in mut_df.columns if 'tumor' in col.lower()][0]
    print('Tumor sample name =', tumor_colname)
    print('Adding DNA AF ...')
    mut_df = AddDNAAF(mut_df, tumor_colname)

    # RNA EXP
    if os.path.isfile(args.rna_rsem_file):
        print('RNA expression file exists. Adding RNA expression level ...')
        mut_df = AddRNAEXP(mut_df, args.rna_rsem_file)
    else:
        print('RNA expression file is not available. Skip RNA expression annotation')
        mut_df['RNA_EXP'] = np.nan

    # RNA AF
    if os.path.isfile(args.rna_af_file):
        print('RNA bam file exists. Adding RNA allelic fraction ...')
        mut_df = AddRNAAF(mut_df, args.rna_af_file)
    else:
        print('RNA bam file is not available. Skip RNA AF annotation')
        mut_df['RNA_DEPTH'] = np.nan
        mut_df['RNA_AF'] = np.nan
    
    # output
    mut_df.to_csv(args.out_file)
    print('Finished abundance annotation!')