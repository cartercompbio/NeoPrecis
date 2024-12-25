#!/bin/python3
# Script Name: update_reference.py
# Description: update motif reference
# Author: Kohan

import os, sys, shutil, h5py, argparse
import logomaker as lm
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from api import *

blosum = Blosum62()
sub_matrix = blosum._ordered(blosum.matrix)
sub_matrix = blosum._normalized(sub_matrix)


def ArgumentParser(args=None):
    parser = argparse.ArgumentParser(prog='Update motif reference')
    # required
    parser.add_argument('ref_file', type=str, help='Reference h5 file')
    parser.add_argument('mhc_file', type=str, help='MHC allele list file')
    # optional
    parser.add_argument('--mhci_pred_exec', type=str, default='netMHCpan-4.1/netMHCpan', help='Path of MHC-I binding predictor')
    parser.add_argument('--mhcii_pred_exec', type=str, default='netMHCIIpan-4.3/netMHCIIpan', help='Path of MHC-II binding predictor')
    parser.add_argument('--mhci_peptide_file', type=str, default='mhci_random_peptides.txt', help='Random human peptides for MHC-I prediction')
    parser.add_argument('--mhcii_peptide_file', type=str, default='mhcii_random_peptides.txt', help='Random human peptides for MHC-II prediction')
    parser.add_argument('--mhci_bind_thrs', type=float, default=2, help='MHC-I binding rank threshold')
    parser.add_argument('--mhcii_bind_thrs', type=float, default=10, help='MHC-II binding rank threshold')
    return parser


# motif matrix = position by amino acid
# sub matrix = amino acid by amino acid
# position by amino acid matrix
def WeightedEntropy(motif_matrix, sub_matrix):
    entropy = motif_matrix @ np.log2(sub_matrix @ motif_matrix.T)
    entropy = -entropy.diagonal()
    return entropy


# get motifs and position factors
def GetMotifs(alleles, pred_df, bind_thrs):
    motif_list, pos_fac_list = list(), list()
    for allele in alleles:
        peptides = pred_df.loc[(pred_df['MHC']==allele) & (pred_df['Rank']<=bind_thrs), 'Core'] # binding peptides
        motif_df = lm.alignment_to_matrix(peptides, to_type='probability', characters_to_ignore='X') # motif
        motif_matrix = motif_df.to_numpy() # convert to matrix
        entropy = WeightedEntropy(motif_matrix, sub_matrix) # position factor
        motif_matrix = np.hstack((motif_matrix, np.zeros((9, 1))))
        motif_list.append(motif_matrix)
        pos_fac_list.append(entropy)
    return np.array(motif_list), np.array(pos_fac_list)


def Main(ref_file, std_alleles, pred_alleles, peptide_file, pred_exec, bind_thrs):
    if len(std_alleles) == 0:
        return

    # tmp files
    tmp_pred_file = './mhc_preds.txt'
    tmp_h5_file = './tmp_ref.h5'

    # prediction
    RunNetMHCpan(','.join(pred_alleles), peptide_file, tmp_pred_file, exe_path=pred_exec)
    pred_df = ReadNetMHCpan(tmp_pred_file)
    os.remove(tmp_pred_file)

    # motifs and position factors
    new_motifs, new_pos_facs = GetMotifs(pred_alleles, pred_df, bind_thrs)

    # load previous h5
    with h5py.File(ref_file, 'r') as ref:
        aa_list = ref['aa_list'].asstr()[:].tolist()
        aa_blosum_encodes = ref['aa_blosum_encodes'][:]
        aa_blosum_pc2_encodes = ref['aa_blosum_pc2_encodes'][:]
        allele_list = ref['allele_list'].asstr()[:].tolist()
        position_factors = ref['position_factors'][:]
        motifs = ref['motifs'][:]
    
    # add new motifs
    allele_list = allele_list + std_alleles
    motifs = np.vstack([motifs, new_motifs])
    position_factors = np.vstack([position_factors, new_pos_facs])

    # save new h5
    with h5py.File(tmp_h5_file, 'w') as f:
        f.create_dataset('aa_list', data=aa_list)
        f.create_dataset('aa_blosum_encodes', data=aa_blosum_encodes)
        f.create_dataset('aa_blosum_pc2_encodes', data=aa_blosum_pc2_encodes)
        f.create_dataset('allele_list', data=allele_list)
        f.create_dataset('position_factors', data=position_factors)
        f.create_dataset('motifs', data=motifs)
    
    # replace previous h5
    os.remove(ref_file)
    shutil.move(tmp_h5_file, ref_file)


if __name__=='__main__':
    args = ArgumentParser().parse_args()

    ### reference
    with h5py.File(args.ref_file, 'r') as ref:
        allele_list = ref['allele_list'].asstr()[:].tolist()

    ### new alleles
    mhc = MHC(args.mhc_file)
    # allele name
    mhci_std_alleles = mhc.name_dict['MHC-I']['standard']
    mhcii_std_alleles = mhc.name_dict['MHC-II']['standard']
    mhci_pred_alleles = mhc.name_dict['MHC-I']['netmhcpan']
    mhcii_pred_alleles = mhc.name_dict['MHC-II']['netmhcpan']
    # new allele idx
    mhci_idx = [i for i,s in enumerate(mhci_std_alleles) if s not in allele_list]
    mhcii_idx = [i for i,s in enumerate(mhcii_std_alleles) if s not in allele_list]
    # new allele name
    mhci_std_alleles = [mhci_std_alleles[i] for i in mhci_idx]
    mhcii_std_alleles = [mhcii_std_alleles[i] for i in mhcii_idx]
    mhci_pred_alleles = [mhci_pred_alleles[i] for i in mhci_idx]
    mhcii_pred_alleles = [mhcii_pred_alleles[i] for i in mhcii_idx]
    print('New MHC-I alleles:', mhci_std_alleles)
    print('New MHC-II alleles:', mhcii_std_alleles)

    ### main
    Main(args.ref_file, mhci_std_alleles, mhci_pred_alleles, args.mhci_peptide_file, args.mhci_pred_exec, args.mhci_bind_thrs)
    Main(args.ref_file, mhcii_std_alleles, mhcii_pred_alleles, args.mhcii_peptide_file, args.mhcii_pred_exec, args.mhcii_bind_thrs)