#!/bin/python3
# Script Name: api.py
# Description: Utility functions for the package
# Author: Kohan

import os, sys, re, subprocess, json, difflib
import numpy as np
import pandas as pd
from collections import defaultdict, OrderedDict
from scipy.stats import norm
from Bio import SeqIO
from Bio.Seq import translate
import warnings
warnings.filterwarnings('ignore')
src_dir = os.path.dirname(os.path.abspath(__file__))

# import peptCRD module
sys.path.append(f'{src_dir}/CRD')
from CRD import SubCRD, PeptCRD

# check and import foreignness module
if os.path.isdir(f'{src_dir}/NeoantigenEditing'):
    foreignness_aval = True
    sys.path.append(f'{src_dir}/NeoantigenEditing')
    from foreignness import Foreignness
else:
    foreignness_aval = False


###############################
###           IO            ###
###############################

# read VCF file
def ReadVCF(file):
    with open(file, 'r') as f:
        lines = f.readlines()
    i = 0
    
    ### columns
    while True:
        # info
        if lines[i].startswith('##INFO'):
            start_idx = lines[i].find('Format: ')
            end_idx = lines[i].rfind('">')
            annot_cols = lines[i][start_idx+len('Format: '): end_idx].split('|')
        # cols
        if lines[i].startswith('#CHROM'):
            cols = lines[i].strip().split()
            break
        i += 1
    info_idx = cols.index('INFO')

    ### rows
    rows = list()
    i += 1
    while i < len(lines):
        row = lines[i].strip().split()
        info = row[info_idx].split(';')[-1][4:]
        info_list = [s.split('|') for s in info.split(',')]
        # multiple transcripts
        for info in info_list:
            new_row = row[:info_idx] + row[info_idx+1:] + info
            rows.append(new_row)
        i += 1

    ### pandas dataframe
    cols = cols[:info_idx] + cols[info_idx+1:] + annot_cols
    df = pd.DataFrame(rows, columns=cols)

    return df


# read the VEP annotated TXT file
def ReadVEPannotTXT(file):
    with open(file, 'r') as f:
        lines = f.readlines()
        count = 0
        for line in lines:
            if line.startswith('##'):
                count += 1
            else:
                break
    df = pd.read_csv(file, sep='\t', skiprows=count)
    return df


# load transcript-related reference sequences (FASTA)
def ReadTranscriptFASTA(file):
    seq_dict = dict()
    with open(file, 'r') as f:
        for record in SeqIO.parse(f, 'fasta'):
            enst = record.id.split('.')[0]
            if seq_dict.get(enst):
                print('Duplicate ENST:', enst)
                # keep the longer one
                if len(str(record.seq)) <= seq_dict[enst]:
                    continue
            if str(record.seq) == 'Sequenceunavailable':
                continue
            seq_dict[enst] = str(record.seq)
    return seq_dict


# write the output peptide file
def WritePeptideTXT(peptides, out_file):
    with open(out_file, 'w') as f:
        for peptide in peptides:
            f.write(f'{peptide}\n')


# read MHC allele TXT file
# A*01:01, DRB1*01:01, ...
class MHC():
    def __init__(self, file):
        # load file
        with open(file, 'r') as f:
            lines = f.readlines()

        # normalize alleles
        self.allele_dict = {'MHC-I': defaultdict(list), 'MHC-II': defaultdict(list)}
        for line in lines:
            # get allele
            line = line.strip()
            if line.startswith('HLA-'):
                line = line[4:]
            if not (re.match(r'(A|B|C)\*[0-9]+\:[0-9]+', line) or re.match(r'(DPA|DPB|DQA|DQB|DRB)[0-9]\*[0-9]+\:[0-9]+', line)):
                print(f"Allele {line} isn't valid")
                continue
            gene, group, prot = self._nomenclature(line)
            # save to dict
            if gene in ['A', 'B', 'C']:
                self.allele_dict['MHC-I'][gene].append((gene, group, prot))
            elif re.match(r'(DPA|DPB|DQA|DQB|DRB)[0-9]', gene):
                self.allele_dict['MHC-II'][gene[:-1]].append((gene, group, prot))
            else:
                print(f"Gene {gene} isn't valid")
                continue

        # get standard and customized allele names
        self.name_dict = dict()
        self.name_dict['MHC-I'] = self._get_mhci_names(self.allele_dict['MHC-I'])
        self.name_dict['MHC-II'] = self._get_mhcii_names(self.allele_dict['MHC-II'])

    def _nomenclature(self, allele):
        gene, code = allele.split('*')
        group, prot = code.split(':')
        return gene, group, prot
    
    def _get_mhci_names(self, allele_dict, genes=['A','B','C']):
        standard_list, netmhcpan_list, mixmhcpred_list = list(), list(), list()
        for gene in genes:
            for gene, group, prot in allele_dict[gene]:
                standard_list.append(f'{gene}*{group}:{prot}')
                netmhcpan_list.append(f'HLA-{gene}{group}:{prot}')
                mixmhcpred_list.append(f'{gene}{group}{prot}')
        d = {
            'standard': standard_list,
            'netmhcpan': netmhcpan_list,
            'mixmhcpred': mixmhcpred_list
        }
        return d
    
    def _get_mhcii_names(self, allele_dict, genes=['DRB', 'DPA', 'DPB', 'DQA', 'DQB']):
        standard_list, netmhcpan_list, mixmhcpred_list = list(), list(), list()
        # DR
        if 'DRB' in genes:
            for gene, group, prot in allele_dict['DRB']:
                standard_list.append(f'{gene}*{group}:{prot}')
                netmhcpan_list.append(f'{gene}_{group}{prot}')
                mixmhcpred_list.append(f'{gene}_{group}_{prot}')
        # DP, DQ
        for gene in ['DP', 'DQ']:
            if (f'{gene}A' not in genes) or (f'{gene}B' not in genes): continue
            for gene_a, group_a, prot_a in allele_dict[f'{gene}A']:
                for gene_b, group_b, prot_b in allele_dict[f'{gene}B']:
                    standard_list.append(f'{gene_a}*{group_a}:{prot_a}_{gene_b}*{group_b}:{prot_b}')
                    netmhcpan_list.append(f'HLA-{gene_a}{group_a}{prot_a}-{gene_b}{group_b}{prot_b}')
                    mixmhcpred_list.append(f'{gene_a}_{group_a}_{prot_a}__{gene_b}_{group_b}_{prot_b}')
        d = {
            'standard': standard_list,
            'netmhcpan': netmhcpan_list,
            'mixmhcpred': mixmhcpred_list
        }
        return d
    
    def get_transform_dict(self, allele_dict, From='netmhcpan', To='standard'):
        return dict(zip(allele_dict[From], allele_dict[To]))


###############################
###   Peptide Generation    ###
###############################

# generate long WT and MT peptides
class PepGen():
    def __init__(self, cdna_fasta, cds_fasta):
        self.cdna_dict = ReadTranscriptFASTA(cdna_fasta)
        self.cds_dict = ReadTranscriptFASTA(cds_fasta)


    # output the shortest peptide that contains all possible mutated peptides with the specific length and aligned wild-type peptides
    def __call__(self, transcript, cds_position, codon, consequence, length=15):
        transcript = transcript.split('.')[0]
        print(f"Extract WT and MT sequence of the mutation {codon} at position {cds_position} of transcript {transcript}")
        
        ### prepare reference sequence
        # check transcript
        if (self.cdna_dict.get(transcript) == -1) or (self.cds_dict.get(transcript) == -1):
            print(f"\tError: Transcript {transcript} is not in the cDNA or CDS reference")
            return ('', '')
        # extend sequence after the stop codon
        cdna_seq, cds_seq = self.cdna_dict[transcript], self.cds_dict[transcript]
        pos = cdna_seq.find(cds_seq)
        if pos == -1:
            print(f"\tWarning: sequence mismatch between cDNA and CDS of the transcript {transcript}")
        else:
            cds_seq = cdna_seq[pos:]
        cds_seq_dict = OrderedDict({i+1: [s, s] for i, s in enumerate(cds_seq)}) # 1-based ({pos:[WT, MT],})
        
        ### get mutation details
        mut_info = self._get_mutation_details(cds_seq, cds_position, codon)
        if mut_info:
            start, end, wt_nt, mt_nt = mut_info
        else:
            print("\tFail mutation quality check")
            return ('', '')
        
        ### mutate nucleotides
        mt_cds_seq = self._mutate_cds(cds_seq_dict, start, end, wt_nt, mt_nt)

        ### mutate proteins
        wt_prot = self._translate(cds_seq)
        mt_prot = self._translate(mt_cds_seq)
        wt_pept, mt_pept = self._mutate_pept(wt_prot, mt_prot, length)

        ### check consequence
        if ('frameshift_variant' in consequence) or ('stop_lost' in consequence):
            wt_pept = wt_pept[:len(mt_pept)]

        return wt_pept, mt_pept
    

    # get start, end, wt_nucleotide, and mt_nucleotide of the mutation
    # check the mutation quality
    def _get_mutation_details(self, cds_seq, cds_position, codon):
        # get start and end of the mutation (1-base)
        try:
            cds_position = cds_position.split('/')[0] # if length provided (position/length)
            start, end = int(str(cds_position).split('-')[0]), int(str(cds_position).split('-')[-1])
        except Exception as e:
            print(f"\t{str(e)}")
            print("\tError: Mutated position is not available")
            return False
        
        # check start, end in the range of CDS length
        if end > len(cds_seq):
            print("\tError: Mutated position beyond the CDS length limit")
            return False
        
        # get mutation (codon)
        if type(codon) != str:
            print("\tError: Codon is not available")
            return False 
        
        # check match between CDS and codon annotation
        cds_codon = cds_seq[int((np.ceil(start/3)-1)*3): int(np.ceil(end/3)*3)]
        annot_codon = codon.split('/')[0]
        if (annot_codon != '-') and (cds_codon.lower() != annot_codon.lower()):
            print("\tError: Sequence mismatch between CDS and annotated codon")
            return False
        
        # check nucleotide changes
        wt_nt, mt_nt = [re.findall(r'[ATCG]+', s) for s in codon.split('/')] # return list
        if (len(wt_nt) == 0) and (len(mt_nt) == 0): # no changes
            print("\tSkip: No nucleotide change detected")
            return False
        elif (len(wt_nt) > 1) or (len(mt_nt) > 1): # not a consecutive change
            print("\tSkip: inconsecutive nucleotide change")
            return False
        wt_nt = '' if len(wt_nt)==0 else wt_nt[0] # get the string
        mt_nt = '' if len(mt_nt)==0 else mt_nt[0] # get the string

        # generate WT sequence and check protein sequence
        cds_seq = cds_seq[:len(cds_seq)//3*3]
        prot_seq = self._translate(cds_seq)
        prot_start = int(np.ceil(start/3))
        if len(prot_seq)+1 < prot_start: # +1 for the stop codon
            print("Error: Mutated position beyond the protein length limit")
            return False

        return (start, end, wt_nt, mt_nt)

    
    # mutate CDS
    def _mutate_cds(self, cds_seq_dict, start, end, wt_nt, mt_nt):
        if len(wt_nt) == len(mt_nt): # substitution
            for i, pos in enumerate(range(start, end+1)):
                cds_seq_dict[pos][1] = mt_nt[i]
        elif len(wt_nt) == 0: # insertion
            cds_seq_dict[start][1] += mt_nt
        elif len(mt_nt) == 0: # deletion
            for i, pos in enumerate(range(start, end+1)):
                cds_seq_dict[pos][1] = ''
        else: # indel
            cds_seq_dict[start][1] = mt_nt
            for i, pos in enumerate(range(start+1, end+1)):
                cds_seq_dict[pos][1] = ''
        # generate mutated CDS sequence
        mt_cds_seq = ''.join([cds_seq_dict[i+1][1] for i in range(len(cds_seq_dict))])
        return mt_cds_seq


    # output WT peptide and MT peptide
    def _mutate_pept(self, wt_prot, mt_prot, length):
        # check if identical
        if mt_prot in wt_prot:
            print("\tSkip: protein seqeuences are the same in WT and MT")
            return ('', '')
        # forward
        idx_f = 0
        while (wt_prot[idx_f:idx_f+length] == mt_prot[idx_f:idx_f+length]):
            idx_f += 1
        # backward
        idx_b = -1
        while (wt_prot[idx_b-length+1:idx_b]+wt_prot[idx_b] == mt_prot[idx_b-length+1:idx_b]+mt_prot[idx_b]):
            idx_b -= 1
        best_idx_f = min(idx_f, idx_b-length+1+len(wt_prot), idx_b-length+1+len(mt_prot))
        best_idx_b = max(idx_b, idx_f+length-1-len(wt_prot), idx_f+length-1-len(mt_prot))
        # pept
        wt_pept = wt_prot[best_idx_f: len(wt_prot)+best_idx_b+1]
        mt_pept = mt_prot[best_idx_f: len(mt_prot)+best_idx_b+1]
        # cut redundant wt pept
        if (len(wt_pept) > len(mt_pept)) & (wt_pept[-1] != mt_pept[-1]):
            wt_pept = wt_pept[:len(mt_pept)]
        elif (len(wt_pept) > len(mt_pept)) & (wt_pept[0] != mt_pept[0]):
            wt_pept = wt_pept[-len(mt_pept):]
        return wt_pept, mt_pept


    # translate into protein
    def _translate(self, nt_seq):
        prot = translate(nt_seq)
        prot = prot.split('*')[0]
        return prot


# generate short MT and aligned WT epitopes
class EpiGen():
    def __init__(self, wt_seq, mt_seq, consequence):
        self.wt_seq = wt_seq
        self.mt_seq = mt_seq
        self.wt_len = len(wt_seq)
        self.mt_len = len(mt_seq)
        self.consequence = consequence


    def __call__(self, length):
        print(f'Generate epitopes with length of {length} from WT "{self.wt_seq}" and MT "{self.mt_seq}"')
        # check length
        if self.mt_len < length:
            print(f"\tSkip: mutated sequence is shorter than {length}")
        # for each mutated epitope
        epi_df = list()
        for i in range(self.mt_len-length+1):
            mt_epi = self.mt_seq[i:i+length]
            start, end = i, i+length-1-self.mt_len
            wt_end = self.wt_len+end
            
            # check right and left ends match
            # left end
            left_end_match = False
            if start < self.wt_len:
                if self.wt_seq[start]==self.mt_seq[start]:
                    left_end_match = True
            # right end
            right_end_match = False
            if ('frameshift_variant' in self.consequence) or ('stop_lost' in self.consequence):
                right_end_match = False
            elif (wt_end >= 0) & (wt_end < self.wt_len):
                if self.wt_seq[wt_end]==self.mt_seq[end]:
                    right_end_match = True
            
            # extract aligned WT seqs
            # both end
            wt_epi_both = self.wt_seq[start: wt_end+1] if (left_end_match & right_end_match) else ''
            # left end
            wt_epi_left = self.wt_seq[start: start+length] if left_end_match else ''
            # right end
            wt_epi_right = self.wt_seq[max(0, wt_end+1-length): wt_end+1] if right_end_match else ''

            # append
            if mt_epi not in self.wt_seq:
                epi_df.append({
                    'MT_start': start,
                    'MT_end': end+self.mt_len,
                    'MT_epitope': mt_epi,
                    'MT_length': length,
                    'WT_epitope_left_aligned': wt_epi_left,
                    'WT_epitope_right_aligned': wt_epi_right,
                    'WT_epitope_both_aligned': wt_epi_both
                })

        epi_df = pd.DataFrame(epi_df)
        return epi_df
    

    def get_peptides(self, epi_df, lengths,
                     cols=['MT_epitope', 'WT_epitope_left_aligned', 'WT_epitope_right_aligned', 'WT_epitope_both_aligned']):
        pepts = set()
        for col in cols:
            pepts = pepts | set(epi_df[col].tolist())
        pepts = list(pepts)
        pepts = [pept for pept in pepts if len(pept) in lengths]
        return pepts


###############################
###    Binding Predictor    ###
###############################

### read NetMHCpan output file
# output columns: MHC, Peptide, Core, CorePos, Rank, Score
def ReadNetMHCpan(file):
    with open(file, 'r') as f:
        lines = f.readlines()
    num_lines = len(lines)
    df = pd.DataFrame()
    idx = 0

    while idx < num_lines:
        # locate the line of column name
        while (idx < num_lines-1) and ((not lines[idx].startswith('---')) | (not lines[idx+1].startswith(' Pos'))):
            idx += 1
        if idx == num_lines-1: break
        col_names = lines[idx+1].strip().split()
        num_cols = len(col_names)
        
        # read rows
        idx += 3
        rows = list()
        while (idx < num_lines) and (not lines[idx].startswith('---')):
            row = lines[idx].strip().split()
            if len(row) < num_cols:
                row += ['',] * (num_cols - len(row))
            elif len(row) > num_cols:
                row = row[:-2] + [''.join(row[-2:]),]
            rows.append(row)
            idx += 1

        # dataframe
        tmp_df = pd.DataFrame(rows, columns=col_names).set_index('Pos')

        # dtype
        tmp_df['%Rank_EL'] = tmp_df['%Rank_EL'].astype(float)
        tmp_df['Score_EL'] = tmp_df['Score_EL'].astype(float)

        # rename columns
        tmp_df = tmp_df.rename(columns={'Of': 'CorePos', '%Rank_EL': 'Rank', 'Score_EL': 'Score'})
        tmp_df = tmp_df[['MHC', 'Peptide', 'Core', 'CorePos', 'Rank', 'Score']]
        df = pd.concat([df, tmp_df], axis=0)

        idx += 1

    return df


### read MixMHCpred output file
# output columns: MHC, Peptide, Core, CorePos, Rank
def ReadMixMHCpred(file):
    df = pd.read_csv(file, comment='#', sep='\t')
    if 'Core_best' in df.columns: # MHC-II
        df = df.rename(columns={'BestAllele':'MHC', 'Core_best':'Core', 'CoreP1_best':'CorePos', '%Rank_best':'Rank'})
    else: # MHC-I
        df = df.rename(columns={'BestAllele':'MHC', '%Rank_bestAllele':'Rank'})
        df['Core'] = df['Peptide']
        df['CorePos'] = 0
    df = df[['MHC', 'Peptide', 'Core', 'CorePos', 'Rank']]
    df = df.dropna()
    df['CorePos'] = df['CorePos'].astype(int) - 1 # 1-base to 0-base
    return df


### run NetMHCpan
def RunNetMHCpan(alleles, peptide_file, output_file,
                 exe_path='/carter/users/kol004/tools/netMHCIIpan-4.3/netMHCIIpan'):
    cmd = [
        exe_path,
        '-a', alleles,
        '-inptype', '1',
        '-f', peptide_file,
    ]
    with open(output_file, 'w') as f:
        subprocess.run(cmd, stdout=f, text=True)


### run MixMHCpred
def RunMixMHCpred(alleles, peptide_file, output_file,
                  context_cmd=True,
                  exe_path='/carter/users/kol004/tools/MixMHC2pred/MixMHC2pred_unix'):
    cmd = [
        exe_path,
        '-a', alleles,
        '-i', peptide_file,
        '-o', output_file
    ]
    if context_cmd: cmd.append('--no_context')
    subprocess.run(' '.join(cmd), shell=True)


### load available alleles
# mhc: I or II
# predictor: mixmhcpred or netmhcpan
def LoadAllowedAlleles(mhc, predictor, dirname):
    if mhc.lower()=='i' and predictor.lower()=='netmhcpan':
        file = f'{dirname}/data/allelenames'
        alleles = np.loadtxt(file, dtype=str)[:,0].tolist()
    elif mhc.lower()=='ii' and predictor.lower()=='netmhcpan':
        file = f'{dirname}/data/allelelist.txt'
        alleles = np.loadtxt(file, dtype=str)[:,0].tolist()
    elif mhc.lower()=='i' and predictor.lower()=='mixmhcpred':
        file = f'{dirname}/lib/alleles_list.txt'
        df = pd.read_csv(file, sep='\t')
        alleles = df['Allele'].unique().tolist()
    elif mhc.lower()=='ii' and predictor.lower()=='mixmhcpred':
        alleles = os.listdir(f'{dirname}/PWMdef')
        alleles = [allele.split('.')[0] for allele in alleles]
    else:
        print(f'MHC={mhc} or predictor={predictor} is not valid')
    return alleles


### normalize allele name
# MHC-I
# MixMHCpred: A0101
# NetMHCpan: HLA-A01:01 or HLA-A*01:01
# standard: A*01:01
def MHCIAlleleTransform(allele, From='mixmhcpred', To='netmhcpan'):
    # standardize
    if From == 'mixmhcpred':
        gene, group, prot = allele[0], allele[1:-2], allele[-2:]
    elif From == 'netmhcpan':
        allele = allele.split('-')[1].replace('*', '')
        code, prot = allele.split(':')
        gene, group = code[0], code[1:]
    else: # standard
        gene, code = allele.split('*')
        group, prot = code.split(':')
    
    # transform
    if To == 'mixmhcpred':
        return f'{gene}{group}{prot}'
    elif To == 'netmhcpan':
        return f'HLA-{gene}{group}:{prot}'
    else:
        return f'{gene}*{group}:{prot}'


# MHC-II
# MixMHCpred: DRB1_03_01, DPA1_01_03__DPB1_01_01
# NetMHCpan: DRB1_0301, HLA-DPA10103-DPB10101
# standard: DRB1*03:01, DPA1*01:03_DPB1*01:01
def MHCIIAlleleTransform(allele, From='mixmhcpred', To='netmhcpan'):
    ### DR
    if 'DR' in allele:
        # standardize
        if From == 'mixmhcpred':
            gene, group, prot = allele.split('_')
        elif From == 'netmhcpan':
            gene, code = allele.split('_')
            group, prot = code[:-2], code[-2:]
        else: # standard
            gene, code = allele.split('*')
            group, prot = code.split(':')
        
        # transform
        if To == 'mixmhcpred':
            return f'{gene}_{group}_{prot}'
        elif To == 'netmhcpan':
            return f'HLA-{gene}_{group}{prot}'
        else:
            return f'{gene}*{group}:{prot}'
    
    ### DP, DQ (combination)
    else:
        # standardize
        if From == 'mixmhcpred': # DPA1_01_03__DPB1_01_01
            allele_a, allele_b = allele.split('__')
            gene_a, group_a, prot_a = allele_a.split('_')
            gene_b, group_b, prot_b = allele_b.split('_')
        elif From == 'netmhcpan': # HLA-DPA10103-DPB10101
            allele_a, allele_b = allele[4:].split('-')
            gene_a, group_a, prot_a = allele_a[:4], allele_a[4:-2], allele_a[-2:]
            gene_b, group_b, prot_b = allele_b[:4], allele_b[4:-2], allele_b[-2:]
        else: # DPA1*01:03_DPB1*01:01
            allele_a, allele_b = allele.split('_')
            gene_a, code_a = allele_a.split('*')
            group_a, prot_a = code_a.split(':')
            gene_b, code_b = allele_b.split('*')
            group_b, prot_b = code_b.split(':')

        # transform
        if To == 'mixmhcpred':
            return f'{gene_a}_{group_a}_{prot_a}__{gene_b}_{group_b}_{prot_b}'
        elif To == 'netmhcpan':
            return f'HLA-{gene_a}{group_a}{prot_a}-{gene_b}{group_b}{prot_b}'
        else:
            return f'{gene_a}*{group_a}:{prot_a}_{gene_b}*{group_b}:{prot_b}'


# parse xls output from netMHCpan
def load_netmhcpan_prediction(xls_file, rank_name='Rank', score_name='Score'): 
    # extract alleles, map to Rank.N column name
    with open(xls_file, 'r') as f:
        alleles = f.readline().strip().split()
    
    allele_rank_dict = defaultdict(str)
    allele_score_dict = defaultdict(str)
    for i, allele in enumerate(alleles):
        if allele in  allele_rank_dict.keys():
            continue
        if i == 0:
            allele_rank_dict[allele] = rank_name
            allele_score_dict[allele] = score_name
        else:
            allele_rank_dict[allele] = '{}.{}'.format(rank_name, i)
            allele_score_dict[allele] = '{}.{}'.format(score_name, i)
    alleles = list(set(alleles))

    # parse output
    predict_dict = dict()
    df = pd.read_csv(xls_file, sep='\t', header=1).drop_duplicates()
    for idx, row in df.iterrows():
        peptide = row['Peptide']
        predict_dict[peptide] = defaultdict(dict)
        for allele, rank_name in allele_rank_dict.items():
            predict_dict[peptide][allele]['rank'] = row[rank_name]
        for allele, score_name in allele_score_dict.items():
            predict_dict[peptide][allele]['score'] = row[score_name]
    
    return predict_dict


###############################
###   Metric Calculation    ###
###############################

### identify the best epitope for each allele
class BestEpi():
    def __init__(
            self,
            pept_df,        # peptides
            alleles,        # MHC alleles
            mhc_bind_df,    # MHC binding predictions
            pept_mt_col='MT_epitope',
            pept_wt_cols=['WT_epitope_left_aligned', 'WT_epitope_right_aligned', 'WT_epitope_both_aligned'],
            mhc_allele_col='MHC',
            mhc_pept_col='Peptide',
            mhc_core_col='Core',
            mhc_rank_col='Rank',
            mhc_score_col='Score'
    ):
        # variables
        self.pept_df = pept_df
        self.alleles = alleles
        self.mhc_bind_df = mhc_bind_df
        self.pept_mt_col = pept_mt_col
        self.pept_wt_cols = pept_wt_cols
        self.mhc_allele_col = mhc_allele_col
        self.mhc_pept_col = mhc_pept_col
        self.mhc_core_col = mhc_core_col
        self.mhc_rank_col = mhc_rank_col
        self.mhc_score_col = mhc_score_col

        # prediction dict
        self.mhc_bind_df = self.mhc_bind_df.set_index([self.mhc_pept_col, self.mhc_allele_col], drop=True).sort_index()
        self.rank_dict = self.mhc_bind_df[self.mhc_rank_col].to_dict()

    
    def __call__(self, match_pept_length=False):
        pepts = self.pept_df[self.pept_mt_col]
        results = list()
        for allele in self.alleles:
            # best MT
            mt_scores = [self.rank_dict.get((pept, allele), 101) for pept in pepts]
            if len(mt_scores) == 0:
                best_mt = None
            elif min(mt_scores) == 101:
                best_mt = None
            else:
                best_idx = np.argmin(mt_scores)
                best_mt = pepts[best_idx]

            # best aligned WT
            if best_mt is not None:
                wt_pepts = [self.pept_df.loc[best_idx, col] for col in self.pept_wt_cols if type(self.pept_df.loc[best_idx, col])==str]
                wt_pepts = list(set(wt_pepts))
                if match_pept_length: # len(MT) == len(WT)
                    wt_pepts = [pept for pept in wt_pepts if len(pept) == len(best_mt)]
                wt_scores = [self.rank_dict.get((pept, allele), 101) for pept in wt_pepts]
                if len(wt_scores) == 0:
                    best_wt = None
                elif min(wt_scores) == 101:
                    best_wt = None
                else:
                    best_wt = wt_pepts[np.argmin(wt_scores)]
            else:
                best_wt = None
            
            # annotation
            if best_mt is not None:
                mt_dict = {
                    'allele': allele,
                    'MT_epitope': best_mt,
                    'MT_core': self.mhc_bind_df.loc[(best_mt, allele), self.mhc_core_col],
                    'MT_score': self.mhc_bind_df.loc[(best_mt, allele), self.mhc_score_col],
                    'MT_rank': self.mhc_bind_df.loc[(best_mt, allele), self.mhc_rank_col],
                }
            else:
                mt_dict = {'allele': allele, 'MT_epitope':'', 'MT_core':'', 'MT_score':np.nan, 'MT_rank':np.nan}
            
            if best_wt is not None:
                wt_dict = {
                    'WT_epitope': best_wt,
                    'WT_core': self.mhc_bind_df.loc[(best_wt, allele), self.mhc_core_col],
                    'WT_score': self.mhc_bind_df.loc[(best_wt, allele), self.mhc_score_col],
                    'WT_rank': self.mhc_bind_df.loc[(best_wt, allele), self.mhc_rank_col],
                }
            else:
                wt_dict = {'WT_epitope':'', 'WT_core':'', 'WT_score':np.nan, 'WT_rank':np.nan}

            result = {**mt_dict, **wt_dict}

            # pseudo-core
            ref_seq, alt_seq, alt_core = result['WT_epitope'], result['MT_epitope'], result['MT_core']
            if (len(ref_seq) == len(alt_seq)) and (ref_seq != ''):
                result['WT_pseudo_core'] = self._pseudo_core(ref_seq, alt_seq, alt_core)
            else:
                result['WT_pseudo_core'] = ''
            
            results.append(result)
            
        return pd.DataFrame(results)

    
    # pseudo core
    # ref and alt seqs should be in the same length
    def _pseudo_core(self, ref_seq, alt_seq, alt_core):
        alt_core_map = self._map_seq_to_core(alt_seq, alt_core)
        pseudo_core = list(alt_core)
        for i, j in alt_core_map.items():
            pseudo_core[j] = ref_seq[i]
        return ''.join(pseudo_core)


    # map seq to core positions
    def _map_seq_to_core(seq, core):
        idx_map = OrderedDict()
        i, j = 0, 0
        while (i < len(seq)) and (j < len(core)):
            if seq[i] == core[j]:
                idx_map[i] = j
                i += 1
                j += 1
            else:
                if len(seq) > len(core):
                    i += 1
                elif len(seq) < len(core):
                    j += 1
                else:
                    continue
        return idx_map


### calculate metrics based on the best epitope dataframe
class EpiMetrics():
    def __init__(
            self,
            df,
            mhc='i', # i or ii
            mhc_col='allele',
            mt_pept_col='MT_epitope',
            mt_core_col='MT_core',
            mt_score_col='MT_score',
            mt_rank_col='MT_rank',
            wt_pept_col='WT_epitope',
            wt_core_col='WT_core',
            wt_pseudo_core_col='WT_pseudo_core',
            wt_score_col='WT_score',
            wt_rank_col='WT_rank'
    ):
        self.df = df
        self.mhc = mhc
        self.mhc_col = mhc_col
        self.mt_score_col = mt_score_col
        self.mt_rank_col = mt_rank_col
        self.wt_score_col = wt_score_col
        self.wt_rank_col = wt_rank_col
        self.alleles = self.df[self.mhc_col].unique().tolist()
        
        # pept: raw peptide sequence
        # core: MHC-binding core region
        # seq: for foreignness, pept for MHC-I while core for MHC-II
        self.wt_core_col = wt_core_col
        self.wt_pseudo_core_col = wt_pseudo_core_col
        self.mt_core_col = mt_core_col
        self.wt_pept_col = wt_pept_col
        self.mt_pept_col = mt_pept_col
        self.wt_seq_col = wt_pept_col if mhc == 'i' else wt_core_col # for foreignness
        self.mt_seq_col = mt_pept_col if mhc == 'i' else mt_core_col # for foreignness

        # foreignness
        if foreignness_aval:
            self.Foreignness = Foreignness()

        # SubCRD model
        sub_crd_weights = json.load(open(f'{src_dir}/CRD/SubCRD_weights.json', 'r'))
        self.SubCRD = SubCRD(
            sub_crd_weights[f'MHC{mhc.upper()}ind_POS'],    # position factor
            sub_crd_weights['SUB'],                         # substitution distance
            specific_length=False,                          # length-specific position factor
            specific_allele=False                           # allele-specific position factor
        )

        # PeptCRD model
        self.PeptCRD = PeptCRD(
            f'{src_dir}/PeptCRD_weights.json'
        )

    def __call__(
            self,
            bind_threshold,
            alleles='all',
            metrics=['Robustness', 'PHBR', 'Agretopicity', 'PeptCRD', 'SubCRD', 'Foreignness'],
            best_aggregation_method='masked_max',
            save_all_aggregation=False,
    ):
        results = dict()

        # check foreignness module
        if (not foreignness_aval) and ('Foreignness' in metrics):
            print('Foreignness module is not available, so skip foreignness metric')
            metrics.remove('Foreignness')
        print('Metrics =', metrics)
        
        # filtered by alleles
        if alleles == 'all':
            alleles = self.alleles
        df = self.df[self.df[self.mhc_col].isin(alleles)]
        print('Alleles =', alleles)

        # preparing array
        mask = ~((df[self.mt_rank_col] <= bind_threshold).to_numpy()) # mask array
        weights = df[self.mt_rank_col].apply(lambda x: self._percentile_to_score(1-x/100)) # weight array
        best_idx = np.argmin(df[self.mt_rank_col].to_numpy()) # best binding index

        # scores
        metric_dict = dict()
        if 'Robustness' in metrics:
            results['Robustness'] = (df[self.mt_rank_col] < bind_threshold).sum() # robustness
        if 'PHBR' in metrics:
            metric_dict['PHBR'] = df[self.mt_rank_col].to_numpy()
        if 'Agretopicity' in metrics:
            metric_dict['Agretopicity'] = self._agretopicity(df)
        if 'PeptCRD' in metrics:
            metric_dict['PeptCRD'] = self._crd(df, self.PeptCRD, mhc_bind=True)
        if 'SubCRD' in metrics:
            metric_dict['SubCRD'] = self._crd(df, self.SubCRD)
        if 'Foreignness' in metrics:
            metric_dict['Foreignness'] = self._foreignness(df)

        # aggregation
        aggregations = dict()
        for name, scores in metric_dict.items():
            if name == 'PHBR': # fill NA -> harmonic meaning without masking
                pad_num = 6 if self.mhc == 'i' else 10 # MHC-I has 6 alleles while MHC-II has 10 allele combinations
                scores_ = np.pad(scores, (0, pad_num-len(scores)), constant_values=50) # fill NA with 50 (median rank score)
                scores_ = np.ma.array(scores_, mask=np.zeros((pad_num,))) # no masking
                score = self._aggregate(scores_, method='harmonic') # harmonic mean
            else: # other metrics - max score with masking
                aggregate_scores = self._aggregate_all_methods(scores, mask, weights, best_idx)
                for method, score in aggregate_scores.items():
                    aggregations[f'{name}_{method}'] = score
                score = aggregate_scores[best_aggregation_method] # best method
            results[name] = score # final score
        
        # output
        if save_all_aggregation:
            return {**results, **aggregations}
        else:
            return results

    # aggregation with all methods
    def _aggregate_all_methods(self, scores, mask, weights, best_idx):
        results = dict()
        for masking in [True, False]:
            # score and weight array
            if masking:
                mask_scores = np.ma.array(scores, mask=mask)
                mask_weights = np.ma.array(weights, mask=mask)
            else:
                mask_scores = np.ma.array(scores, mask=np.zeros_like(mask))
                mask_weights = np.ma.array(weights, mask=np.zeros_like(mask))
            # aggregating with each method
            for aggret in ['best', 'min', 'max', 'mean', 'harmonic', 'weight']:
                score = self._aggregate(mask_scores, method=aggret, weights=mask_weights, best_idx=best_idx)
                colname = f'masked_{aggret}' if masking else aggret
                results[colname] = score
        return results
    
    # method = min, max, mean, harmonic, and weight
    def _aggregate(self, scores, method='max', minimal_score=1e-3, weights=None, best_idx=0):
        scores = np.ma.masked_invalid(scores)
        if scores.mask.all():
            return np.nan
        if method == 'min':
            return float(np.nanmin(scores))
        elif method == 'max':
            return float(np.nanmax(scores))
        elif method == 'harmonic':
            return float(scores.count() / np.nansum(1 / (scores + minimal_score)))
        elif method == 'weight':
            weights = np.ma.masked_array(weights, mask=scores.mask)
            return float(np.ma.dot(scores, weights) / np.nansum(weights))
        elif method == 'best':
            score = float(scores[best_idx])
            return 0 if np.isnan(score) else score
        else: # mean
            return float(np.mean(scores))
        
    def _agretopicity(self, df, minimal_score=1e-4):
        scores = np.log10(df[self.mt_score_col] / (df[self.wt_score_col].fillna(0)+minimal_score)).to_numpy()
        return scores

    def _foreignness(self, df):
        pepts = df[self.mt_seq_col].tolist()
        pepts = ['' if type(pept)==float else ''.join(pept.split('-')) for pept in pepts]
        scores = self.Foreignness(pepts)
        return scores
    
    def _crd(self, df, model, mhc_bind=False):
        if mhc_bind:
            scores = df.apply(lambda row: model.score_peptide(
                row[self.wt_pseudo_core_col],
                row[self.mt_core_col],
                row[self.mhc_col],
                row[self.mt_score_col],
            ), axis=1)
        else:
            scores = df.apply(lambda row: model.score_peptide(
                row[self.wt_pseudo_core_col],
                row[self.mt_core_col],
                row[self.mhc_col]
            ), axis=1)
        return scores
    
    def _harmonic_mean(self, arr, minimal_score=1e-3):
        return arr.shape[0] / np.sum(1/(arr+minimal_score))
    
    def _percentile_to_score(self, p, mu=0, sigma=1, z_min=-2.5, z_max=2.5):
        z = norm.ppf(p, loc=mu, scale=sigma)
        z_clip = np.clip(z, z_min, z_max)
        value = (z_clip - z_min) / (z_max - z_min)
        return value