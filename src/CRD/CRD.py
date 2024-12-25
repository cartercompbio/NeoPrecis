#!/bin/python3
# Script Name: CRD.py
# Description: Cross-reactivity distance models
# Author: Kohan

import os, sys, json, yaml, difflib, warnings
import numpy as np
from model import *
warnings.filterwarnings("ignore")


class PeptCRD():
    def __init__(self, ref_file, config_file, ckpt_file):
        # reference
        with h5py.File(ref_file, 'r') as f:
            aa_list = f['aa_list'].asstr()[:].tolist()
            self.aa_dict = {s:i for i,s in enumerate(aa_list)}
            allele_list = f['allele_list'].asstr()[:].tolist()
            self.allele_dict = {s:i for i,s in enumerate(allele_list)}

        # checkpoint path
        self.ckpt = torch.load(ckpt_file, map_location=torch.device('cpu'), weights_only=True)

        # load config
        config = yaml.safe_load(open(config_file, 'r'))
        self.pept_emb_dim = config['model']['pept_emb_dim']

        # model
        self.model = CLF(
            ref_h5_file=ref_file,
            archt=config['model']['architecture'],
            aa_encode_key=config['model']['aa_encode_key'],
            aa_vocab_size=config['model']['aa_vocab_size'],
            aa_encode_dim=config['model']['aa_encode_dim'],
            aa_emb_dim=config['model']['aa_emb_dim'],
            pept_emb_dim=config['model']['pept_emb_dim'],
            peptide_length=config['model']['peptide_length'],
            pos_emb_method=config['model']['pos_emb_method'],
            feature_idx=config['model']['feature_idx'],
        )
        self.model.load_state_dict(self.ckpt['state_dict'])
        self.model.eval()

    # preds = (origin(size=pept_emb_dim), direction(pept_emb_dim), GeoDist(1), scaler(1), PeptCRD, Immgen)
    def score_peptide(self, ref_seq, alt_seq, allele, alt_bind_score):
        # check sequence
        out_dim = self.pept_emb_dim*2 + 4
        if (type(alt_seq) == float) | (alt_seq == ''):
            return np.full((out_dim,), np.nan)
        if (type(ref_seq) == float) | (ref_seq == ''):
            return np.full((out_dim,), np.nan)
        if len(ref_seq) != len(alt_seq):
            return np.full((out_dim,), np.nan)

        # tokenization
        ref_token = self._tokenization(ref_seq)
        alt_token = self._tokenization(alt_seq)
        gene = allele.split('*')[0]
        mhc = 0 if gene in ['A','B','C'] else 1
        allele_id = self.allele_dict[allele]

        # prediction
        peptides = torch.tensor([alt_token, ref_token]).unsqueeze(dim=0)
        features = torch.tensor([mhc, allele_id, alt_bind_score]).unsqueeze(dim=0)
        geo_vectors, immgens = self.model(peptides, features)
        preds = torch.cat((geo_vectors, immgens), dim=1).squeeze().detach().numpy()

        return preds

    # tokenize peptide sequence
    def _tokenization(self, seq):
        return [self.aa_dict[s] for s in seq]


class SubCRD():
    def __init__(self, pos_weights, sub_weights,
                 specific_allele=True, specific_length=False):
        self.pos_weights = pos_weights
        self.sub_weights = sub_weights
        self.specific_allele = specific_allele
        self.specific_length = specific_length
        self.mean_sub_weights = {k: np.mean(list(v.values())) for k,v in sub_weights.items()}
        self.aa_list = list(self.sub_weights.keys())

    # score single mutation
    def score_mutation(self, pos, ref_aa, alt_aa, allele='A*01:01', length=9):
        # position weights
        pos_weights = self.pos_weights[f'{length}mer'] if self.specific_length else self.pos_weights
        if self.specific_allele:
            if allele not in pos_weights:
                gene = allele.split('*')[0]
                if len(gene) > 2: gene = gene[:2]
                pos_weight = pos_weights[gene][f'P{pos+1}']
            else:
                pos_weight = pos_weights[allele][f'P{pos+1}']
        else:
            pos_weight = pos_weights[f'P{pos+1}']

        # substitution weight
        if (ref_aa not in self.aa_list) & (alt_aa not in self.aa_list):
            return 0
        if ref_aa not in self.aa_list:
            sub_weight = self.mean_sub_weights[alt_aa]
        elif alt_aa not in self.aa_list:
            sub_weight = self.mean_sub_weights[ref_aa]
        else:
            sub_weight = self.sub_weights[ref_aa][alt_aa]
        
        return pos_weight * sub_weight
    
    # score peptide sequence
    def score_peptide(self, ref_seq, alt_seq, allele):
        # check sequence
        if (type(alt_seq) == float) | (alt_seq == ''):
            return np.nan
        if (type(ref_seq) == float) | (ref_seq == ''):
            return np.nan
        if len(ref_seq) == len(alt_seq):
            mismatches = self._get_missense_mismatch(ref_seq, alt_seq)
        else:
            mismatches = self._get_indel_mismatch(ref_seq, alt_seq)
        
        # score
        score = np.sum([self.score_mutation(pos, ref, alt, allele=allele, length=len(alt_seq)) for (pos, ref, alt) in mismatches])
        
        return score

    # get mismatches for missense mutation
    # return [(pos, ref_aa, alt_aa), ...]
    def _get_missense_mismatch(self, ref_seq, alt_seq):
        length = len(ref_seq)
        mismatch_list = list()
        for i in range(length):
            if ref_seq[i] != alt_seq[i]:
                mismatch_list.append((i, ref_seq[i], alt_seq[i]))
        return mismatch_list
        
    # get mismatches for indel mutation
    # return [(pos, ref_aa, alt_aa), ...]
    def _get_indel_mismatch(self, ref_seq, alt_seq):
        mismatch_list = list()
        matcher = difflib.SequenceMatcher(None, ref_seq, alt_seq)
        opcodes = matcher.get_opcodes()
        optypes = [code[0] for code in opcodes]

        # bug: replace -> deletion + insertion
        if ('insert' in optypes) and (len(ref_seq)==len(alt_seq)):
            for i in range(len(alt_seq)):
                if ref_seq[i] != alt_seq[i]:
                    mismatch_list.append((i, ref_seq[i], alt_seq[i]))
            return mismatch_list

        for (annot, ref_s, ref_e, alt_s, alt_e) in matcher.get_opcodes():
            if annot == 'equal':
                continue

            # replacement
            elif annot == 'replace':
                ref_len = ref_e - ref_s
                alt_len = alt_e - alt_s
                # substitution
                if alt_len == ref_len:
                    for i in range(alt_s, alt_e):
                        mismatch_list.append((i, ref_seq[ref_s+i-alt_s], alt_seq[i]))
                # substitution + insertion
                elif alt_len > ref_len:
                    for i in range(alt_s, alt_e):
                        if (i - alt_s) < ref_len:
                            mismatch_list.append((i, ref_seq[ref_s+i-alt_s], alt_seq[i]))
                        else:
                            mismatch_list.append((i, '', alt_seq[i]))
                # substitution + deletion
                else:
                    for i in range(ref_s, ref_e):
                        if (i - ref_s) < alt_len:
                            mismatch_list.append((alt_s+i-ref_s, ref_seq[i], alt_seq[alt_s+i-ref_s]))
                        else:
                            mismatch_list.append((alt_e-1, ref_seq[i], ''))

            # insertion
            elif annot == 'insert':
                for i in range(alt_s, alt_e):
                    mismatch_list.append((i, '', alt_seq[i]))

            # deletion
            else:
                for i in range(ref_s, ref_e):
                    mismatch_list.append((alt_s, ref_seq[i], ''))

        return mismatch_list