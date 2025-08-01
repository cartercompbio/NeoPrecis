#!/bin/python3
# Script Name: construct_landscape.py
# Description: Construct neoantigen landscape and compute tumor-centric scores
# Author: Kohan

import os, sys, argparse
from collections import defaultdict
import numpy as np
import pandas as pd

def ArgumentParser(args=None):
    parser = argparse.ArgumentParser(prog='Construct neoantigen landscape and compute tumor-centric scores',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # required arguments
    parser.add_argument('input_file', type=str, help='Neoangien file generated by neoagfinder.sh')
    parser.add_argument('cluster_file', type=str, help='Cluster file generated by PyClone')
    parser.add_argument('loci_file', type=str, help='Loci file generated by PyClone')
    parser.add_argument('output_file', type=str, help='Output file')
    # optional arguments
    parser.add_argument('--rank_ref', type=str, default='', help='Percentile ranking reference file')
    return parser


# annotate neoantigens with pyclone loci information
def PycloneLociAnnotation(neoag_df, loci_df, id_col='mutation_id', cluster_col='cluster_id', prev_col='cellular_prevalence'):
    index_cols = ['#CHROM', 'POS', 'REF', 'ALT']
    loci_df[index_cols] = loci_df[id_col].str.split('_', expand=True) # split ID col into chr, pos, ref, alt
    loci_df = loci_df[index_cols + [cluster_col, prev_col]] # keep necessary columns
    loci_df.loc[:, '#CHROM'] = loci_df['#CHROM'].apply(lambda x: 'chr' + str(x)) # 1 -> chr1
    loci_df.loc[:, 'POS'] = loci_df['POS'].astype(int)
    neoag_df = neoag_df.merge(loci_df, on=index_cols, how='left')
    return neoag_df


# compute NP-LandscapeClone
class SubcloneScoring():
    def __init__(
        self,
        mutation_df,
        cluster_df,
        cluster_col='cluster_id',
        mutation_prev_col='cellular_prevalence', 
        cluster_prev_col='mean'
    ):
        # data
        self.mutation_df = mutation_df
        self.cluster_df = cluster_df

        # column names
        self.cluster_col = cluster_col
        self.mutation_prev_col = mutation_prev_col
        self.cluster_prev_col = cluster_prev_col

        # add clonal prevalence to mutation_df
        cluster_prev_dict = self.cluster_df.set_index([self.cluster_col])[self.cluster_prev_col].to_dict()
        self.mutation_df[self.cluster_prev_col] = self.mutation_df[self.cluster_col].apply(lambda x: cluster_prev_dict.get(x, np.nan))


    # scoring for each sample/tumor
    def scoring(
        self,
        metric_cols=['NP-Immuno-I', 'NP-Immuno-II'],                                    # target metrics in mutation_df
        dual_metrics_dict={'NP-Immuno-dual': ('NP-Immuno-I', 'NP-Immuno-II')},          # metrics to be combined (MHCI + MHCII)
        mutation_pooling='sum',                                                         # method for mutation pooling
        subclone_pooling='weight',                                                      # method for subclonal pooling
    ):
        score_dict = dict()

        # scoring by single metric
        for metric in metric_cols:
            score_dict[metric] = self._single_pathway_scoring(
                self.mutation_df,
                self.cluster_df,
                metric,
                mutation_pooling=mutation_pooling,
                subclone_pooling=subclone_pooling
            )

        # scoring by dual metrics
        for name, (metric_a, metric_b) in dual_metrics_dict.items():
            score_dict[name] = self._dual_pathway_scoring(
                self.mutation_df,
                self.cluster_df,
                metric_a,
                metric_b,
                mutation_pooling=mutation_pooling,
                subclone_pooling=subclone_pooling
            )
        
        return score_dict
    

    # scoring with single metric
    # pooling = 'max' or 'sum' or 'mean'
    def _single_pathway_scoring(self, mutation_df, cluster_df, metric, mutation_pooling='max', subclone_pooling='mean'):
        # mutation pooling
        score_df = mutation_df.groupby([self.cluster_col])[metric]
        if mutation_pooling == 'max':
            score_df = score_df.max()
        elif mutation_pooling == 'sum':
            score_df = score_df.sum()
        elif mutation_pooling == 'mean':
            score_df = score_df.mean()
        else:
            print('mutation_pooling should be max, sum, or mean')
            return
        
        # cluster
        if cluster_df.index.names != self.cluster_col:
            cluster_df = cluster_df.set_index(self.cluster_col)
        cluster_df[metric] = score_df # appending cluster score
        cluster_df[metric] = cluster_df[metric].fillna(0)

        # subclone pooling (with weight)
        cluster_df['weighted_score'] = cluster_df[metric] * cluster_df[self.cluster_prev_col]
        if subclone_pooling == 'max':
            score = cluster_df['weighted_score'].max()
        elif subclone_pooling == 'sum':
            score = cluster_df['weighted_score'].sum()
        elif subclone_pooling == 'mean':
            score = cluster_df['weighted_score'].mean()
        elif subclone_pooling == 'weight':
            score = cluster_df['weighted_score'].sum() / cluster_df[self.cluster_prev_col].sum()
        else:
            print('subclone_pooling should be max, sum, mean, or weight')
            return

        return score


    # scoring with dual metrics (MHC-I + MHC-II)
    def _dual_pathway_scoring(self, mutation_df, cluster_df, mhci_method, mhcii_method, mutation_pooling='max', subclone_pooling='mean'):
        # mutation pooling
        mhci_score_df = mutation_df.groupby([self.cluster_col])[mhci_method]
        if mutation_pooling == 'max':
            mhci_score_df = mhci_score_df.max()
        elif mutation_pooling == 'sum':
            mhci_score_df = mhci_score_df.sum()
        elif mutation_pooling == 'mean':
            mhci_score_df = mhci_score_df.mean()
        else:
            print('mutation_pooling should be max, sum, or mean')
            return

        mhcii_score_df = mutation_df.groupby([self.cluster_col])[mhcii_method]
        if mutation_pooling == 'max':
            mhcii_score_df = mhcii_score_df.max()
        elif mutation_pooling == 'sum':
            mhcii_score_df = mhcii_score_df.sum()
        elif mutation_pooling == 'mean':
            mhcii_score_df = mhcii_score_df.mean()
        else:
            print('mutation_pooling should be max, sum, or mean')
            return
        
        score_df = mhci_score_df * mhcii_score_df
        
        # cluster
        if cluster_df.index.names != self.cluster_col:
            cluster_df = cluster_df.set_index(self.cluster_col)
        cluster_df['score'] = score_df # appending cluster score
        cluster_df['score'] = cluster_df['score'].fillna(0)

        # subclone pooling (with weight)
        cluster_df['weighted_score'] = cluster_df['score'] * cluster_df[self.cluster_prev_col]     
        if subclone_pooling == 'max':
            score = cluster_df['weighted_score'].max()
        elif subclone_pooling == 'sum':
            score = cluster_df['weighted_score'].sum()
        elif subclone_pooling == 'mean':
            score = cluster_df['weighted_score'].mean()
        elif subclone_pooling == 'weight':
            score = cluster_df['weighted_score'].sum() / cluster_df[self.cluster_prev_col].sum()
        else:
            print('subclone_pooling should be max, sum, mean, or weight')
            return

        return score


def FindPercentile(score, ref_arr):
    return np.searchsorted(ref_arr, score, side='right')


def Main(input_file, cluster_file, loci_file, output_file, rank_ref_file):
    df = pd.read_csv(input_file)
    loci_df = pd.read_csv(loci_file, sep='\t')
    cluster_df = pd.read_csv(cluster_file, sep='\t')
    df['NP-Immuno-dual'] = df['NP-Immuno-I'] * df['NP-Immuno-II']
    df = PycloneLociAnnotation(df, loci_df)
    
    ### metric computation
    metric_dict = defaultdict(dict)
    
    # w/o clonality
    metric_dict['value']['TMB'] = df.shape[0]
    metric_dict['value']['TNB'] = (df['PHBR-I'] <= 2).sum()
    metric_dict['value']['NPB'] = (df['NP-Immuno-dual'] >= 0.16).sum()
    metric_dict['value']['NP-LandscapeSum'] = np.log10(df['NP-Immuno-dual'].sum())

    # w/ clonality
    metric_dict['value']['NP-LandscapeCCF'] = np.log10((df['NP-Immuno-dual'] * df['cellular_prevalence']).sum())
    subclone_scoring = SubcloneScoring(df, cluster_df) # subclone scoring obj
    subclone_metric_dict = subclone_scoring.scoring() # scoring with subclone architecture
    metric_dict['value']['NP-LandscapeClone'] = np.log10(subclone_metric_dict['NP-Immuno-dual'])

    ### percentile ranking
    rank_df = pd.read_csv(rank_ref_file, index_col=0)
    for col in rank_df.columns:
        cancer, metric = col.split('_')
        percentile = FindPercentile(metric_dict['value'][metric], rank_df[col])
        metric_dict[f'percentile_{cancer}'][metric] = percentile/100
    
    ### result
    result_df = pd.DataFrame(metric_dict)
    result_df.to_csv(output_file)


if __name__=='__main__':
    args = ArgumentParser().parse_args()
    Main(args.input_file, args.cluster_file, args.loci_file, args.output_file, args.rank_ref)