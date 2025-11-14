# benchmarking and interpreting the PeptCRD model

import os
import sys
import h5py
from collections import defaultdict
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
import seaborn as sns
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import least_squares
from scipy.linalg import svd
from sklearn.metrics import roc_auc_score, average_precision_score
from adjustText import adjust_text
import warnings
from utils import NonParamTest

project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_dir)
from neoprecis.CRD.model import CLF

warnings.filterwarnings('ignore')
dpi = 600 # plot

###################
# a.a. properties #
###################
aa_list = list('GAVLIMPFYWSTCNQDEKRH')
aa_dict = {
    'W': 'aromatic', 'F': 'aromatic', 'Y': 'aromatic',
    'H': 'basic', 'K': 'basic', 'R': 'basic',
    'D': 'acidic', 'E': 'acidic',
    'N': 'amidic', 'Q': 'amidic',
    'I': 'non-polar', 'L': 'non-polar', 'V': 'non-polar', 'M': 'non-polar', 'A': 'non-polar', 'P': 'non-polar', 'G': 'non-polar',
    'S': 'polar', 'T': 'polar', 'C': 'polar',
}
aa_color_map = {
    'aromatic': 'purple',
    'acidic': 'red',
    'basic': 'blue',
    'amidic': 'orange',
    'polar': 'cyan',
    'non-polar': 'green',
    'N/S': to_rgba('grey', alpha=0.5),
}


########################
# model interpretation #
########################
# pept_emb_dim should be 2 for affine transformation
class ModelInterpretation():
    def __init__(self, ref_file, ckpt_file):
        ### ref
        with h5py.File(ref_file, 'r') as ref:
            self.ref_aa_list = list(ref['aa_list'].asstr())
            self.ref_allele_list = list(ref['allele_list'].asstr())
            self.ref_motifs = ref['motifs'][:]
            self.ref_pos_facs = ref['position_factors'][:]
            self.blosum_aa_encode = ref['aa_blosum_encodes'][:] # BLOSUM62
            self.pmbec_aa_encode = ref['aa_pmbec_encodes'][:] # PMBEC
            self.blosum_aa_pc2_encode = ref['aa_blosum_pc2_encodes'][:] # BLOSUM62
            self.pmbec_aa_pc2_encode = ref['aa_pmbec_pc2_encodes'][:] # PMBEC

        ### BLOSUM62 PC2 embedding distance
        
        # distance based on PCA
        aa_blosum_df = pd.DataFrame(self.blosum_aa_pc2_encode, index=self.ref_aa_list, columns=['emb1', 'emb2'])
        self.aa_blosum_dist_series = self._calculate_aa_emb_distance(aa_blosum_df)
        
        # distance based on original encoding
        aa_blosum_df = pd.DataFrame(self.blosum_aa_encode, index=self.ref_aa_list, columns=self.ref_aa_list)
        self.aa_blosum_subdist_series = -aa_blosum_df.stack() # negative for distance

        ### PMBEC PC2 embedding distance
        
        # distance based on PCA
        aa_pmbec_df = pd.DataFrame(self.pmbec_aa_pc2_encode, index=self.ref_aa_list, columns=['emb1', 'emb2'])
        self.aa_pmbec_dist_series = self._calculate_aa_emb_distance(aa_pmbec_df)
        
        # distance based on original encoding
        aa_pmbec_df = pd.DataFrame(self.pmbec_aa_encode, index=self.ref_aa_list, columns=self.ref_aa_list)
        self.aa_pmbec_subdist_series = -aa_pmbec_df.stack() # negative for distance
        
        ### ckpt
        self.ckpt = torch.load(ckpt_file)
        self.aa_emb_df = self._build_aa_emb_df(self.ckpt)
        self.aa_emb_dist_series = self._calculate_aa_emb_distance(self.aa_emb_df)
        self.mhci_pos_facs = self.ckpt['state_dict']['model.mhci_glb_pos_fac'].cpu().numpy()
        self.mhcii_pos_facs = self.ckpt['state_dict']['model.mhcii_glb_pos_fac'].cpu().numpy()

        ### model
        self.model = CLF(
            ref_h5_file=self.ckpt['hyper_parameters']['ref_h5_file'],
            archt=self.ckpt['hyper_parameters']['archt'],
            aa_emb_dim=self.ckpt['hyper_parameters']['aa_emb_dim'],
            pept_emb_dim=self.ckpt['hyper_parameters']['pept_emb_dim'],
            feature_idx=self.ckpt['hyper_parameters']['feature_idx']
        )
        self.model.load_state_dict(self.ckpt['state_dict'])
        self.model.eval()

        ### variables
        self.num_pos = 9
        self.num_aa = 21
        self.aa_emb_dim = self.ckpt['hyper_parameters']['aa_emb_dim']
        self.pept_emb_dim = self.ckpt['hyper_parameters']['pept_emb_dim']


    # annotate with blosum, substitution, and sub+pos distances
    # pos has to be 0-based
    def AnnotateSubDistance(self, df, wt_aa_col='WT_aa', mt_aa_col='MT_aa', mhc_col='mhc', pos_col='pos'):
        # substitution distance: baseline (BLOSUM62 and PMBEC)
        df['BLOSUMDist'] = df.set_index([wt_aa_col, mt_aa_col]).index.map(self.aa_blosum_dist_series) # PCA distance
        df['PMBECDist'] = df.set_index([wt_aa_col, mt_aa_col]).index.map(self.aa_pmbec_dist_series) # PCA distance
        df['BLOSUMDist(sub)'] = df.set_index([wt_aa_col, mt_aa_col]).index.map(self.aa_blosum_subdist_series) # substitution distance
        df['PMBECDist(sub)'] = df.set_index([wt_aa_col, mt_aa_col]).index.map(self.aa_pmbec_subdist_series) # substitution distance

        # substitution distance
        df['SubDist'] = df.set_index([wt_aa_col, mt_aa_col]).index.map(self.aa_emb_dist_series)

        # substitution x position factor
        def _get_pos_facs(row):
            if np.isnan(row[pos_col]):
                return 0
            elif row[mhc_col].lower()=='i':
                return self.mhci_pos_facs[int(row[pos_col])]
            else:
                return self.mhcii_pos_facs[int(row[pos_col])]
        pos_facs = df.apply(lambda row: _get_pos_facs(row), axis=1)
        df['SubPosDist'] = df['SubDist'] * pos_facs

        return df


    # entropy for anchor positions; prob for anchor residues
    # anchor_pos_series: index=[pos, allele], value=binary
    # anchor_residue_dict: {(allele, pos): [(residue, freq), ...]}
    def BuildAnchorMapping(self, entropy_thrs=1.5, prob_thrs=0.15):
        # anchor positions
        pos_fac_df = pd.DataFrame(self.ref_pos_facs, columns=list(range(9)), index=self.ref_allele_list)
        self.anchor_pos_series = (pos_fac_df < entropy_thrs).unstack()

        # anchor residues
        indices = np.where(self.ref_motifs > prob_thrs)
        self.anchor_residue_dict = defaultdict(list)
        for i in range(indices[0].shape[0]):
            allele = self.ref_allele_list[indices[0][i]]
            pos = indices[1][i]
            residue = self.ref_aa_list[indices[2][i]]
            freq = self.ref_motifs[indices[0][i], indices[1][i], indices[2][i]]
            if self.anchor_pos_series.loc[(pos, allele)]:
                self.anchor_residue_dict[(allele, pos)].append((residue, freq))
        

    # enrich_embs = (#alleles, #positions, #aa, emb_dim); motif enrichment
    # enrich_emb_affine_matrices = (#alleles, #positions, pept_emb_dim, pept_emb_dim+1); affine transformation (original emb = substitution emb)
    # enrich_emb_scale_matrices = (#alleles, #positions, pept_emb_dim, pept_emb_dim+1); scaling matrices (direction + scale)
    # enrich_emb_project_scales = (#alleles, #positions, pept_emb_dim); scaling factor along each axis
    def MotifEnrichmentInterpretation(self):
        ### motif enrichment
        self.enrich_embs = list()                                   # empty enriched embedding list
        for idx, allele in enumerate(self.ref_allele_list):         # for each allele
            motif = self.ref_motifs[idx]                            # motif
            embs = self._enrich_by_motif(motif)[:,:-1,:]            # motif enriched embedding (remove '-'))
            self.enrich_embs.append(embs[np.newaxis,:])             # append to embedding list
        self.enrich_embs = np.concatenate(self.enrich_embs, axis=0) # concat embedding list

        ### affine transformation
        self.enrich_emb_affine_matrices = np.zeros((len(self.ref_allele_list), self.num_pos, self.pept_emb_dim, self.pept_emb_dim+1))   # empty affine matrices
        self.enrich_emb_scale_matrices = np.zeros((len(self.ref_allele_list), self.num_pos, self.pept_emb_dim, self.pept_emb_dim+1))    # empty scaling matrices (direction + scale)
        self.enrich_emb_project_scales = np.zeros((len(self.ref_allele_list), self.num_pos, self.pept_emb_dim))                         # empty x, y scales (aligned with the sub_emb axes)
        sub_emb = self.aa_emb_df.to_numpy()                                                         # substitution embedding
        for i, allele in enumerate(self.ref_allele_list):                                           # for each allele
            for j in list(range(self.num_pos)):                                                     # for each position
                enrich_emb = self.enrich_embs[i, j, :, :][:]                                        # enriched embedding

                affine_matrix = self._affine_transformation(sub_emb, enrich_emb)                    # calculate affine matrix
                self.enrich_emb_affine_matrices[i, j, :, :] = affine_matrix[:]                      # append to affine matrices

                A = affine_matrix[:,:-1]
                S = self._axis_scaling(A)                                                           # calculate first-level axis scaling
                self.enrich_emb_scale_matrices[i, j, :, :] = S[:]

                scales = self._project_scaling(S)                                                   # calculate projected scaling (aligned with x, y axes of sub_emb)
                self.enrich_emb_project_scales[i, j, :] = scales[:]

    
    # calculate the allele benefit score
    # allele_pos_annot_df: per allele and position; columns = [emb1_scaling, emb2_scaling, anchor_pos, residue, group, freq]
    # allele_annot_df: per allele; add "benefitScore"
    def AlleleSummarization(self):
        # annotation per allele and position
        self.allele_pos_annot_df = self._annot_allele_pos()

        # annotation per allele (allele benefit score)
        self.allele_annot_df = self._summarize_allele(self.allele_pos_annot_df)


    # calculate the difference of pairwise distance between baseline embedding and motif-enriched embedding
    # baseline embedding: motifs' a.a. frequency of non-anchor position
    # output = (num_allele, num_pos, num_aa, num_aa)
    def PairwiseDistanceDifference(self, embs, nonanchor_pos_list=[4,6,7]):
        # baseline distance
        freq_arr = self.ref_motifs[:,nonanchor_pos_list,:].mean(axis=0).mean(axis=0)        # a.a. frequency of non-anchor positions
        baseline_motif = np.tile(freq_arr[np.newaxis, :], (self.num_pos, 1))                # baseline motif based on a.a. frequency
        baseline_embs = self._enrich_by_motif(baseline_motif)                               # (num_pos, num_aa, emb_dim)
        baseline_dist_matrix = self._matrix_to_distance(baseline_embs)                      # (num_pos, num_aa, num_aa)

        # allele-specific distance
        D = np.zeros((len(self.ref_allele_list), self.num_pos, self.num_aa, self.num_aa))   # empty difference matrix
        for idx in range(embs.shape[0]):
            dist_matrix = self._matrix_to_distance(embs[idx])
            D[idx] = (dist_matrix - baseline_dist_matrix) / baseline_dist_matrix

        return D
    

    # compute affine matrix
    def _affine_transformation(self, original_emb, transformed_emb):
        dim = self.pept_emb_dim
        init_params = np.hstack([np.eye(dim).flatten(), np.zeros(dim)]) # init parameters
        result = least_squares(
            self._affine_loss_function,
            init_params,
            args=(original_emb, transformed_emb)
        )
        A = result.x[:dim**2].reshape(dim, dim)             # linear transformation matrix
        t = result.x[dim**2:]                               # transition vector
        affine_matrix = np.hstack([A, t[:, np.newaxis]])    # combine A and t into an affine matrix
        return affine_matrix


    # loss function for affine transformation
    def _affine_loss_function(self, params, original_emb, transformed_emb):
        dim = self.pept_emb_dim
        A = params[:dim**2].reshape(dim, dim)       # linear transformation matrix
        t = params[dim**2:]                         # transition vector
        reconstruct_emb = original_emb @ A.T + t    # affine transformation
        return (reconstruct_emb - transformed_emb).flatten()
    
    
    # interpret the transformation (only for 2D matrix)
    # (reflection, rotation, x_scaling, y_scaling, reflection, rotation)
    def _affine_interpretation(self, affine_matrix):
        A = affine_matrix[:, :2]                        # remove transition
        U, S, Vt = svd(A)                               # SVD
        rf1, rt1 = self._calculate_rotation_degree(Vt)  # first reflection and rotation from Vt
        xs, ys = S[0], S[1]                             # scaling from S
        rf2, rt2 = self._calculate_rotation_degree(U)   # second reflection and rotation from U
        return (rf1, rt1, xs, ys, rf2, rt2)


    # calculate the determinant (reflection) and rotation degree from a rotation matrix
    def _calculate_rotation_degree(self, R):
        det = np.linalg.det(R)
        reflection = det < 0  # Check for reflection
        if reflection:
            R = R.copy()  # Avoid modifying the input matrix directly
            R[1, :] *= -1  # Flip reflection
        theta = np.arctan2(R[1, 0], R[0, 0])  # Extract rotation angle
        degree = np.degrees(theta)  # Convert to degrees
        return reflection, degree

    
    # extract the first-level axis and scaling from affine matrix
    # affine matrix w/o transition
    def _axis_scaling(self, affine_matrix):
        M = np.zeros((self.pept_emb_dim, self.pept_emb_dim+1)) # output matrix
        U, S, Vt = svd(affine_matrix) # SVD
        for i in range(self.pept_emb_dim):
            axis, scale = Vt[i], S[i]
            M[i][:-1] = axis
            M[i][-1] = scale
        return M
    

    # project the scaling vectors to x and y
    def _project_scaling(self, S):
        x = S[0][0]*S[0][2] + S[1][0]*S[1][2]
        y = S[0][1]*S[0][2] + S[1][1]*S[1][2]
        return np.abs(np.array([x, y]))

    
    # motif enrichment for each position each a.a.
    # output: (num_pos, num_aa, emb_dim)
    def _enrich_by_motif(self, motif):
        # motif embedding
        motif = torch.tensor(motif).float().unsqueeze(0)                                # (num_pos, num_aa) -> (1, num_pos, num_aa)
        motif_emb = self.model.model._aa_embedding(motif, OHE=False)                    # (1, num_pos, aa_emb_dim)
        conv_layer = self.model.model._build_kernels(motif_emb, 1)                      # (1, num_kernels, num_pos, aa_emb_dim)
        
        # motif enrichment
        enriched_matrix = list()
        for i in range(self.num_aa):
            # a.a. embedding
            seq = torch.tensor([i,]*self.num_pos).long().unsqueeze(0)                   # (1, num_pos); single a.a.
            seq_emb = self.model.model._aa_embedding(seq, OHE=True)                     # (1, num_pos, aa_emb_dim); a.a. embedding
            seq_emb_enriched = self.model.model._motif_enrichment(seq_emb, conv_layer)  # (1, num_pos, pept_emb_dim); motif enrichment
            enriched_matrix.append(seq_emb_enriched.detach().numpy())

        enriched_matrix = np.concatenate(enriched_matrix, axis=0).transpose((1,0,2))    # (num_pos, num_aa, pept_emb_dim)

        return enriched_matrix
    

    # calculate the pairwise distance across the second dimension
    # input = (A, B, C); output = (A, B, B)
    def _matrix_to_distance(self, matrix):
        diff = matrix[:, :, None, :] - matrix[:, None, :, :]
        distances = np.linalg.norm(diff, axis=-1)
        return distances
        
    
    def _build_aa_emb_df(self, ckpt):
        blosum_encode = ckpt['state_dict']['model.aa_encode_layer'].cpu().numpy() # blosum encode matrix
        aa_emb_weight = ckpt['state_dict']['model.aa_emb_layer.weight'].cpu().numpy() # aa_emb weight
        aa_emb_bias = ckpt['state_dict']['model.aa_emb_layer.bias'].cpu().numpy() # aa_emb bias
        aa_embs = np.matmul(blosum_encode, aa_emb_weight.T) + aa_emb_bias # aa_emb = blosum_encode * weight + bias
        aa_emb_df = pd.DataFrame(aa_embs, index=self.ref_aa_list, columns=['emb1', 'emb2']) # build dataframe
        aa_emb_df = aa_emb_df.drop(index=['-'])
        return aa_emb_df
    

    def _calculate_aa_emb_distance(self, aa_emb_df, drop_duplicates=False):
        pairwise_distances = pdist(aa_emb_df, metric='euclidean') # pairwise distance
        distance_matrix = squareform(pairwise_distances) # matrix
        distance_df = pd.DataFrame(distance_matrix, index=aa_emb_df.index, columns=aa_emb_df.index) # dataframe
        aa_emb_distance = distance_df.stack() # multi-index series
        if drop_duplicates: # drop duplicated pairs
            aa_emb_distance = aa_emb_distance[aa_emb_distance.index.get_level_values(0) < aa_emb_distance.index.get_level_values(1)]
        return aa_emb_distance
    

    # determine the probability threshold for anchor residues
    # min(max residue of each anchor position)
    def _determine_anchor_thrs(self, anchor_pos_series):
        anchor_pos_list = anchor_pos_series[anchor_pos_series].index.tolist() # anchor positions
        max_prob_list = list()
        for pos, allele in anchor_pos_list: # for each anchor position
            allele_idx = self.ref_allele_list.index(allele) # allele idx
            max_prob = self.ref_motifs[allele_idx, pos].max() # max prob
            max_prob_list.append(max_prob)
        return min(max_prob_list)
    

    # annotate allele-position with scaling factors and anchor residue
    def _annot_allele_pos(self):
        # build annot df
        df = list()
        for allele_idx, allele in enumerate(self.ref_allele_list):
            for pos_idx in range(self.num_pos):
                scale_x, scale_y = self.enrich_emb_project_scales[allele_idx, pos_idx, :] # scaling factors
                df.append([allele, pos_idx, scale_x, scale_y]) # append to annot df
        df = pd.DataFrame(df, columns=['allele', 'position', 'emb1_scaling', 'emb2_scaling'])

        # anchor position (binary)
        df = df.set_index(['position', 'allele']) # per allele and position
        df['anchor_pos'] = self.anchor_pos_series
        df = df.reset_index()

        # add anchor residue (most frequent one)
        residue_list, group_list, freq_list = list(), list(), list()
        for idx, row in df.iterrows():
            if not row['anchor_pos']: # not anchor position
                group, residue, freq = 'N/S', 'N/S', np.nan
            else:
                residues = self.anchor_residue_dict[(row['allele'], row['position'])] # get anchor residues
                if len(residues) == 0: # no anchor residue
                    group, residue, freq = 'N/S', 'N/S', np.nan
                else:
                    residues = sorted(residues, key=lambda x: x[1]) # sort by frequency
                    residue = residues[-1][0] # most frequent residue
                    group = aa_dict[residue] # a.a. group
                    freq = residues[-1][1] # freq. of the residue
            residue_list.append(residue)
            group_list.append(group)
            freq_list.append(freq)
        df['residue'] = residue_list
        df['group'] = group_list
        df['freq'] = freq_list

        # combined scaling (geometric mean)
        df['scaling'] = np.sqrt(df['emb1_scaling'] * df['emb2_scaling'])

        return df
    

    # summarize allele annotation: max motif and benefitScore
    def _summarize_allele(self, allele_pos_df):
        # max motif
        allele_df = allele_pos_df.loc[allele_pos_df.fillna(0).groupby('allele')['freq'].idxmax()]
        allele_df = allele_df.set_index('allele')

        # allele scaling score
        allele_scores = allele_pos_df.groupby('allele')['scaling'].mean().to_dict() # mean
        allele_df['benefitScore'] = allele_scores

        return allele_df
    

# emb_df: index is a.a. list; columns = [emb1, emb2]
def EmbeddingPlot(
    emb_df,
    ax=None,
    fig=None,
    figsize=(4, 3),
    dpi=dpi,
    figfile=None,
):
    # plot df
    plot_df = emb_df.copy()
    plot_df = plot_df.reset_index(names='aa') # reset index
    plot_df['group'] = plot_df['aa'].apply(lambda x: aa_dict[x]) # annotate with a.a. group

    # plot
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
    sns.scatterplot(data=plot_df, x='emb1', y='emb2', hue='group', palette=aa_color_map, ax=ax) # scatter plot

    # annotation
    texts = list()
    for i, row in plot_df.iterrows(): # annotate plot with a.a.
            texts.append(ax.text(row['emb1'] + 0.02, row['emb2'] + 0.02, row['aa'], fontsize=7))
    adjust_text(texts, ax=ax)

    # save
    if fig is not None:
        fig.tight_layout()
        if figfile:
            fig.savefig(figfile)


################
# benchmarking #
################
# benchmarking object for aggregation
class Benchmarking():
    # mut_df: mutation-level
    # pred_df: peptide-level; have to do aggregation
    def __init__(self, mut_df, pred_df, pred_mhc_col='MHC', pred_mt_rank_col='MT_rank', pred_wt_rank_col='WT_rank'):
        self.mut_df = mut_df
        self.pred_df = pred_df
        self.pred_mhc_col = pred_mhc_col
        self.pred_mt_rank_col = pred_mt_rank_col
        self.pred_wt_rank_col = pred_wt_rank_col

    
    # aggregate scores in pred_df for each mutation in mut_df
    # method: binding maximum
    def Aggregation(self, index_cols, metric_cols, mhci_thrs=2, mhcii_thrs=10):
        mut_df = self.mut_df.copy()
        mut_df = mut_df.set_index(index_cols) # set index
        mhc_thrs_dict = {'I': mhci_thrs, 'II': mhcii_thrs} # mhc binding thresholds
        for mhc, thrs in mhc_thrs_dict.items(): # for each mhc
            tmp_df = self.pred_df[self.pred_df[self.pred_mhc_col]==mhc] # filtered by mhc
            tmp_df = tmp_df[tmp_df[self.pred_mt_rank_col]<=thrs] # filtered by binding
            for col in metric_cols: # for each metric
                scores = tmp_df.groupby(index_cols)[col].max() # max score
                mut_df[f'{col}-{mhc}'] = scores
        mut_df = mut_df.reset_index()
        return mut_df


    # identify best index in pred_df for each mutation in mut_df
    # method: binding maximum
    def BestIndex(self, index_cols, metric_col, mhci_thrs=2, mhcii_thrs=10):
        best_idx_dict = dict()
        mhc_thrs_dict = {'I': mhci_thrs, 'II': mhcii_thrs} # mhc binding thresholds
        for mhc, thrs in mhc_thrs_dict.items(): # for each mhc
            tmp_df = self.pred_df[self.pred_df[self.pred_mhc_col]==mhc] # filtered by mhc
            tmp_df = tmp_df[tmp_df[self.pred_mt_rank_col]<=thrs] # filtered by binding
            best_idx = tmp_df.groupby(index_cols)[metric_col].idxmax().values # index of the binding maximum
            best_idx_dict[mhc] = best_idx
        return best_idx_dict
    

    def AggregationByIndex(self, index_cols, metric_cols, idx_dict):
        mut_df = self.mut_df.copy()
        mut_df = mut_df.set_index(index_cols) # set index
        for mhc in ['I', 'II']: # for each mhc
            tmp_df = self.pred_df.loc[idx_dict[mhc]] # filtered by index
            tmp_df = tmp_df.reset_index(names='index').set_index(index_cols) # set index
            mut_df['pred_df_index'] = tmp_df['index']
            for col in metric_cols: # for each metric
                mut_df[f'{col}-{mhc}'] = tmp_df[col] # assign scores
        mut_df = mut_df.reset_index()
        return mut_df


# bootstrapping for evaluating model performance
# use Performance function for evaluating models
def Bootstrapping(df, x_col_dict, y_col, n_iter=1000, fillna=False, random_seed=42):
    n_sample = df.shape[0] # sample size

    # for each bootstrap
    results = list()
    for i in range(n_iter):
        seed = random_seed + i if random_seed is not None else None
        bstp_df = df.sample(n=n_sample, replace=True, random_state=seed) # bootstrapping
        perf_df = Performance(bstp_df, x_col_dict, y_col, fillna=fillna) # performacne
        perf_df = perf_df.reset_index(names=['Model']) # reset index (rename the model column name as Model)
        perf_df['exp'] = i+1 # add column of exp. index
        results.append(perf_df) # append
    
    # concat
    results = pd.concat(results, axis=0, ignore_index=True)
    
    return results


def BootstrappingPN(df, pred_col_dict, label_col, p_n_ratio=1.0, fillna=True, n_iter=1000, random_seed=42):
    """
    Bootstrapping with stratified sampling based on positive-to-negative ratio.
    
    Parameters:
    -----------
    df : DataFrame
        Input dataframe
    pred_col_dict : dict
        Dictionary of prediction columns
    label_col : str
        Name of the label column
    p_n_ratio : float, default=1.0
        Positive to negative ratio for sampling (e.g., 2.0 means 2:1 positive:negative)
    fillna : bool, default=True
        Whether to fill NA values
    n_iter : int, default=1000
        Number of bootstrap iterations
    """
    # Calculate sample sizes based on ratio
    total_samples = df.shape[0]
    n_positive = int(total_samples * p_n_ratio / (1 + p_n_ratio))
    n_negative = total_samples - n_positive
    
    # Split data by label
    positive_df = df[df[label_col] == 1]  # Assuming 1 is positive
    negative_df = df[df[label_col] == 0]  # Assuming 0 is negative
    
    # Check if we have enough samples
    if positive_df.shape[0] == 0 or negative_df.shape[0] == 0:
        raise ValueError("Dataset must contain both positive and negative labels")
    
    # Bootstrap iterations
    results = list()
    for i in range(n_iter):
        seed = random_seed + i if random_seed is not None else None

        # Sample with replacement from each class
        bstp_positive = positive_df.sample(n=n_positive, replace=True, random_state=seed)
        bstp_negative = negative_df.sample(n=n_negative, replace=True, random_state=seed)
        
        # Combine samples
        bstp_df = pd.concat([bstp_positive, bstp_negative], axis=0, ignore_index=True)
        
        # Shuffle the combined dataframe
        bstp_df = bstp_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        
        # Calculate performance
        perf_df = Performance(bstp_df, pred_col_dict, label_col, fillna=fillna)
        perf_df = perf_df.reset_index(names=['Model'])
        perf_df['exp'] = i + 1
        results.append(perf_df)
    
    # Concatenate results
    results = pd.concat(results, axis=0, ignore_index=True)
    
    return results


# evaluating model performance
# x_col_dict = {model_col_name: greater/less, ...}; greater / less for direction
# if fillna, fill NA with the most negative value (0 for greater; 100 for less) 
def Performance(df, x_col_dict, y_col, ppv_k_list=None, fillna=False):
    # fill NA
    if fillna:
        greater_cols = [k for k,v in x_col_dict.items() if v=='greater']
        less_cols = [k for k,v in x_col_dict.items() if v=='less']
        df[greater_cols] = df[greater_cols].fillna(0) # fill 0 for greater cols
        df[less_cols] = df[less_cols].fillna(100) # fill 100 for less cols, such as PHBR, PRIME

    # ppv k list
    if ppv_k_list is None:
        ppv_k_list = [10, 15, 20]
    else:
        ppv_k_list = list(ppv_k_list)
    ppv_k_list.append(df[y_col].sum())
    ppv_k_list = list(sorted(set(ppv_k_list)))

    # performance
    result_df = dict()
    for col, comp in x_col_dict.items():
        tmp_df = df.dropna(subset=[col]) # drop NA
        na_ratio = 1 - tmp_df.shape[0] / df.shape[0] # NA ratio
        pval = NonParamTest(tmp_df, col, y_col, alternative=comp) # p-value
        y, pred = tmp_df[y_col], tmp_df[col]
        if comp == 'less':
            pred = -pred
        auroc = roc_auc_score(y, pred)
        auprc = average_precision_score(y, pred)
        ppv_dict = dict()
        for k in ppv_k_list:
            ppv_dict[f'PPV@{k}'] = precision_at_k(y, pred, k)
        result_df[col] = {
            '%NA': na_ratio,
            'P-val': pval,
            '-logP': -np.log10(pval),
            'AUROC': auroc,
            'AUPRC': auprc,
            **ppv_dict
        }

    return pd.DataFrame(result_df).T
    

def precision_at_k(y_true, y_score, k):
    """
    Calculate precision at top K predictions.
    
    Parameters:
    -----------
    y_true : array-like
        True binary labels (0/1)
    y_score : array-like
        Prediction scores (higher = more likely positive)
    k : int
        Number of top predictions to consider
    
    Returns:
    --------
    float
        Precision at K
    """
    # Sort by score (descending) and get top K indices
    top_k_idx = np.argsort(y_score)[::-1][:k]
    
    # Get true labels for top K predictions
    y_true_top_k = np.array(y_true)[top_k_idx]
    
    # Calculate precision (all predictions are "positive" in top K)
    return np.mean(y_true_top_k)


# barplot for the performance of MHC-I or MHC-II
# performacne result is derived from the function Performance()
def PerfBarPlot(perf_df, methods, metric, figfile=None, dpi=dpi, figsize=(4, 3)):
    # plot df
    plot_df = perf_df.reset_index(names='method')
    val_vars = [col for col in plot_df.columns if col not in ['%NA', 'MHC', 'method']]
    plot_df = plot_df.melt(id_vars=['method'], value_vars=val_vars, var_name='metric')
    plot_df['method'] = plot_df['method'].apply(lambda x: '-'.join(x.split('-')[:-1]))
    
    # filtering
    plot_df = plot_df[plot_df['method'].isin(methods)]
    plot_df = plot_df[plot_df['metric']==metric]

    # plot
    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
    sns.barplot(data=plot_df, y='method', x='value', color=sns.color_palette("Set2")[0], ax=ax)
    for container in ax.containers:  # Iterate over all bar containers
        ax.bar_label(container, fmt='%.3f', fontsize=8, label_type='center')  # Add labels for each container
    ax.set_xlabel(metric)
    ax.set_ylabel('')
    fig.tight_layout()
    if figfile:
        fig.savefig(figfile)


# barplot for the performance of MHC-I and MHC-II
# performacne result is derived from the function Performance()
def TwoPerfBarPlot(mhci_df, mhcii_df, methods, metric, annot=False, palette='pastel', hue_order=None, ax=None, figfile=None, dpi=dpi, figsize=(4, 3)):
    # add MHC
    mhci_df['MHC'] = 'I'
    mhcii_df['MHC'] = 'II'
    
    # concat df
    mhci_df = mhci_df.reset_index(names='method')
    mhcii_df = mhcii_df.reset_index(names='method')
    df = pd.concat([mhci_df.reset_index(), mhcii_df.reset_index()], axis=0, ignore_index=True)
    
    # plot df
    val_vars = [col for col in df.columns if col not in ['%NA', 'MHC', 'method']]
    plot_df = df.melt(id_vars=['MHC', 'method'], value_vars=val_vars, var_name='metric')
    plot_df['method'] = plot_df['method'].apply(lambda x: '-'.join(x.split('-')[:-1]))
    
    # filtering
    plot_df = plot_df[plot_df['method'].isin(methods)]
    plot_df = plot_df[plot_df['metric']==metric]

    # plot
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
    else:
        fig = None
    sns.barplot(data=plot_df, x='MHC', y='value', hue='method', hue_order=hue_order, palette=palette, ax=ax)

    # annot
    if annot:
        n = len(plot_df['method'].unique())
        for i in range(n):
            ax.bar_label(ax.containers[i], fmt='%.3f', fontsize=8, label_type='center')

    ax.set_ylabel(metric)
    ax.legend(loc='lower left', bbox_to_anchor=(1, 0))

    if fig is not None:
        fig.tight_layout()
        if figfile:
            fig.savefig(figfile)


###############
# NCI Dataset #
###############

def FilterNCI(mut_df, mhc, x_cols, y_col, dropna=False):
    """Filter NCI dataframe."""
    # x_cols
    x_col_dict = {f'{col}-{mhc}': ('less' if col in ['PHBR', 'PRIME', 'ICERFIRE'] else 'greater') for col in x_cols}
    x_col_dict = {k:v for k,v in x_col_dict.items() if k in mut_df.columns}

    # filtering
    filter_mut_df = mut_df[(mut_df['DNA_AF']>0) & (mut_df['RNA_AF']>0) & (mut_df['RNA_EXP_QRT']>1)] # filter by expression
    filter_mut_df = filter_mut_df[filter_mut_df['Consequence']=='missense_variant'] # focus on substitutions
    filter_mut_df = filter_mut_df.dropna(subset=[f'PHBR-{mhc}']) # drop invalid mutations
    filter_mut_df = filter_mut_df[filter_mut_df[f'Robustness-{mhc}']>0]
    if dropna:
        cols = list(x_col_dict.keys()) + [y_col,]
        filter_mut_df = filter_mut_df.dropna(subset=cols) # drop NA
    
    return filter_mut_df
