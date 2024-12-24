#!/bin/python3
# Description: Model architectures
# Author: Kohan

import torch
import torch.nn as nn
import torch.nn.functional as F


#########################################
### Parent model (substitution-based) ###
#########################################

# geometric vector = (origin(size=pept_emb_dim), direction(pept_emb_dim), length(1), scaler(1), scaled_length(1))
class PeptideDiffModel(nn.Module):
    def __init__(
            self,
            aa_encode,                  # pre-definied a.a. encoding matrix; size = (aa_vocab_size, aa_encode_dim)
            aa_vocab_size=21,           # a.a. vocabulary size
            aa_encode_dim=21,           # a.a. encoding size
            aa_emb_dim=2,               # a.a. embedding size
            pept_emb_dim=2,             # peptide embedding size
            peptide_length=9,           # peptide length
            pos_emb_method='global',    # position embedding method ("global", "local", or "both")
            crd_idx=-1,                 # index of the geometric representation used as CRD
            feature_num=0,              # number of features except for CRD used for prediction
            task_num=1,                 # number of binary tasks (default = 1 for immunogenicity)
    ):
        super().__init__()

        # variables
        self.aa_encode = aa_encode
        self.aa_vocab_size = aa_vocab_size
        self.aa_encode_dim = aa_encode_dim
        self.aa_emb_dim = aa_emb_dim
        self.pept_emb_dim = pept_emb_dim
        self.peptide_length = peptide_length
        self.pos_emb_method = pos_emb_method
        self.crd_idx = crd_idx
        self.feature_num = feature_num
        self.task_num = task_num
        self.eps = 1e-12                        # epsilon to avoid divided by zero

        # aa encoding
        self.aa_encode_layer = nn.Parameter(self.aa_encode.float(), requires_grad=False)    # (aa_vocab_size, aa_encode_dim)

        # aa embedding
        self.aa_emb_layer = nn.Linear(self.aa_encode_dim, self.aa_emb_dim)                  # (aa_encode_dim, aa_emb_dim)

        # global position factors
        self.mhci_glb_pos_fac = nn.Parameter(torch.ones((self.peptide_length,), dtype=torch.float32))    # (peptide_length,)
        self.mhcii_glb_pos_fac = nn.Parameter(torch.ones((self.peptide_length,), dtype=torch.float32))   # (peptide_length,)

        # length scaler
        self.length_scaler_layer = nn.Sequential(
            nn.Linear(2*self.pept_emb_dim, self.pept_emb_dim),
            nn.LeakyReLU(),
            nn.Linear(self.pept_emb_dim, 1)
        )

        # classifier
        input_dim = 1 + self.feature_num
        self.clf = nn.Sequential(
            nn.Linear(input_dim, self.task_num),
            nn.Sigmoid()
        )
        
    # peptides (N, 2, L):  2 for MT (idx=0) and WT (idx=1), L=peptide_length
    # mhcs (N,): MHC-I=0, MHC-II=1
    # motifs (N, L, T): L=peptide_length, T=aa_vocab_size
    # pos_facs (N, L): derived from motifs (entropy), L=peptide_length
    # features (N, feature_num): features such as MHC-binding score
    # output1: geometric_vectors (N, 2*pept_emb_dim+3); (origin(pept_emb_dim), direction(pept_emb_dim), length(1), scaler(1), scaled_length(1))
    # output2: preds (N, task_num)
    def forward(self, peptides, mhcs, motifs, pos_facs, features):
        # peptides
        mt_peptides = peptides[:,0,:]                                                   # MT peptides (N, L)
        wt_peptides = peptides[:,1,:]                                                   # WT peptides (N, L)

        # a.a. embedding
        mt_embs = self._aa_embedding(mt_peptides)                                       # (N, L, aa_emb_dim)
        wt_embs = self._aa_embedding(wt_peptides)                                       # (N, L, aa_emb_dim)

        # position embedding
        pos_embs = self._pos_embedding(mhcs, pos_facs, method=self.pos_emb_method)      # (N, L)

        # peptide embedding
        mt_pept_embs = self._peptide_pooling(mt_embs, pos_embs)                         # (N, pept_emb_dim)
        wt_pept_embs = self._peptide_pooling(wt_embs, pos_embs)                         # (N, pept_emb_dim)

        # geometric transformation
        geos, preds = self._geometric_tf(wt_pept_embs, mt_pept_embs, features)          # (N, 2*pept_emb_dim+3); (N, task_num)

        return geos, preds

    # a.a. -> one-hot -> encode -> embed
    # return (N, L, aa_emb_dim)
    def _aa_embedding(self, seqs, OHE=True):
        if OHE:
            seqs = F.one_hot(seqs, num_classes=self.aa_vocab_size).float()  # one-hot encoding; (N, L, aa_vocab_size)
        encodes = torch.matmul(seqs, self.aa_encode_layer)                  # a.a. encoding; (N, L, aa_encode_dim)
        embs = self.aa_emb_layer(encodes)                                   # a.a. embedding; (N, L, aa_emb_dim)
        return embs
    
    # position embedding (method = "global", "local", and "both")
    # global PF: all MHC-I/MHC-II alleles share the same PFs, respectively
    # local PF: allele-specific PFs
    # return (N, L)
    def _pos_embedding(self, mhcs, pos_facs, method='global'):
        if method == 'local':
            return pos_facs
        pos_embs = torch.zeros((len(mhcs), self.peptide_length)).to(mhcs.device)    # empty matirx
        pos_embs[mhcs==0, :] = self.mhci_glb_pos_fac                                # add MHC-I
        pos_embs[mhcs==1, :] = self.mhcii_glb_pos_fac                               # add MHC-II
        if method == 'global':
            return pos_embs.abs()                                                   # absolute values
        else: # method == 'both'
            return pos_embs.abs() * pos_facs                                        # multiply with local pos. fac.

    # weighted average across residues
    # return (N, pept_emb_dim)
    def _peptide_pooling(self, seq_embs, pos_embs):
        weights = pos_embs + self.eps                                       # (N, L)
        weights = weights / weights.sum(dim=-1).unsqueeze(-1)               # normalize; (N, L)
        weights = weights.unsqueeze(-1).expand(-1, -1, self.pept_emb_dim)   # expand; (N, L, pept_emb_dim)
        pept_embs = (seq_embs * weights).sum(dim=1)                         # weighted sum; (N, pept_emb_dim)
        return pept_embs
    
    # compute geometric representation and clf prediction
    # geo = (origin, direction, length, scaler, scaled_length)
    def _geometric_tf(self, wt_pept_embs, mt_pept_embs, features):
        # geos = (origin, direction, length)
        origin = wt_pept_embs                                                           # (N, pept_emb_dim)
        direction = F.normalize(mt_pept_embs - wt_pept_embs, p=2, dim=1, eps=self.eps)  # (N, pept_emb_dim)
        length = (mt_pept_embs - wt_pept_embs).pow(2).sum(-1).sqrt().unsqueeze(-1)      # (N, 1)
        geos = torch.cat((origin, direction, length), dim=1)                            # (N, 2*pept_emb_dim+1)

        # scaled length
        scaler = self.length_scaler_layer(geos[:, :-1])                                 # (N, 1)
        scaled_length = length * F.sigmoid(scaler)                                      # (N, 1)

        # classification
        if self.feature_num > 0:
            Xs = torch.cat((scaled_length, features), dim=-1)                           # (N, 1+feature_num)
        else:
            Xs = scaled_length                                                          # (N, 1)
        preds = self.clf(Xs)                                                            # (N, task_num)

        # output = (origin, direction, length, scaler, scaled_length)
        outputs = torch.cat((geos, scaler, scaled_length), dim=1)                       # (N, 2*pept_emb_dim+3)

        return outputs, preds


#########################
# Motif attention model #
#########################

class MotifAttnModel(PeptideDiffModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # motif attention
        self.attn_layer = nn.MultiheadAttention(self.aa_emb_dim, 1, batch_first=True)   # W_Q = W_K = W_V = (aa_emb_dim, aa_emb_dim)

    def forward(self, peptides, mhcs, motifs, pos_facs, features):
        # peptides
        mt_peptides = peptides[:,0,:]                                                   # MT peptides (N, L)
        wt_peptides = peptides[:,1,:]                                                   # WT peptides (N, L)

        # a.a. embedding
        mt_embs = self._aa_embedding(mt_peptides)                                       # (N, L, aa_emb_dim)
        wt_embs = self._aa_embedding(wt_peptides)                                       # (N, L, aa_emb_dim)

        # motif embedding
        motif_embs = self._aa_embedding(motifs, OHE=False)                              # (N, L, aa_emb_dim)

        # attention
        mt_attns, _ = self.attn_layer(motif_embs, mt_embs, mt_embs)                     # (N, L, aa_emb_dim)
        wt_attns, _ = self.attn_layer(motif_embs, wt_embs, wt_embs)                     # (N, L, aa_emb_dim)

        # position embedding
        pos_embs = self._pos_embedding(mhcs, pos_facs, method=self.pos_emb_method)      # (N, L)

        # peptide embedding (weighted average)
        mt_pept_embs = self._peptide_pooling(mt_attns, pos_embs)                        # (N, pept_emb_dim)
        wt_pept_embs = self._peptide_pooling(wt_attns, pos_embs)                        # (N, pept_emb_dim)

        # geometric transformation
        geos, preds = self._geometric_tf(wt_pept_embs, mt_pept_embs, features)          # (N, 2*pept_emb_dim+3); (N, task_num)

        return geos, preds


##########################
# Motif enrichment model #
##########################
class MotifEnrModel(PeptideDiffModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # aa_enrich_embedding
        self.enr_emb_layer = nn.Linear(self.aa_emb_dim+1, self.aa_emb_dim)              # (aa_emb_dim+1, aa_emb_dim)

    def forward(self, peptides, mhcs, motifs, pos_facs, features):
        # peptides
        mt_peptides = peptides[:,0,:]                                                   # MT peptides (N, L)
        wt_peptides = peptides[:,1,:]                                                   # WT peptides (N, L)

        # a.a. embedding
        mt_embs = self._aa_embedding(mt_peptides)                                       # (N, L, aa_emb_dim)
        wt_embs = self._aa_embedding(wt_peptides)                                       # (N, L, aa_emb_dim)

        # motif embedding
        motif_embs = self._aa_embedding(motifs, OHE=False)                              # (N, L, aa_emb_dim)

        # enrichment
        mt_enrichs = self._motif_enrichment(mt_embs, motif_embs)                        # (N, L, 1)
        wt_enrichs = self._motif_enrichment(wt_embs, motif_embs)                        # (N, L, 1)

        # concat
        mt_embs = torch.cat((mt_embs, mt_enrichs), dim=-1)                              # (N, L, aa_emb_dim+1)
        wt_embs = torch.cat((wt_embs, wt_enrichs), dim=-1)                              # (N, L, aa_emb_dim+1)

        # a.a. embedding
        mt_embs = self.enr_emb_layer(mt_embs)                                           # (N, L, aa_emb_dim)
        wt_embs = self.enr_emb_layer(wt_embs)                                           # (N, L, aa_emb_dim)

        # position embedding
        pos_embs = self._pos_embedding(mhcs, pos_facs, method=self.pos_emb_method)      # (N, L)

        # peptide embedding (weighted average)
        mt_pept_embs = self._peptide_pooling(mt_embs, pos_embs)                         # (N, pept_emb_dim)
        wt_pept_embs = self._peptide_pooling(wt_embs, pos_embs)                         # (N, pept_emb_dim)

        # geometric transformation
        geos, preds = self._geometric_tf(wt_pept_embs, mt_pept_embs, features)          # (N, 2*pept_emb_dim+3); (N, task_num)

        return geos, preds

    # motif enrichment; return (N, L, 1)
    def _motif_enrichment(self, seq_embs, motif_embs):
        enrichs = (seq_embs * motif_embs).sum(dim=-1).unsqueeze(-1) # (N, L, 1)
        return enrichs


#########################################
# Motif convolution model w/ projection #
#########################################
class MotifConvProjModel(PeptideDiffModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_kernels = 8
        self.conv_layer = nn.Parameter(torch.randn(self.num_kernels, self.aa_emb_dim, self.aa_emb_dim, dtype=torch.float32))
        self.proj_layer = nn.Linear(self.num_kernels, self.pept_emb_dim)

    def forward(self, peptides, mhcs, motifs, pos_facs, features):
        n = peptides.shape[0]

        # peptides
        mt_peptides = peptides[:,0,:]                                                   # MT peptides (N, L)
        wt_peptides = peptides[:,1,:]                                                   # WT peptides (N, L)

        # a.a. embedding
        mt_embs = self._aa_embedding(mt_peptides)                                       # (N, L, aa_emb_dim)
        wt_embs = self._aa_embedding(wt_peptides)                                       # (N, L, aa_emb_dim)

        # motif embedding
        motif_embs = self._aa_embedding(motifs, OHE=False)                              # (N, L, aa_emb_dim)

        # motif-dependent conv layer
        conv_layer = self._build_kernels(motif_embs, n)                                 # (N, num_kernels, L, aa_emb_dim)

        # motif enrichment
        mt_embs = self._motif_enrichment(mt_embs, conv_layer)                           # (N, L, pept_emb_dim)
        wt_embs = self._motif_enrichment(wt_embs, conv_layer)                           # (N, L, pept_emb_dim)

        # position embedding
        pos_embs = self._pos_embedding(mhcs, pos_facs, method=self.pos_emb_method)      # (N, L)

        # peptide embedding (weighted average)
        mt_pept_embs = self._peptide_pooling(mt_embs, pos_embs)                         # (N, pept_emb_dim)
        wt_pept_embs = self._peptide_pooling(wt_embs, pos_embs)                         # (N, pept_emb_dim)

        # geometric transformation
        geos, preds = self._geometric_tf(wt_pept_embs, mt_pept_embs, features)          # (N, 2*pept_emb_dim+3); (N, task_num)

        return geos, preds
    
    # build conv. kernels for motif enrichment
    def _build_kernels(self, motif_embs, n):
        conv_layer = self.conv_layer.unsqueeze(0).expand(n, -1, -1, -1)                 # (N, num_kernels, aa_emb_dim, aa_emb_dim)
        motif_embs = motif_embs.unsqueeze(1).expand(-1, self.num_kernels, -1, -1)       # (N, num_kernels, L, aa_emb_dim)
        conv_layer = torch.matmul(motif_embs, conv_layer)                               # (N, num_kernels, L, aa_emb_dim)
        return conv_layer

    # position-wise motif enrichment (w/ projection)
    def _motif_enrichment(self, seq_embs, conv_layer):
        seq_embs = torch.einsum('nijk,njk->nij', conv_layer, seq_embs).transpose(1,2)   # (N, L, num_kernels)
        seq_embs = self.proj_layer(seq_embs)                                            # (N, L, pept_emb_dim)
        return seq_embs


##########################################
# Motif convolution model w/o projection #
##########################################
class MotifConvModel(PeptideDiffModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_kernels = self.pept_emb_dim
        self.conv_layer = nn.Parameter(torch.randn(self.num_kernels, self.aa_emb_dim, self.aa_emb_dim, dtype=torch.float32))
        
    def forward(self, peptides, mhcs, motifs, pos_facs, features):
        n = peptides.shape[0]

        # peptides
        mt_peptides = peptides[:,0,:]                                                   # MT peptides (N, L)
        wt_peptides = peptides[:,1,:]                                                   # WT peptides (N, L)

        # a.a. embedding
        mt_embs = self._aa_embedding(mt_peptides)                                       # (N, L, aa_emb_dim)
        wt_embs = self._aa_embedding(wt_peptides)                                       # (N, L, aa_emb_dim)

        # motif embedding
        motif_embs = self._aa_embedding(motifs, OHE=False)                              # (N, L, aa_emb_dim)

        # motif-dependent conv layer
        conv_layer = self._build_kernels(motif_embs, n)                                 # (N, num_kernels, L, aa_emb_dim)

        # motif enrichment
        mt_embs = self._motif_enrichment(mt_embs, conv_layer)                           # (N, L, num_kernels); num_kernels = pept_emb_dim
        wt_embs = self._motif_enrichment(wt_embs, conv_layer)                           # (N, L, num_kernels)

        # position embedding
        pos_embs = self._pos_embedding(mhcs, pos_facs, method=self.pos_emb_method)      # (N, L)

        # peptide embedding (weighted average)
        mt_pept_embs = self._peptide_pooling(mt_embs, pos_embs)                         # (N, pept_emb_dim)
        wt_pept_embs = self._peptide_pooling(wt_embs, pos_embs)                         # (N, pept_emb_dim)

        # geometric transformation
        geos, preds = self._geometric_tf(wt_pept_embs, mt_pept_embs, features)          # (N, 2*pept_emb_dim+3); (N, task_num)

        return geos, preds
    
    # build conv. kernels for motif enrichment
    def _build_kernels(self, motif_embs, n):
        conv_layer = self.conv_layer.unsqueeze(0).expand(n, -1, -1, -1)                 # (N, num_kernels, aa_emb_dim, aa_emb_dim)
        motif_embs = motif_embs.unsqueeze(1).expand(-1, self.num_kernels, -1, -1)       # (N, num_kernels, L, aa_emb_dim)
        conv_layer = torch.matmul(motif_embs, conv_layer)                               # (N, num_kernels, L, aa_emb_dim)
        return conv_layer

    # position-wise motif enrichment (w/o projection)
    def _motif_enrichment(self, seq_embs, conv_layer):
        seq_embs = torch.einsum('nijk,njk->nij', conv_layer, seq_embs).transpose(1,2)   # (N, L, num_kernels)
        return seq_embs