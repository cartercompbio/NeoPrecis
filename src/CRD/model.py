#!/bin/python3
# Description: peptide-based model
# Author: Kohan

import h5py
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics.classification import BinaryAUROC, BinaryAveragePrecision
from . import architecture


class PeptideTrainingModule(pl.LightningModule):
    def __init__(
        self,
        ref_h5_file: str,                           # reference h5 file (aa_embs, position_factors, and motifs)
        archt: nn.Module,                           # model architecture
        
        aa_encode_key='aa_blosum_encodes',          # key for aa_encode matrix
        position_factor_key='position_factors',     # key for position_factor matrix
        motif_key='motifs',                         # key for motif_matrix

        aa_vocab_size=21,                           # a.a. vocabulary size
        aa_encode_dim=21,                           # a.a. encoding size
        aa_emb_dim=2,                               # a.a. embedding size
        pept_emb_dim=2,                             # peptide embedding size
        peptide_length=9,                           # peptide length
        pos_emb_method='global',                    # position embedding method ("global", "local", or "both")
        crd_idx=-1,                                 # index of the geometric representation used as CRD
        feature_idx=[],                             # indices of features used for prediction
        task_num=1,                                 # number of binary tasks (only support 1 currently)
        
        lr=1e-2,                                    # learning rate
        lr_scheduler=True,                          # learning rate scheduler
        weight_decay=1e-2,                          # L2 regularization in Adam optimizer
        monitor='valid/loss',                       # monitor for lr_scheduler
        monitor_mode='min'                          # monitor mode (min or max)
    ):
        super().__init__()

        # reference
        with h5py.File(ref_h5_file, 'r') as f:
            self.aa_encode_matrix = torch.tensor(f[aa_encode_key][:])
            self.motifs = torch.tensor(f[motif_key][:])
            self.position_factors = torch.tensor(f[position_factor_key][:])

        # variables
        self.save_hyperparameters()
        self.feature_idx = feature_idx
        self.feature_num = len(feature_idx)
        self.lr = lr
        self.lr_scheduler = lr_scheduler
        self.weight_decay = weight_decay
        self.monitor = monitor
        self.monitor_mode = monitor_mode

        # model
        self.model = getattr(architecture, archt)
        self.model = self.model(self.aa_encode_matrix, aa_vocab_size, aa_encode_dim, aa_emb_dim, pept_emb_dim, peptide_length, pos_emb_method, crd_idx, self.feature_num, task_num)
        
    
    # input peptide (N, 2, L): 2 for MT and WT, L=peptide_length
    # input feature (N, K): K features, where K>=2 for (MHCI/II, allele_ID, features)
    # output geo_vectors (N, geo_dim) & preds (N, task_num); geo_dim = 2*aa_emb_dim + 3
    # geo_dim = (origin(aa_emb_dim), direction(aa_emb_dim), length(1), scaler(1), scaled_length(1))
    def forward(self, peptides: torch.tensor, features: torch.tensor):
        mhcs = features[:, 0]
        allele_idxs = features[:, 1].cpu().numpy()
        motifs = self.motifs[allele_idxs].to(self.device)
        position_factors = self.position_factors[allele_idxs].to(self.device)
        features = features[:, self.feature_idx]
        geo_vectors, preds = self.model(peptides.long(), mhcs.long(), motifs.float(), position_factors.float(), features.float())
        return geo_vectors, preds


    def configure_optimizers(self):
        d = dict()
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        d['optimizer'] = optimizer

        if self.lr_scheduler:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=10, threshold=1e-4, min_lr=1e-4, mode=self.monitor_mode)
            d['lr_scheduler'] = {'scheduler': scheduler, 'monitor': self.monitor}

        return d
    

# task: cross-reactivity distance
# triplet contrastive learning
class CRD(PeptideTrainingModule):
    def __init__(
        self,
        triplet_margin=1,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.save_hyperparameters()
        self.triplet_margin = triplet_margin

    def training_step(self, batch: list, batch_idx: int):
        peptides, features, ys = batch
        distances = self._calculate_CRD(peptides, features)
        loss = self._compute_loss(distances)
        self.log('train/CRD/loss', loss, on_step=False, on_epoch=True)
        return loss
    
    def validation_step(self, batch: list, batch_idx: int):
        peptides, features, ys = batch
        distances = self._calculate_CRD(peptides, features)
        loss = self._compute_loss(distances)
        self.log('valid/CRD/loss', loss, on_step=False, on_epoch=True)
    
    def test_step(self, batch: list, batch_idx: int):
        peptides, features, ys = batch
        distances = self._calculate_CRD(peptides, features)
        loss = self._compute_loss(distances)
        self.log('test/CRD/loss', loss, on_step=False, on_epoch=True)

    def predict_step(self, batch: list, batch_idx: int) -> torch.tensor:
        peptides, features, ys = batch
        distances = self._calculate_CRD(peptides, features)
        #loss = self._compute_loss(distances)
        return distances
    
    def _calculate_CRD(self, peptides, features):
        pos_peptides = peptides[:, [1, 0], :]   # positive pairs (positive, anchor)
        neg_peptides = peptides[:, [2, 0], :]   # negative pairs (negative, anchor)
        pos_geo_vectors, _ = self.forward(pos_peptides, features)  # (N, geo_dim)
        neg_geo_vectors, _ = self.forward(neg_peptides, features)  # (N, geo_dim)
        pos_distances = pos_geo_vectors[:, self.model.crd_idx].unsqueeze(-1)    # (N, 1)
        neg_distances = neg_geo_vectors[:, self.model.crd_idx].unsqueeze(-1)    # (N, 1)
        distances = torch.cat((pos_distances, neg_distances), dim=1)    # (N, 2)
        return distances
    
    def _compute_loss(self, distances):
        pos_distances = distances[:, 0]
        neg_distances = distances[:, 1]
        loss = torch.fmax(pos_distances - neg_distances + self.triplet_margin, torch.tensor(0))
        return torch.mean(loss)
    

# task: binary classification (like immunogenicity)
# not available for #tasks > 1
class CLF(PeptideTrainingModule):
    def __init__(
        self,
        loss_weights=[0.5, 0.5],
        **kwargs
    ):
        super().__init__(**kwargs)

        self.save_hyperparameters()
        self.loss_weights = torch.tensor(loss_weights)

        # metrics
        self.train_auroc = BinaryAUROC()
        self.valid_auroc = BinaryAUROC()
        self.test_auroc = BinaryAUROC()
        self.train_auprc = BinaryAveragePrecision()
        self.valid_auprc = BinaryAveragePrecision()
        self.test_auprc = BinaryAveragePrecision()


    def training_step(self, batch: list, batch_idx: int):
        peptides, features, ys = batch
        geo_vectors, preds = self.forward(peptides, features)
        loss = self._compute_loss(preds, ys)

        # metrics
        self.train_auroc(preds.float(), ys.int())
        self.train_auprc(preds.float(), ys.int())

        # log
        self.log('train/CLF/loss', loss, on_step=False, on_epoch=True)
        self.log('train/CLF/AUROC', self.train_auroc, on_step=False, on_epoch=True)
        self.log('train/CLF/AUPRC', self.train_auprc, on_step=False, on_epoch=True)

        return loss
    

    def validation_step(self, batch: list, batch_idx: int):
        peptides, features, ys = batch
        geo_vectors, preds = self.forward(peptides, features)
        loss = self._compute_loss(preds, ys)

        # metrics
        self.valid_auroc(preds.float(), ys.int())
        self.valid_auprc(preds.float(), ys.int())

        # log
        self.log('valid/CLF/loss', loss, on_step=False, on_epoch=True)
        self.log('valid/CLF/AUROC', self.valid_auroc, on_step=False, on_epoch=True)
        self.log('valid/CLF/AUPRC', self.valid_auprc, on_step=False, on_epoch=True)

    
    def test_step(self, batch: list, batch_idx: int):
        peptides, features, ys = batch
        geo_vectors, preds = self.forward(peptides, features)
        loss = self._compute_loss(preds, ys)

        # metrics
        self.test_auroc(preds.float(), ys.int())
        self.test_auprc(preds.float(), ys.int())

        # log
        self.log('test/CLF/loss', loss, on_step=False, on_epoch=True)
        self.log('test/CLF/AUROC', self.test_auroc, on_step=False, on_epoch=True)
        self.log('test/CLF/AUPRC', self.test_auprc, on_step=False, on_epoch=True)


    def predict_step(self, batch: list, batch_idx: int) -> torch.tensor:
        peptides, features, ys = batch
        geo_vectors, preds = self.forward(peptides, features)
        output = torch.cat((geo_vectors, preds, ys), dim=1)    # (N, geo_dim + 1 + 1)
        return output


    def _compute_loss(self, preds, ys):
        # loss weights
        loss_weights = self.loss_weights.to(ys.device)
        loss_weights = loss_weights[ys.long()]
        # loss
        loss_func = nn.BCELoss(weight=loss_weights)
        loss = loss_func(preds.float(), ys.float())
        return loss
    

class MultiTasks(PeptideTrainingModule):
    def __init__(
        self,
        triplet_margin=1,
        loss_weights=[0.5, 0.5],
        **kwargs
    ):
        super().__init__(**kwargs)

        self.save_hyperparameters()
        self.triplet_margin = triplet_margin
        self.loss_weights = torch.tensor(loss_weights)

        # metrics
        self.train_auroc = BinaryAUROC()
        self.valid_auroc = BinaryAUROC()
        self.test_auroc = BinaryAUROC()
        self.train_auprc = BinaryAveragePrecision()
        self.valid_auprc = BinaryAveragePrecision()
        self.test_auprc = BinaryAveragePrecision()


    def training_step(self, batch: list, batch_idx: int):
        batch, batch_idx, dataloader_idx = batch

        # CRD task
        if dataloader_idx==0:
            peptides, features, ys = batch
            distances = self._calculate_CRD(peptides, features)                     # 3 peptides: anchor, positive, negative
            loss = self._compute_CRD_loss(distances)
            self.log('train/CRD/loss', loss, on_step=False, on_epoch=True)
        
        # CLF task
        elif dataloader_idx==1:
            peptides, features, ys = batch
            geo_vectors, preds = self.forward(peptides, features)                   # 2 peptides: MT and WT
            loss = self._compute_CLF_loss(preds, ys)
            self.log('train/CLF/loss', loss, on_step=False, on_epoch=True)
            self.train_auroc(preds.float(), ys.int())
            self.train_auprc(preds.float(), ys.int())
            self.log('train/CLF/AUROC', self.train_auroc, on_step=False, on_epoch=True)
            self.log('train/CLF/AUPRC', self.train_auprc, on_step=False, on_epoch=True)
        
        return loss
    

    def validation_step(self, batch: list, batch_idx: int):
        batch, batch_idx, dataloader_idx = batch

        # CRD task
        if dataloader_idx==0:
            peptides, features, ys = batch
            distances = self._calculate_CRD(peptides, features)                     # 3 peptides: anchor, positive, negative
            crd_loss = self._compute_CRD_loss(distances)
            self.log('valid/CRD/loss', crd_loss, on_step=False, on_epoch=True)
        
        # CLF task
        elif dataloader_idx==1:
            peptides, features, ys = batch
            geo_vectors, preds = self.forward(peptides, features)                   # 2 peptides: MT and WT
            clf_loss = self._compute_CLF_loss(preds, ys)
            self.log('valid/CLF/loss', clf_loss, on_step=False, on_epoch=True)
            self.valid_auroc(preds.float(), ys.int())
            self.valid_auprc(preds.float(), ys.int())
            self.log('valid/CLF/AUROC', self.valid_auroc, on_step=False, on_epoch=True)
            self.log('valid/CLF/AUPRC', self.valid_auprc, on_step=False, on_epoch=True)

    
    def test_step(self, batch: list, batch_idx: int):
        batch, batch_idx, dataloader_idx = batch

        # CRD task
        if dataloader_idx==0:
            peptides, features, ys = batch
            distances = self._calculate_CRD(peptides, features)                     # 3 peptides: anchor, positive, negative
            crd_loss = self._compute_CRD_loss(distances)
            self.log('test/CRD/loss', crd_loss, on_step=False, on_epoch=True)

        # CLF task
        elif dataloader_idx==1:
            peptides, features, ys = batch
            geo_vectors, preds = self.forward(peptides, features)                   # 2 peptides: MT and WT
            clf_loss = self._compute_CLF_loss(preds, ys)
            self.log('test/CLF/loss', clf_loss, on_step=False, on_epoch=True)
            self.test_auroc(preds.float(), ys.int())
            self.test_auprc(preds.float(), ys.int())
            self.log('test/CLF/AUROC', self.test_auroc, on_step=False, on_epoch=True)
            self.log('test/CLF/AUPRC', self.test_auprc, on_step=False, on_epoch=True)


    def predict_step(self, batch: list, batch_idx: int) -> torch.tensor:
        peptides, features, ys = batch
        geo_vectors, preds = self.forward(peptides, features)
        output = torch.cat((geo_vectors, preds, ys), dim=1)     # (N, geo_dim + 1 + 1)
        return output
    

    def _calculate_CRD(self, peptides, features):
        pos_peptides = peptides[:, [1, 0], :]   # positive pairs (positive, anchor)
        neg_peptides = peptides[:, [2, 0], :]   # negative pairs (negative, anchor)
        pos_geo_vectors, _ = self.forward(pos_peptides, features)  # (N, geo_dim)
        neg_geo_vectors, _ = self.forward(neg_peptides, features)  # (N, geo_dim)
        pos_distances = pos_geo_vectors[:, self.model.crd_idx].unsqueeze(-1)    # (N, 1)
        neg_distances = neg_geo_vectors[:, self.model.crd_idx].unsqueeze(-1)    # (N, 1)
        distances = torch.cat((pos_distances, neg_distances), dim=1)    # (N, 2)
        return distances


    def _compute_CRD_loss(self, distances):
        pos_distances = distances[:, 0]
        neg_distances = distances[:, 1]
        loss = torch.fmax(pos_distances - neg_distances + self.triplet_margin, torch.tensor(0))
        return torch.mean(loss)
    

    def _compute_CLF_loss(self, preds, ys):
        # loss weights
        loss_weights = self.loss_weights.to(ys.device)
        loss_weights = loss_weights[ys.long()]
        # loss
        loss_func = nn.BCELoss(weight=loss_weights)
        loss = loss_func(preds.float(), ys.float())
        return loss