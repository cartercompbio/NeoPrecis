import os, sys, json, warnings
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from scipy.stats import pearsonr, spearmanr, zscore, mannwhitneyu, ttest_ind
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
from venn import venn
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test
from cliffs_delta import cliffs_delta


##############################
# Immunogenicity Prediction  #
##############################

# preprocessing mutation metrics for training immunogenicity model
# reversing PHBR: 1 - phbr/100
# keeping only missense mutations
# fill NA with 0
def PreprocessingMutations(df, features, missense_name='missense_variant'):
    # PHBR
    df['PHBR-I'] = df['PHBR-I'].fillna(100) # fill NA with 100 for PHBR
    df['PHBR-II'] = df['PHBR-II'].fillna(100) # fill NA with 100 for PHBR
    df['PHBR-I'] = 1 - df['PHBR-I']/100 # reverse
    df['PHBR-II'] = 1 - df['PHBR-II']/100 # reverse
    
    # missense mutations
    df = df[df['Consequence']==missense_name]
    
    # fill NA
    df[features] = df[features].fillna(0) # fill NA with 0 for other features
    
    df.reset_index(drop=True, inplace=True)
    return df


# immunogenicity predictor
# input: neoantigen df, feature columns, and label column
# neoantigen df is training data
# optional: normalized features or not
class Predictor():
    def __init__(self, df, x_cols, y_col):
        self.df = df
        self.x_cols = x_cols
        self.y_col = y_col
        self.X = df[self.x_cols].to_numpy()
        self.Y = df[self.y_col].to_numpy()
        self.mean_arr = self.X.mean(axis=0)
        self.std_arr = self.X.std(axis=0)

    def train(self, model, normalized=False):
        self.normalized = normalized # normalization
        self.model = model
        if normalized:
            X = (self.X - self.mean_arr) / self.std_arr
        else:
            X = self.X
        self.model.fit(X, self.Y)
        auroc, auprc = self.evaluation(X, self.Y)
        print(f'Training performance: AUROC={auroc:.2f} / AUPRC={auprc:.2f}')

    def prediction(self, X):
        if self.normalized:
            X = (X - self.mean_arr) / self.std_arr
        return self.model.predict_proba(X)[:,1]

    def evaluation(self, X, Y):
        pred = self.prediction(X)
        auroc = roc_auc_score(Y, pred)
        auprc = average_precision_score(Y, pred)
        return auroc, auprc


# performance of metrics
def MetricPerformance(
    sample_df,          # sample dataframe
    metric_cols,        # columns of metrics to be compared
    label_col,          # label column
    group_col=None,     # compared by groups
    verbose=False,
):
    # filtering
    if verbose: print('#Samples =', sample_df.shape[0])
    df = sample_df.dropna(subset=[label_col]+metric_cols, ignore_index=True)
    if verbose: print('#Samples after dropping NA =', df.shape[0])

    # check groups
    cols = df.columns.tolist() # all columns
    if group_col in cols:
        groups = df[group_col].unique().tolist() # get groups
    
    # main
    result_list = list()
    for col in metric_cols: # for each metric
        if group_col in cols: # with groups
            for group in groups:
                group_df = df[df[group_col]==group] # filter by group
                result = EvaluationMetrics(group_df, col, label_col) # compute evaluation metrics
                result[group_col] = group # add group
                result_list.append(result)
        
        else: # without groups
            result = EvaluationMetrics(sample_df, col, label_col) # compute evaluation metrics
            result_list.append(result)
    
    return pd.DataFrame(result_list)


# compute AUROC and AUPRC
def EvaluationMetrics(df, x_col, y_col):
    if len(df[y_col].unique()) != 2:
        print(f'Only one class in {y_col}')
        return {}
    
    auroc = roc_auc_score(df[y_col], df[x_col])
    auprc = average_precision_score(df[y_col], df[x_col])
    result = {
        'size': df.shape[0],
        'method': x_col,
        'AUROC': auroc,
        'AUPRC': auprc,
    }

    return result


# group performance
# two groups based on split_col
def TwoGroupsPerf(
    df,                         # patient df
    split_col,                 # column used to split
    x_cols,                     # target metrics to compare
    y_col,                      # label column
    cancer_col='cancer',        # cancer column (split for each cancer)
):
    cancers = df[cancer_col].unique().tolist() # cancer types

    # split for each cancer
    perf_dict = dict()
    for cancer in cancers:
        # split
        tmp_df = df[df[cancer_col]==cancer]
        median = tmp_df[split_col].median()
        df_dict = {
            'high': tmp_df[tmp_df[split_col]>median],
            'low': tmp_df[tmp_df[split_col]<=median]
        }

        # performance
        for split_name, split_df in df_dict.items():
            tmp_perf_df = MetricPerformance(split_df, x_cols, y_col)
            tmp_perf_df[cancer_col] = cancer
            tmp_perf_df['group'] = split_name
            perf_dict[f'{cancer}:{split_name}'] = tmp_perf_df
    
    # concat
    perf_df = pd.concat(perf_dict.values(), ignore_index=True)

    return perf_df


# group performance
# four groups based on split_col1 and split_col2
def FourGroupsPerf(
    df,                         # patient df
    split_col1,                 # first column used to split
    split_col2,                 # second column used to split
    x_cols,                     # target metrics to compare
    y_col,                      # label column
    cancer_col='cancer',        # cancer column (split for each cancer)
):  
    cancers = df[cancer_col].unique().tolist() # cancer types

    # split for each cancer
    perf_dict = dict()
    for cancer in cancers:
        # split
        tmp_df = df[df[cancer_col]==cancer]
        median1 = tmp_df[split_col1].median()
        median2 = tmp_df[split_col2].median()
        df_dict = {
            'high-high': tmp_df[(tmp_df[split_col1]>median1) & (tmp_df[split_col2]>median2)],
            'high-low': tmp_df[(tmp_df[split_col1]>median1) & (tmp_df[split_col2]<=median2)],
            'low-high': tmp_df[(tmp_df[split_col1]<=median1) & (tmp_df[split_col2]>median2)],
            'low-low': tmp_df[(tmp_df[split_col1]<=median1) & (tmp_df[split_col2]<=median2)]
        }

        # performance
        for split_name, split_df in df_dict.items():
            tmp_perf_df = MetricPerformance(split_df, x_cols, y_col)
            tmp_perf_df[cancer_col] = cancer
            tmp_perf_df['group'] = split_name
            perf_dict[f'{cancer}:{split_name}'] = tmp_perf_df
    
    # concat
    perf_df = pd.concat(perf_dict.values(), ignore_index=True)

    return perf_df


##############################
#     Clonality Analysis     #
##############################

# parsing pyclone loci file
def ParsePycloneLoci(file):
    df = pd.read_csv(file, sep='\t')
    df['#CHROM'] = df['mutation_id'].apply(lambda x: x.split('_')[0])
    df['POS'] = df['mutation_id'].apply(lambda x: int(x.split('_')[1]))
    df['REF'] = df['mutation_id'].apply(lambda x: x.split('_')[2])
    df['ALT'] = df['mutation_id'].apply(lambda x: x.split('_')[3])
    df['Sample'] = df['sample_id'].apply(lambda x: '_'.join(x.split('_')[:-1]))
    return df


# parsing pyclone cluster file
def ParsePycloneCluster(file):
    df = pd.read_csv(file, sep='\t')
    df['Sample'] = df['sample_id'].apply(lambda x: '_'.join(x.split('_')[:-1]))
    return df


# measuring tumor heterogeneity
class TumorClonality():
    def __init__(self, cluster_df, sample_col='Sample', cluster_col='cluster_id', size_col='size', prevalence_col='mean'):
        self.cluster_df = cluster_df
        self.sample_col = sample_col
        self.cluster_col = cluster_col
        self.size_col = size_col
        self.prevalence_col = prevalence_col

    def heterogeneity(self):
        samples = self.cluster_df[self.sample_col].unique().tolist()
        df = pd.DataFrame(index=samples)
        df['#Clusters'] = self.cluster_df.groupby(self.sample_col).size()

        # size
        df['sShannon'] = self.cluster_df.groupby(self.sample_col)[self.size_col].apply(lambda x: self._shannon(x))
        df['sSimpson'] = self.cluster_df.groupby(self.sample_col)[self.size_col].apply(lambda x: self._simpson(x))
        df['sGini'] = self.cluster_df.groupby(self.sample_col)[self.size_col].apply(lambda x: self._gini(x))

        # prevalence
        df['pShannon'] = self.cluster_df.groupby(self.sample_col)[self.prevalence_col].apply(lambda x: self._shannon(x))
        df['pSimpson'] = self.cluster_df.groupby(self.sample_col)[self.prevalence_col].apply(lambda x: self._simpson(x))
        df['pGini'] = self.cluster_df.groupby(self.sample_col)[self.prevalence_col].apply(lambda x: self._gini(x))

        # weighted
        self.cluster_df['wSize'] = self.cluster_df[self.size_col] * self.cluster_df[self.prevalence_col]
        df['wShannon'] = self.cluster_df.groupby(self.sample_col)['wSize'].apply(lambda x: self._shannon(x))
        df['wSimpson'] = self.cluster_df.groupby(self.sample_col)['wSize'].apply(lambda x: self._simpson(x))
        df['wGini'] = self.cluster_df.groupby(self.sample_col)['wSize'].apply(lambda x: self._gini(x))

        return df

    def _shannon(self, arr):
        norm_arr = arr / arr.sum()
        entropy = -np.sum(norm_arr * np.log(norm_arr))
        norm_entropy = entropy / np.log(len(arr))
        return norm_entropy
    
    def _simpson(self, arr):
        norm_arr = arr / arr.sum()
        return 1 - np.sum(norm_arr ** 2)
    
    def _gini(self, arr):
        n = len(arr)
        mu = np.mean(arr)
        return sum(sum(abs(i-j) for i in arr) for j in arr) / (2 * n**2 * mu)


# scoring by subclonal structure
# supporting one presentation and one recognition metrics
class SubcloneScoring():
    def __init__(
        self,
        mutation_df,
        cluster_df,
        cluster_col='cluster_id',
        sample_col='Sample',
        mutation_prev_col='cellular_prevalence', 
        cluster_prev_col='mean'
    ):
        # data
        self.mutation_df = mutation_df
        self.cluster_df = cluster_df

        # column names
        self.cluster_col = cluster_col
        self.sample_col = sample_col
        self.mutation_prev_col = mutation_prev_col
        self.cluster_prev_col = cluster_prev_col

        # add clonal prevalence to mutation_df
        cluster_prev_dict = self.cluster_df.set_index([self.sample_col, self.cluster_col])[self.cluster_prev_col].to_dict()
        self.mutation_df[self.cluster_prev_col] = self.mutation_df.apply(lambda row: cluster_prev_dict.get((row[self.sample_col], row[self.cluster_col]), np.nan), axis=1)


    # scoring for each sample/tumor
    def scoring(
        self,
        sample_df,
        metric_cols=['NeoPrecis_MHCI', 'NeoPrecis_MHCII'],                              # target metrics in mutation_df
        dual_metrics_dict={'NeoPrecis_dualMHC': ('NeoPrecis_MHCI', 'NeoPrecis_MHCII')}, # metrics to be combined (MHCI + MHCII)
        mutation_pooling='sum',                                                         # method for mutation pooling
        subclone_pooling='mean',                                                        # method for subclonal pooling
    ):
        # scoring by single metric
        for metric in metric_cols:
            score_dict = self._scoring_tumor(self.mutation_df, self.cluster_df, metric, mutation_pooling=mutation_pooling, subclone_pooling=subclone_pooling)
            sample_df[metric] = score_dict

        # scoring by dual metrics
        for name, (metric_a, metric_b) in dual_metrics_dict.items():
            sample_df[name] = self._both_pathway_scoring(
                self.mutation_df,
                self.cluster_df,
                metric_a,
                metric_b,
                mutation_pooling=mutation_pooling,
                subclone_pooling=subclone_pooling
            )
        
        return sample_df
    

    # scoring for each sample/tumor with single metric
    # pooling = 'max' or 'sum' or 'mean'
    def _scoring_tumor(self, mutation_df, cluster_df, metric, mutation_pooling='max', subclone_pooling='mean'):
        # mutation pooling
        score_df = mutation_df.groupby([self.sample_col, self.cluster_col])[metric]
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
        if cluster_df.index.names != [self.sample_col, self.cluster_col]:
            cluster_df = cluster_df.set_index([self.sample_col, self.cluster_col])
        cluster_df[metric] = score_df # appending cluster score
        cluster_df[metric] = cluster_df[metric].fillna(0)

        # subclone pooling (with weight)
        cluster_df['weighted_score'] = cluster_df[metric] * cluster_df[self.cluster_prev_col]
        g = cluster_df.groupby(self.sample_col)
        if subclone_pooling == 'max':
            score_dict = (g['weighted_score'].max()).to_dict()
        elif subclone_pooling == 'sum':
            score_dict = (g['weighted_score'].sum()).to_dict()
        elif subclone_pooling == 'mean':
            score_dict = (g['weighted_score'].mean()).to_dict()
        elif subclone_pooling == 'weight':
            score_dict = (g['weighted_score'].sum() / g[self.cluster_prev_col].sum()).to_dict()
        else:
            print('subclone_pooling should be max, sum, mean, or weight')
            return

        return score_dict


    # scoring for each sample/tumor with dual metrics (MHC-I + MHC-II)
    def _both_pathway_scoring(self, mutation_df, cluster_df, mhci_method, mhcii_method, mutation_pooling='max', subclone_pooling='mean'):
        # mutation pooling
        mhci_score_df = mutation_df.groupby([self.sample_col, self.cluster_col])[mhci_method]
        if mutation_pooling == 'max':
            mhci_score_df = mhci_score_df.max()
        elif mutation_pooling == 'sum':
            mhci_score_df = mhci_score_df.sum()
        elif mutation_pooling == 'mean':
            mhci_score_df = mhci_score_df.mean()
        else:
            print('mutation_pooling should be max, sum, or mean')
            return

        mhcii_score_df = mutation_df.groupby([self.sample_col, self.cluster_col])[mhcii_method]
        if mutation_pooling == 'max':
            mhcii_score_df = mhcii_score_df.max()
        elif mutation_pooling == 'sum':
            mhcii_score_df = mhcii_score_df.sum()
        elif mutation_pooling == 'mean':
            mhcii_score_df = mhcii_score_df.mean()
        else:
            print('mutation_pooling should be max, sum, or mean')
            return

        ## MHC-I / MHC-II ratio
        #ratio = (mutation_df[mhci_method] > 0).sum() / (mutation_df[mhcii_method] > 0).sum()
        #score_df = ((mhci_score_df ** ratio) * mhcii_score_df) ** (1 / (ratio + 1))
        
        score_df = mhci_score_df * mhcii_score_df
        
        # cluster
        if cluster_df.index.names != [self.sample_col, self.cluster_col]:
            cluster_df = cluster_df.set_index([self.sample_col, self.cluster_col])
        cluster_df['score'] = score_df # appending cluster score
        cluster_df['score'] = cluster_df['score'].fillna(0)

        # subclone pooling (with weight)
        cluster_df['weighted_score'] = cluster_df['score'] * cluster_df[self.cluster_prev_col]
        g = cluster_df.groupby(self.sample_col)
        if subclone_pooling == 'max':
            score_dict = (g['weighted_score'].max()).to_dict()
        elif subclone_pooling == 'sum':
            score_dict = (g['weighted_score'].sum()).to_dict()
        elif subclone_pooling == 'mean':
            score_dict = (g['weighted_score'].mean()).to_dict()
        elif subclone_pooling == 'weight':
            score_dict = (g['weighted_score'].sum() / g[self.cluster_prev_col].sum()).to_dict()
        else:
            print('subclone_pooling should be max, sum, mean, or weight')
            return

        return score_dict
    

# decode the string in Newick format
# input: string, e.g., "((((3)2)1,(5)4)0)root;"
# return: parent_dict[child node, parent node]
def DecodeTree(s):
    # check string
    if not s.endswith('root;'):
        print('Invalid Newick format')
        return

    # preprocessing
    s = s.replace('root;', '@')

    # main
    parent_dict = dict()
    parent_stack = list()
    for c in s[::-1]:
        if c == '(':
            parent_stack = parent_stack[:-1]
        elif c == ')':
            parent_stack.append(current_node)
        elif c == ',':
            continue
        else:
            current_node = c
            if len(parent_stack) > 0:
                parent_dict[c] = parent_stack[-1]
    
    return parent_dict


##############################
#       Visualization        #
##############################

# bar plot of evaluation metric
def PerformanceBarPlot(
    perf_df,                    # performance dataframe, the output of MetricPerformance
    eval_metric,                # evaluation metric; AUROC or AUPRC
    method_col='method',        # method column name
    group_col='cancer',         # group column name (x-axis; default=cancer)
    method_rename_dict={},      # rename method names
    legend=True,                # w/ or w/o legend
    ncol=3,                     # legend ncol
    figsize=(4, 3),
    dpi=600,
    figfile=None,
    fig = None,
    ax=None,
):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)

    # rename methods
    plot_df = perf_df.copy()
    for prev, post in method_rename_dict.items():
        plot_df[method_col] = plot_df[method_col].replace(prev, post)
    
    # bar plot
    group_order = sorted(plot_df[group_col].unique().tolist(), reverse=True)
    sns.barplot(data=plot_df, x=group_col, y=eval_metric, hue=method_col, order=group_order, ax=ax, palette='pastel')
    if eval_metric == 'AUROC':
        ax.axhline(0.5, color='red', linestyle='--') # baseline line
    _ = ax.set_xlabel('')

    # annot
    n = len(perf_df[method_col].unique())
    for i in range(n):
        ax.bar_label(ax.containers[i], fmt='%.3f', fontsize=8, label_type='center')

    # legend
    if legend:
        sns.move_legend(ax, ncol=ncol, loc='lower right', bbox_to_anchor=(1, 1), title=None)
    else:
        ax.legend().remove()

    if fig is not None:
        fig.tight_layout()
        if figfile:
            fig.savefig(figfile)


# curve of ROC
def ROCCurve(
    sample_df,
    x_cols,
    y_col,
    cancer,
    cancer_col='cancer',
    method_rename_dict={},
    colors=None,
    legend=True,
    legend_fontsize=10,
    figsize=(3,3),
    dpi=600,
    figfile=None,
    fig=None,
    ax=None,
):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
    if colors is None:
        colors = sns.color_palette()[:len(x_cols)]

    # curve plot (ROC)
    plot_df = sample_df[sample_df[cancer_col]==cancer]
    for i, x_col in enumerate(x_cols):
        fpr, tpr, _ = roc_curve(plot_df[y_col], plot_df[x_col])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f'{method_rename_dict.get(x_col, x_col)} ({roc_auc:.2f})', color=colors[i])
    ax.plot([0,1], [0,1], color='navy', linestyle='--')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'ROC curve for {cancer}')
    ax.grid(alpha=0.3)
    
    # legend
    if legend:
        ax.legend(loc='lower right', bbox_to_anchor=(1, 0), fontsize=legend_fontsize)
    else:
        ax.legend().remove()

    if fig is not None:
        fig.tight_layout()
        if figfile:
            fig.savefig(figfile)


# mutation burden plot
# M-W U test
def BurdenBoxPlot(
    sample_df,                  # sample-centric dataframe
    x_cols,                     # columns to be compared
    y_col,                      # label column
    cancer,                     # cancer type
    cancer_col='cancer',        # cancer column name
    method_rename_dict={},      # rename method names
    legend=True,
    figsize=(6,4),
    dpi=600,
    figfile=None,
    fig = None,
    ax=None,
):
    # p-values
    pval_dict = dict()
    pos_df = sample_df[(sample_df[cancer_col]==cancer) & (sample_df[y_col]==1)]
    neg_df = sample_df[(sample_df[cancer_col]==cancer) & (sample_df[y_col]==0)]
    for col in x_cols:
        s, p = mannwhitneyu(pos_df[col], neg_df[col])
        pval_dict[col] = p

    # plot data
    plot_df = sample_df.reset_index().melt(id_vars=['tumor_sra_id', cancer_col, y_col], value_vars=x_cols, var_name='metric', value_name='burden')
    
    # plot
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
    sns.boxplot(data=plot_df[plot_df[cancer_col]==cancer], x='metric', y='burden', hue=y_col, ax=ax, palette='muted')
    rename_x_cols = [method_rename_dict.get(col, col) for col in x_cols]
    ax.set_xticklabels(rename_x_cols)
    ax.set_xlabel('')
    ax.set_yscale('log')
    ax.set_title(cancer)

    # extend y-axis
    y_max = plot_df['burden'].max() # overall max
    y_offset_factor = 3
    ax.set_ylim(0, y_max * y_offset_factor)

    # annot
    box_pairs = [(col, y_col) for col in x_cols]
    for i, (col, _) in enumerate(box_pairs):
        pval = pval_dict[col]
        y_max = plot_df[plot_df[cancer_col]==cancer]['burden'].max() # subgroup max
        y_offset_factor = 1.2
        x_pos = i
        y = y_max * y_offset_factor
        pval_text = f"p = {pval:.2e}"
        ax.text(x_pos, y, pval_text, ha='center', va='bottom', fontsize=10, color='black')

    # move legend
    if legend:
        handles, labels = ax.get_legend_handles_labels()
        title_handle = Line2D([], [], color="none")  # No line, no marker
        new_labels = [y_col] + labels
        new_handles = [title_handle] + handles  # Use the empty handle for the title
        ax.legend(new_handles, new_labels, loc="upper right", bbox_to_anchor=(1, 1), ncol=3)
    else:
        ax.legend().remove()

    if fig is not None:
        fig.tight_layout()
        if figfile:
            fig.savefig(figfile)


# Cliff's Delta effect size
# bar plot
def BurdenCliffPlot(
    sample_df,                  # sample-centric dataframe
    x_cols,                     # columns to be compared
    y_col,                      # label column
    cancers,                    # target cancers
    cancer_col='cancer',        # cancer column name
    method_rename_dict={},      # rename method names
    legend=True,
    figsize=(4,3),
    dpi=600,
    figfile=None,
    fig = None,
    ax=None,
    
):
    # Cliff's delta
    results = list()
    for cancer in cancers: # for each cancer type
        pos_df = sample_df[(sample_df[cancer_col]==cancer) & (sample_df[y_col]==1)] # positives
        neg_df = sample_df[(sample_df[cancer_col]==cancer) & (sample_df[y_col]==0)] # negatives
        for col in x_cols:
            pos_list = pos_df[col].tolist()
            neg_list = neg_df[col].tolist()
            d, _ = cliffs_delta(pos_list, neg_list)
            results.append({
                'cancer': cancer,
                'metric': method_rename_dict[col],
                "Cliff's Delta effect size": d,
            })
    plot_df = pd.DataFrame(results)

    # plot
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
    sns.barplot(data=plot_df, x='cancer', y="Cliff's Delta effect size", hue='metric', ax=ax, palette='pastel')
    ax.set_xlabel('')
    ax.legend(title=None)
    
    if fig is not None:
        fig.tight_layout()
        if figfile:
            fig.savefig(figfile)


def SurvivalCurvePlot(
    sample_df,                  # smaple-centric dataframe
    metric_col,                 # metric column name
    cancer,                     # cancer type
    cancer_col='cancer',        # cancer column name
    event_col='OS',             # event column name
    duration_col='OS.time',     # duration column name
    method_rename_dict={},      # rename method names
    figsize=(4,3),
    dpi=600,
    figfile=None,
    fig = None,
    ax=None,
):
    kmf = KaplanMeierFitter()
    cph = CoxPHFitter()

    # df
    srv_df = sample_df[(sample_df[cancer_col]==cancer)].dropna(subset=[event_col, duration_col])
    
    # split top and bottom
    thrs = srv_df[metric_col].median()
    top_df = srv_df[srv_df[metric_col]>thrs]
    bot_df = srv_df[srv_df[metric_col]<=thrs]
    print(f'#top = {top_df.shape[0]}; #bot = {bot_df.shape[0]}')

    # log rank test
    stat_test = logrank_test(top_df[duration_col], bot_df[duration_col],
                            event_observed_A=top_df[event_col], event_observed_B=bot_df[event_col])
    print(stat_test.test_statistic, stat_test.p_value)

    # K-P curve
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
    kmf.fit(top_df[duration_col], event_observed=top_df[event_col], label=f'top (n={top_df.shape[0]})')
    kmf.plot_survival_function(ax=ax)
    kmf.fit(bot_df[duration_col], event_observed=bot_df[event_col], label=f'bot (n={bot_df.shape[0]})')
    kmf.plot_survival_function(ax=ax)
    ax.text(0.3, 0.6, f'log-rank P = {stat_test.p_value:.2e}', transform=ax.transAxes)
    ax.set_title(f'{event_col} in {cancer} by {method_rename_dict.get(metric_col, metric_col)}')
    ax.set_ylabel('probability')
    ax.set_xlabel('days')

    if fig is not None:
        fig.tight_layout()
        if figfile:
            fig.savefig(figfile)


# Cox proportional hazards model
def CoxModeling(df, confounders, target, duration='OS.time', event='OS'):
    tmp_df = df.copy()
    tmp_df[target] = (tmp_df[target] - tmp_df[target].mean()) / tmp_df[target].std() # normalization
    cph = CoxPHFitter()
    cols = confounders[:]
    cols.append(target)
    cph.fit(tmp_df, duration_col=duration, event_col=event , formula=' + '.join(cols))
    hazard = cph.hazard_ratios_[target]
    ci_arr = cph.confidence_intervals_.loc[target].values
    ci_arr = np.exp(np.array(ci_arr))
    return hazard, ci_arr


# plot for hazard ratio
def HRPlot(
    sample_df,                  # smaple-centric dataframe
    metric_cols,                # metric columns (list)
    confounder_cols=[],         # confounder columns (list)
    event_col='OS',             # event column name
    duration_col='OS.time',     # duration column name
    method_rename_dict={},      # rename method names
    figsize=(3,3),
    dpi=600,
    figfile=None,
    fig = None,
    ax=None,
):
    # for each metric
    hrs, cis = list(), list()
    for metric in metric_cols:
        hazard, ci_arr = CoxModeling(sample_df, confounder_cols, metric, duration=duration_col, event=event_col)
        hrs.append(hazard)
        cis.append(ci_arr)
    hrs = np.array(hrs)
    cis = np.array(cis)
    y_pos = np.arange(len(metric_cols))
    lower_errors = hrs - cis[:, 0]
    upper_errors = cis[:, 1] - hrs

    # plot
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
    ax.errorbar(hrs, y_pos, xerr=[lower_errors, upper_errors], fmt='o', capsize=5, color='black')
    ax.set_yticks(y_pos)
    ax.set_yticklabels([method_rename_dict.get(col, col) for col in metric_cols])
    ax.axvline(1, color='gray', linestyle='--')
    ax.set_xlabel('Hazard Ratio (95% CI)')
    ax.grid(True, axis='x')
    
    if fig is not None:
        fig.tight_layout()
        if figfile:
            fig.savefig(figfile)
