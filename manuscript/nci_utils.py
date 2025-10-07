import os, sys, copy, warnings
from collections import defaultdict, OrderedDict
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import plot_tree
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm
warnings.filterwarnings('ignore')

dpi = 600 # plot


def NonParamTest(df, x, y, alternative='greater'):
    x0 = df.loc[df[y]==0, x].astype(float).to_numpy()
    x1 = df.loc[df[y]==1, x].astype(float).to_numpy()
    s, p = mannwhitneyu(x1, x0, alternative=alternative)
    return p


def BuildDataset(file,
                 abundance_features,
                 presentation_features,
                 recognition_features,
                 sample_col='Patient',
                 index_cols=['Patient', 'Mutation_Index', 'Mutation_ID'],
                 missense_name='missense_variant'):
    
    df = pd.read_csv(file)

    ### filtering
    print('#Mutations')
    print(f'Before filtering: {df.shape[0]}')
    print(f'#CD8: {(df["CD8"]==1).sum()}')
    print(f'#CD4: {(df["CD4"]==1).sum()}')
    # substitution mutations
    df = df[df['Consequence']==missense_name]
    print(f'Drop non-SNVs: {df.shape[0]}')
    print(f'#CD8: {(df["CD8"]==1).sum()}')
    print(f'#CD4: {(df["CD4"]==1).sum()}')

    # normalization
    #df['PHBR-I'] = -np.log((df['PHBR-I']+1e-3)/100)
    #df['PHBR-II'] = -np.log((df['PHBR-II']+1e-3)/100)
    df['PHBR-I'] = 1 - df['PHBR-I']/100
    df['PHBR-II'] = 1 - df['PHBR-II']/100
    
    ### data object
    data = NeoAgData(
        df,
        sample_col=sample_col, # individual ID
        index_cols=index_cols, # unique neoantigen
        abundance_features=abundance_features,
        presentation_features=presentation_features,
        recognition_features=recognition_features,
    )

    return data


def FeatureSelection(df, x_cols, y_col, eval_metric='roc_auc'):
    X = df[x_cols].to_numpy()
    y = df[y_col].to_numpy()

    # normalization
    scaler = StandardScaler()
    X_norm = scaler.fit_transform(X)

    # model
    model = LogisticRegression()

    # RFECV
    rfecv = RFECV(
        estimator=model,
        step=1,
        cv=4,
        scoring=eval_metric,
    )
    rfecv.fit(X_norm, y)

    return rfecv


### mutation-level data object for machine learning model
class NeoAgData():
    def __init__(
            self,
            neo_df,
            sample_col='ID',                                                    # individual ID
            index_cols=['ID', 'Mutation_Index', 'Mutation_ID'],                 # unique neoantigen
            abundance_features=['DNA_AF', 'RNA_AF', 'RNA_EXP'],                 # 'DNA_AF', 'RNA_AF', 'RNA_EXP'
            presentation_features=['PHBR', 'Robustness'],                       # 'PHBR', 'Robustness'
            recognition_features=['CRDistance', 'Agretopicity', 'Foreignness'], # 'CRDistance', 'Agretopicity', 'Foreignness'
            mhci_label='CD8',
            mhcii_label='CD4',
    ):
        self.df = neo_df.copy()
        self.sample_col = sample_col
        self.samples = self.df[sample_col].unique().tolist()
        
        # feature set
        self.feature_dict = defaultdict(list)
        self.feature_dict['abundance'] = abundance_features
        self.feature_dict['presentation-I'] = [f'{feature}-I' for feature in presentation_features]
        self.feature_dict['presentation-II'] = [f'{feature}-II' for feature in presentation_features]
        self.feature_dict['recognition-I'] = [f'{feature}-I' for feature in recognition_features]
        self.feature_dict['recognition-II'] = [f'{feature}-II' for feature in recognition_features]
        self.features = sum(list(self.feature_dict.values()), [])
        self.labels = [mhci_label, mhcii_label]

        # solve missing
        self.df = self.df[index_cols + self.features + self.labels]
        self.df = self.df.fillna(0)
        self.df = self.df.reset_index(drop=True)


    def GetData(self, label, feature_groups=list(), features=list()):
        self.current_label = label

        # features
        if (len(features)==0) & (len(feature_groups)==0):
            features = self.features
        elif len(feature_groups) != 0:
            features = list()
            for feature_group in feature_groups:
                features += self.feature_dict[feature_group]
        else:
            features = features
        self.current_features = features
        
        # x, y
        x = self.df[features].to_numpy()
        y = self.df[label].to_numpy()

        return x, y


    # if split_by_samples == False, train_list and test_list should be the index
    # if split_by_samples == True, split train, test by samples; train_list and test_list should be sample idx
    # return x, y, group (sample)
    def GetSplitData(self, train_list, test_list, label, split_by_samples=False, feature_groups=list(), features=list()):
        # index
        if split_by_samples:
            train_samples = [self.samples[i] for i in train_list]
            test_samples = [self.samples[i] for i in test_list]
            train_idx = self.df[self.df[self.sample_col].isin(train_samples)].index.tolist()
            test_idx = self.df[self.df[self.sample_col].isin(test_samples)].index.tolist()
        else:
            train_idx = train_list[:]
            test_idx = test_list[:]
        train_groups = self.df.loc[train_idx, self.sample_col].tolist()
        test_groups = self.df.loc[test_idx, self.sample_col].tolist()

        # features
        if (len(features)==0) & (len(feature_groups)==0):
            features = self.features
        elif len(feature_groups) != 0:
            features = list()
            for feature_group in feature_groups:
                features += self.feature_dict[feature_group]
        else:
            features = features
        self.current_features = features

        # dataset split
        train_x = self.df.iloc[train_idx][features].to_numpy()
        train_y = self.df.iloc[train_idx][label].to_numpy()
        test_x = self.df.iloc[test_idx][features].to_numpy()
        test_y = self.df.iloc[test_idx][label].to_numpy()

        return train_x, train_y, train_groups, test_x, test_y, test_groups

        
### cross validation
class CrossValidation():
    def __init__(
            self,
            data: NeoAgData,    # NeoAgData object
            model,              # sklearn ML model
            importance=True,    # output feature importance from sklearn ML model
    ):
        self.data = data
        self.samples = self.data.samples
        self.model = model
        self.importance = importance
        self.eval = Evaluation()

    def __call__(self, tasks, split_by_samples=False, n_fold=5, n_exp=10, shuffle=True, normalized=False, random_state=42):
        performances, importances = list(), list()
        split_list = self.samples if split_by_samples else list(range(self.data.df.shape[0]))
        
        # for each exp (different data split)
        for i in tqdm(range(n_exp)):
            # Set a different seed for each experiment for reproducibility
            exp_seed = random_state + i if random_state is not None else None
            kf = KFold(n_splits=n_fold, shuffle=shuffle, random_state=exp_seed)
            
            # for each task - collect predictions across all folds
            for task, info in tqdm(tasks.items(), total=len(tasks), leave=False):
                # Initialize lists to collect all predictions and labels across folds
                all_test_y = []
                all_test_pred = []
                all_test_groups = []
                all_importances_per_fold = []
                
                # for each fold
                for k, (train_idx, test_idx) in enumerate(kf.split(split_list)):
                    # dataset
                    train_x, train_y, train_groups, test_x, test_y, test_groups = self.data.GetSplitData(
                        train_idx,
                        test_idx,
                        info['label'],
                        split_by_samples=split_by_samples,
                        feature_groups=info['feature_group']
                    )
                    
                    # training
                    if normalized:
                        mean_arr = train_x.mean(axis=0)
                        std_arr = train_x.std(axis=0)
                        # Avoid division by zero
                        std_arr = np.where(std_arr == 0, 1, std_arr)
                        train_x = (train_x - mean_arr) / std_arr
                        test_x = (test_x - mean_arr) / std_arr
    
                    self.model.fit(train_x, train_y)
                    test_pred = self.model.predict_proba(test_x)[:, 1]
    
                    # Collect predictions and labels
                    all_test_y.append(test_y)
                    all_test_pred.append(test_pred)
                    if test_groups is not None:
                        all_test_groups.append(test_groups)
    
                    # importance per fold
                    if self.importance:
                        all_importances_per_fold.append(self.model.feature_importances_)
    
                # Convert to numpy arrays
                all_test_y = np.concatenate(all_test_y)
                all_test_pred = np.concatenate(all_test_pred)
                all_test_groups = np.concatenate(all_test_groups) if len(all_test_groups) > 0 else None
                
                # Compute performance once on all predictions
                result = self.eval(all_test_y, all_test_pred, groups=all_test_groups)
                performance = {
                    'task': task,
                    'exp': i,
                    **result,
                }
                performances.append(performance)
    
                # Average importance across folds for this experiment
                if self.importance:
                    avg_importance = np.mean(all_importances_per_fold, axis=0)
                    importance = {
                        'task': task,
                        'exp': i,
                        **dict(zip(self.data.current_features, avg_importance))
                    }
                    importances.append(importance)
    
        # aggregation across experiments
        performance_df = pd.DataFrame(performances)
        
        if self.importance:
            importance_df = pd.DataFrame(importances)
        else:
            importance_df = None
    
        return performance_df, importance_df


### isolated validation
class IsolatedValidation():
    def __init__(
        self,
        train_data: NeoAgData,    # NeoAgData object
        test_data: NeoAgData,     # NeoAgData object
    ):
        self.train_data = train_data
        self.test_data = test_data
        self.eval = Evaluation()

    
    def __call__(self, model, tasks, out_dir, importance=True, tree=True, normalized=False):
        if not os.path.isdir(out_dir): os.mkdir(out_dir)
        if not os.path.isdir(f'{out_dir}/interpretation'): os.mkdir(f'{out_dir}/interpretation')

        pred_df, results, importances = pd.DataFrame(), dict(), dict()
        for name, info in tasks.items():
            # data
            train_x, train_y = self.train_data.GetData(info['label'], feature_groups=info['feature_group'])
            test_x, test_y = self.test_data.GetData(info['label'], feature_groups=info['feature_group'])

            # normalized
            if normalized:
                mean_arr = train_x.mean(axis=0)
                std_arr = train_x.mean(axis=0)
                train_x = (train_x - mean_arr) / std_arr
                test_x = (test_x - mean_arr) / std_arr
            
            # prediction
            model.fit(train_x, train_y)
            test_pred = model.predict_proba(test_x)[:,1]
            pred_df[name] = test_pred
            results[name] = self.eval(test_y, test_pred)
            
            # model interpretation
            name = name.replace('-', '__').replace('+', '_')
            plot_dir = f'{out_dir}/interpretation/{name}'
            if not os.path.isdir(plot_dir): os.mkdir(plot_dir)
            features = self.train_data.current_features
            if importance:
                importances[name] = dict(zip(features, model.feature_importances_))
                self._importance_plot(model, features, plot_dir)
            if tree:
                self._tree_plot(model, features, plot_dir)
        
        # save
        result_df = pd.DataFrame(results)
        result_df.to_csv(f'{out_dir}/performances.csv')
        pred_df.to_csv(f'{out_dir}/predictions.csv')
        if importance:
            importance_df = pd.DataFrame(importances)
            importance_df.to_csv(f'{out_dir}/importances.csv')
        return pred_df, result_df

    
    def _importance_plot(self, model, features, out_dir):
        plot_df = pd.DataFrame({'feature': features, 'importance': model.feature_importances_})
        fig, ax = plt.subplots(1, 1, figsize=(len(features), 4), dpi=300)
        sns.barplot(data=plot_df, x='feature', y='importance', ax=ax)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
        fig.tight_layout()
        fig.savefig(f'{out_dir}/importance.png')
    

    def _tree_plot(self, model, features, out_dir):
        for i in range(len(model.estimators_)):
            estimator = model.estimators_[i, 0]
            fig, ax = plt.subplots(1, 1, figsize=(6, 4), dpi=300)
            plot_tree(estimator, filled=True, feature_names=features, rounded=True, ax=ax)
            fig.tight_layout()
            fig.savefig(f'{out_dir}/tree_{i}.png')


class Evaluation():
    def __init__(self):
        return
    
    def __call__(self, y, pred, groups=None, max_fpr=0.1, topk_list=None):
        # top K for PPV
        if topk_list is None:
            topk_list = [10, 20, 30, 40, 50]
        else:
            topk_list = topk_list[:]
        topk_list.append(sum(y))

        # metrics
        auroc = metrics.roc_auc_score(y, pred)
        auroc_ = metrics.roc_auc_score(y, pred, max_fpr=max_fpr)
        auprc = metrics.average_precision_score(y, pred)
        topk_result = self._topk_performance(y, pred, topk_list=topk_list)
        avg_rank = self._avg_rank(y, pred, groups) if groups is not None else np.nan
        
        result = {'AUROC': auroc, f'AUROC_{max_fpr}': auroc_, 'AUPRC': auprc, 'avgRank': avg_rank, **topk_result}
        return result


    def _avg_rank(self, y, pred, groups):
        uniq_groups = list(set(groups))
        ranks = list()
        for group in uniq_groups:
            tmp_idx = [i for i, g in enumerate(groups) if g == group]
            tmp_pred = pred[tmp_idx]
            tmp_y = y[tmp_idx]
            if 1 not in tmp_y: continue
            tmp_ranks = self._rank(tmp_y, tmp_pred)
            ranks.append(np.mean(tmp_ranks))
        return np.mean(ranks)
    
    
    def _rank(self, y, pred):
        n = len(y)
        t = [(pred[i], y[i]) for i in range(n)]
        t = sorted(t, reverse=True)
        ranks = [i/n for i in range(n) if t[i][1]==1]
        return ranks
        
    
    def _topk_performance(self, y, pred, topk_list=list(range(10, 51, 10))):
        total_positives = sum(y)
        sort_idx = np.argsort(pred)
        results = dict()
        for topk in topk_list:
            positives = y[sort_idx[-topk:]].sum()
            results[f'top{topk}_recall'] = positives / total_positives
            results[f'top{topk}_precision'] = positives / topk
        return results
        
    
    def _transfer_topk_df(self, df):
        id_vars = ['task', 'exp']
        
        # recall
        recall_cols = [i for i in df.columns if 'recall' in i]
        recall_df = pd.melt(df, id_vars=id_vars, value_vars=recall_cols, value_name='recall', var_name='topK')
        recall_df['topK'] = recall_df['topK'].apply(lambda x: x.split('_')[0][3:])
    
        # precision
        precision_cols = [i for i in df.columns if 'precision' in i]
        precision_df = pd.melt(df, id_vars=id_vars, value_vars=precision_cols, value_name='precision', var_name='topK')
        precision_df['topK'] = precision_df['topK'].apply(lambda x: x.split('_')[0][3:])
        
        return recall_df, precision_df
    

    def _performance_plot(self, df, target_tasks, title='', figfile=None):
        tmp_df = df[df['task'].isin(target_tasks)]
        fig, ax = plt.subplots(1, 2, figsize=(12,5), dpi=300)
        sns.barplot(data=tmp_df, x='task', y='AUROC', order=target_tasks, ax=ax[0])
        sns.barplot(data=tmp_df, x='task', y='AUPRC', order=target_tasks, ax=ax[1])
        ax[0].set_xticklabels(ax[0].get_xticklabels(), rotation=90)
        ax[1].set_xticklabels(ax[1].get_xticklabels(), rotation=90)
        ax[0].set_xlabel('')
        ax[1].set_xlabel('')
        fig.suptitle(title)
        fig.tight_layout()
        if figfile:
            fig.savefig(figfile)

    
    def _rank_plot(self, df, target_tasks, title='', figfile=None):
        tmp_df = df[df['task'].isin(target_tasks)]
        fig, ax = plt.subplots(1, 1, figsize=(6,5), dpi=300)
        sns.barplot(data=tmp_df, x='task', y='avgRank', order=target_tasks, ax=ax)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
        ax.set_xlabel('')
        fig.suptitle(title)
        fig.tight_layout()
        if figfile:
            fig.savefig(figfile)


    def _topk_plot(self, df, target_tasks, title='', figfile=None):
        tmp_df = df[df['task'].isin(target_tasks)]
        recall_df, precision_df = self._transfer_topk_df(tmp_df)

        fig, ax = plt.subplots(1, 3, figsize=(12, 5), dpi=300, gridspec_kw={'width_ratios': [0.2, 0.4, 0.4]})
        sns.lineplot(data=recall_df, x='topK', y='recall', hue='task', ax=ax[1])
        sns.lineplot(data=precision_df, x='topK', y='precision', hue='task', ax=ax[2])
        ax[1].set_xticklabels(ax[1].get_xticklabels(), rotation=90)
        ax[2].set_xticklabels(ax[2].get_xticklabels(), rotation=90)
        handles, labels = ax[1].get_legend_handles_labels()
        ax[1].get_legend().remove()
        ax[2].get_legend().remove()
        ax[0].legend(handles=handles, labels=labels, loc='upper left')
        ax[0].spines['top'].set_visible(False)
        ax[0].spines['right'].set_visible(False)
        ax[0].spines['bottom'].set_visible(False)
        ax[0].spines['left'].set_visible(False)
        ax[0].tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
        fig.suptitle(title)
        fig.tight_layout()
        if figfile:
            fig.savefig(figfile)

    
    def _importance_plot(self, importance_df, title='', figfile=None):
        df = importance_df.groupby('task').mean()
        df = df.drop(columns=['exp', 'fold'])

        fig, ax = plt.subplots(1, 1, figsize=(8, 8), dpi=300)
        sns.heatmap(df, cmap='Blues', annot=True, fmt='.2f', ax=ax)
        ax.set_ylabel('')
        fig.suptitle(title)
        fig.tight_layout()
        if figfile:
            fig.savefig(figfile)


'''
### feature sets
task_dict = OrderedDict({
    ### CD8
    'CD8-A': {'label':'CD8', 'feature_group':['abundance']},
    'CD8-P1': {'label':'CD8', 'feature_group':['presentation-I']},
    'CD8-R1': {'label':'CD8', 'feature_group':['recognition-I']},
    'CD8-P1+P2': {'label':'CD8', 'feature_group':['presentation-I', 'presentation-II']},
    'CD8-A+P1': {'label':'CD8', 'feature_group':['abundance', 'presentation-I']},
    'CD8-A+P1+R1': {'label':'CD8', 'feature_group':['abundance', 'presentation-I', 'recognition-I']},
    'CD8-A+P1+P2': {'label':'CD8', 'feature_group':['abundance', 'presentation-I', 'presentation-II']},
    'CD8-A+P1+R1+P2': {'label':'CD8', 'feature_group':['abundance', 'presentation-I', 'recognition-I', 'presentation-II']},
    'CD8-A+P1+R1+P2+R2': {'label':'CD8', 'feature_group':['abundance', 'presentation-I', 'recognition-I', 'presentation-II', 'recognition-II']},
    ### CD4
    'CD4-A': {'label':'CD4', 'feature_group':['abundance']},
    'CD4-P2': {'label':'CD4', 'feature_group':['presentation-II']},
    'CD4-R2': {'label':'CD4', 'feature_group':['recognition-II']},
    'CD4-P2+P1': {'label':'CD4', 'feature_group':['presentation-II', 'presentation-I']},
    'CD4-A+P2': {'label':'CD4', 'feature_group':['abundance', 'presentation-II']},
    'CD4-A+P2+R2': {'label':'CD4', 'feature_group':['abundance', 'presentation-II', 'recognition-II']},
    'CD4-A+P2+P1': {'label':'CD4', 'feature_group':['abundance', 'presentation-II', 'presentation-I']},
    'CD4-A+P2+R2+P1': {'label':'CD4', 'feature_group':['abundance', 'presentation-II', 'recognition-II', 'presentation-I']},
    'CD4-A+P2+R2+P1+R1': {'label':'CD4', 'feature_group':['abundance', 'presentation-II', 'recognition-II', 'presentation-I', 'recognition-I']},
})
'''