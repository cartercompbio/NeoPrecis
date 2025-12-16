import os
import sys
import json
from collections import defaultdict, OrderedDict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu, chi2_contingency
import warnings
warnings.filterwarnings("ignore")

project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_dir)
from neoprecis import SubCRD


'''
from matplotlib.gridspec import GridSpec
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import seaborn as sns
from scipy.stats import ttest_ind, mannwhitneyu, chi2_contingency
from sklearn import metrics
'''

######################
### Figure Setting ###
######################

def set_font_style(font_family='sans-serif', font='DejaVu Sans', base_fontsize=7):
    """
    Sets global plot style for publication quality.
    """
    
    # 1. Reset any previous settings to avoid conflicts
    plt.rcdefaults()
    
    # 2. Set Font Family
    # If Arial isn't found, it falls back to standard sans-serif (DejaVu Sans)
    plt.rcParams['font.family'] = font_family
    plt.rcParams[f'font.{font_family}'] = [font]
    
    # 3. Set Font Sizes (Globally)
    plt.rcParams['font.size'] = base_fontsize          # Default text size
    plt.rcParams['axes.titlesize'] = base_fontsize     # Subplot titles slightly larger
    plt.rcParams['axes.labelsize'] = base_fontsize     # Axis labels (x, y)
    plt.rcParams['xtick.labelsize'] = base_fontsize    # Tick numbers on x
    plt.rcParams['ytick.labelsize'] = base_fontsize    # Tick numbers on y
    plt.rcParams['legend.fontsize'] = base_fontsize    # Legend text


######################
###### Blosum 62 #####
######################

class Blosum62():
    # U (X): unknown
    def __init__(self):
        self.aa_list = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V', 'B', 'Z', 'U', '*']
        self.aa_idx_dict = {c:i for i,c in enumerate(self.aa_list)}
        self.matrix = np.array([
            [ 4, -1, -2, -2,  0, -1, -1,  0, -2, -1, -1, -1, -1, -2, -1,  1,  0, -3, -2,  0, -2, -1,  0, -4],
            [-1,  5,  0, -2, -3,  1,  0, -2,  0, -3, -2,  2, -1, -3, -2, -1, -1, -3, -2, -3, -1,  0, -1, -4],
            [-2,  0,  6,  1, -3,  0,  0,  0,  1, -3, -3,  0, -2, -3, -2,  1,  0, -4, -2, -3,  3,  0, -1, -4],
            [-2, -2,  1,  6, -3,  0,  2, -1, -1, -3, -4, -1, -3, -3, -1,  0, -1, -4, -3, -3,  4,  1, -1, -4],
            [ 0, -3, -3, -3,  9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1, -3, -3, -2, -4],
            [-1,  1,  0,  0, -3,  5,  2, -2,  0, -3, -2,  1,  0, -3, -1,  0, -1, -2, -1, -2,  0,  3, -1, -4],
            [-1,  0,  0,  2, -4,  2,  5, -2,  0, -3, -3,  1, -2, -3, -1,  0, -1, -3, -2, -2,  1,  4, -1, -4],
            [ 0, -2,  0, -1, -3, -2, -2,  6, -2, -4, -4, -2, -3, -3, -2,  0, -2, -2, -3, -3, -1, -2, -1, -4],
            [-2,  0,  1, -1, -3,  0,  0, -2,  8, -3, -3, -1, -2, -1, -2, -1, -2, -2,  2, -3,  0,  0, -1, -4],
            [-1, -3, -3, -3, -1, -3, -3, -4, -3,  4,  2, -3,  1,  0, -3, -2, -1, -3, -1,  3, -3, -3, -1, -4],
            [-1, -2, -3, -4, -1, -2, -3, -4, -3,  2,  4, -2,  2,  0, -3, -2, -1, -2, -1,  1, -4, -3, -1, -4],
            [-1,  2,  0, -1, -3,  1,  1, -2, -1, -3, -2,  5, -1, -3, -1,  0, -1, -3, -2, -2,  0,  1, -1, -4],
            [-1, -1, -2, -3, -1,  0, -2, -3, -2,  1,  2, -1,  5,  0, -2, -1, -1, -1, -1,  1, -3, -1, -1, -4],
            [-2, -3, -3, -3, -2, -3, -3, -3, -1,  0,  0, -3,  0,  6, -4, -2, -2,  1,  3, -1, -3, -3, -1, -4],
            [-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4,  7, -1, -1, -4, -3, -2, -2, -1, -2, -4],
            [ 1, -1,  1,  0, -1,  0,  0,  0, -1, -2, -2,  0, -1, -2, -1,  4,  1, -3, -2, -2,  0,  0,  0, -4],
            [ 0, -1,  0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1,  1,  5, -2, -2,  0, -1, -1,  0, -4],
            [-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1,  1, -4, -3, -2, 11,  2, -3, -4, -3, -2, -4],
            [-2, -2, -2, -3, -2, -1, -2, -3,  2, -1, -1, -2, -1,  3, -3, -2, -2,  2,  7, -1, -3, -2, -1, -4],
            [ 0, -3, -3, -3, -1, -2, -2, -3, -3,  3,  1, -2,  1, -1, -2, -2,  0, -3, -1,  4, -3, -2, -1, -4],
            [-2, -1,  3,  4, -3,  0,  1, -1,  0, -3, -4,  0, -3, -3, -2,  0, -1, -4, -3, -3,  4,  1, -1, -4],
            [-1,  0,  0,  1, -3,  3,  4, -2,  0, -3, -3,  1, -1, -3, -1,  0, -1, -3, -2, -2,  1,  4, -1, -4],
            [ 0, -1, -1, -1, -2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2,  0,  0, -2, -1, -1, -1, -1, -1, -4],
            [-4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4,  1]
        ])
        self.sub_dict = pd.DataFrame(self.matrix, columns=self.aa_list, index=self.aa_list).to_dict()


######################
## Statistical Test ##
######################

def NonParamTest(df, x, y, alternative='greater'):
    x0 = df.loc[df[y]==0, x].astype(float).to_numpy()
    x1 = df.loc[df[y]==1, x].astype(float).to_numpy()
    s, p = mannwhitneyu(x1, x0, alternative=alternative)
    return p

def Chi2Test(arr_a, arr_b):
    # ensure boolean masks (works for numpy arrays, pandas Series, lists, etc.)
    a_mask = np.asarray(arr_a).astype(bool)
    b_mask = np.asarray(arr_b).astype(bool)

    table = np.zeros((2,2), dtype=int)
    table[0][0] = np.sum(~a_mask & ~b_mask)
    table[0][1] = np.sum(~a_mask & b_mask)
    table[1][0] = np.sum(a_mask & ~b_mask)
    table[1][1] = np.sum(a_mask & b_mask)

    result = chi2_contingency(table)
    return result.pvalue


####################################
### Data Processing for Epitopes ###
####################################

def MapSeqToCorePos(seq, core):
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

# MT and WT should be with the same length
def PseudoCore(mt, mt_core, wt):
    idx_map_dict = MapSeqToCorePos(mt, mt_core)
    pseudo_core = list(mt_core)
    for i, j in idx_map_dict.items():
        pseudo_core[j] = wt[i]
    return ''.join(pseudo_core)

def ExtractSubstitution(file, wt_seq_col='WT_epitope', mt_seq_col='MT_epitope', wt_core_col='WT_core', mt_core_col='MT_core'):
    df = pd.read_csv(file)
    subs = list()
    for idx, row in df.iterrows():
        wt, mt, wt_core, mt_core = row[wt_seq_col], row[mt_seq_col], row[wt_core_col], row[mt_core_col]
        if type(wt) is float:
            continue # check if WT exists
        if len(wt) != len(mt):
            continue # check if substitution

        # seq to core index of MT
        idx_map_dict = MapSeqToCorePos(mt, mt_core)

        # get subs
        count = 0
        for i in range(len(mt)):
            if wt[i] != mt[i]:
                if i not in idx_map_dict:
                    continue # mutation is not in core
                pos, wt_aa, mt_aa = idx_map_dict[i], wt[i], mt[i]
                count += 1

        # pesudo core for WT based on the mapping to MT core
        pseudo_core = PseudoCore(mt, mt_core, wt)

        # add to list
        if count == 1: # add substitution if #mismatches = 1
            subs.append([row['allele'], pseudo_core, count, pos, wt_aa, mt_aa])
        else:
            subs.append([row['allele'], pseudo_core, count, np.nan, '', ''])
        
    # sub_df
    sub_df = pd.DataFrame(subs, columns=['allele', 'WT_core_pseudo', 'mismatches', 'pos', 'WT_aa', 'MT_aa'])
    sub_df = df.merge(sub_df, how='right', on=['allele'])
    
    return sub_df

# single substitution on the core region
# convert to 1-base
def FilterSingleSub(df, binding_threshold=None):
    if binding_threshold is not None:
        df = df[df['MT_rank']<=binding_threshold] # MHC binding
    df = df[df['MT_core']!=df['WT_core']] # core mutation
    df = df[~df['MT_aa'].isna()] # single mutation
    df = df[~(df['MT_aa']=='-')]
    df = df[~(df['WT_aa']=='-')]
    df['pos'] += 1 # 0-base to 1-base
    return df

# count mismatches between two peptides
def countMismatches(row, mt_col='MT_core', wt_col='WT_core_pseudo'):
    length = 9 # core length should always be 9
    count = 0
    for i in range(length):
        if row[mt_col][i] != row[wt_col][i]:
            count += 1
    return count


#####################################
### Data Processing for netMHCpan ###
#####################################

### prepare input files for NetMHCpan
# input allele: "A*01:01"
# rename allele: "A_01_01"
def PrepNetMHCpanInput(
        df: pd.DataFrame,
        allele_col: str,
        pept_cols: list,
        out_dir: str,
        length_list=[8,9,10,11]
):
    if not os.path.isdir(f'{out_dir}/inputs'): # create input dir
        os.mkdir(f'{out_dir}/inputs')
    
    alleles = list()
    for allele in df[allele_col].unique().tolist():
        allele_name = allele.replace('*', '_').replace(':', '_') # normalized allele

        # get peptides
        pepts = list()
        for pept_col in pept_cols:
            tmp_pepts = df.loc[df[allele_col]==allele, pept_col].dropna().tolist()  # filter by alleles
            tmp_pepts = [pept for pept in tmp_pepts if len(pept) in length_list]    # filter by length
            pepts += tmp_pepts
        pepts = sorted(list(set(pepts)))
        
        # save peptides
        if len(pepts) > 0:
            alleles.append(allele)
            with open(f'{out_dir}/inputs/{allele_name}.txt', 'w') as f:
                for pept in pepts:
                    f.write(f'{pept}\n')
    
    # allele files
    alleles = [f"HLA-{s.replace('*', '')}" for s in alleles]
    with open(f'{out_dir}/alleles.txt', 'w') as f:
        for allele in alleles:
            f.write(f'{allele}\n')

### prepare input files for NetMHCIIpan
# input allele: "DRB1*01:01", "DQA1*01:01_DQB1*01:01"
# rename alleles: "DRB1_0101", "HLA-DQA10101-DQB10101"
def PrepNetMHCIIpanInput(
        df: pd.DataFrame,
        allele_col: str,
        pept_cols: list,
        out_dir: str
):
    if not os.path.isdir(f'{out_dir}/inputs'): # create input dir
        os.mkdir(f'{out_dir}/inputs')
    
    alleles = list()
    for allele in df[allele_col].unique().tolist():
        allele_name = ConvertMHCII(allele) # normalized allele

        # get peptides
        pepts = list()
        for pept_col in pept_cols:
            tmp_pepts = df.loc[df[allele_col]==allele, pept_col].dropna().tolist()  # filter by alleles
            pepts += tmp_pepts
        pepts = sorted(list(set(pepts)))
        
        # save peptides
        if len(pepts) > 0:
            alleles.append(allele)
            with open(f'{out_dir}/inputs/{allele_name}.txt', 'w') as f:
                for pept in pepts:
                    f.write(f'{pept}\n')
    
    # allele files
    alleles = [ConvertMHCII(s) for s in alleles]
    with open(f'{out_dir}/alleles.txt', 'w') as f:
        for allele in alleles:
            f.write(f'{allele}\n')

### normalize MHCII allele name
# input allele: "DRB1*01:01", "DQA1*01:01_DQB1*01:01"
# rename alleles: "DRB1_0101", "HLA-DQA10101-DQB10101"
def ConvertMHCII(name):
    name = name.replace(':', '')
    if name.startswith('DRB'):
        name = name.replace('*', '_')
    else:
        name = name.replace('*', '')
        name = name.replace('_', '-')
        name = f'HLA-{name}'
    return name

### rename MHC alleles
# from netMHCpan to standard
def renameMHC(allele):
    if allele.startswith('HLA-'):
        allele = allele[4:]
    if allele[0] in ['A', 'B', 'C']:
        return allele
    elif allele.startswith('DRB'):
        gene, code = allele.split('_')
        return f'{gene}*{code[:2]}:{code[2:]}'
    else:
        a1, a2 = allele.split('-')
        return f'{a1[:4]}*{a1[4:6]}:{a1[6:]}_{a2[:4]}*{a2[4:6]}:{a2[6:]}'


######################
####### Scoring ######
######################

def LoadLukszaModel(file):
    model = json.load(open(file, 'r'))
    pos_weights = {f'P{i+1}': w for i, w in enumerate(model['d_i'])}
    sub_weights = defaultdict(dict)
    for s, w in model['M_ab'].items():
        ref, alt = s.split('->')
        sub_weights[ref][alt] = w
    return pos_weights, sub_weights

# Substitution scoring
def SubScoring(mismatch_df, pos_weights, sub_weights, specific_allele=False, specific_length=False):
    # cross reactivity object
    CR = SubCRD(pos_weights, sub_weights, specific_allele=specific_allele, specific_length=specific_length)

    # score
    scores = list()
    for idx, row in mismatch_df.iterrows():
        allele = row['allele']
        pos = int(row['pos']) - 1 # 1-base to 0-base
        ref_aa = row['WT_aa']
        alt_aa = row['MT_aa']
        scores.append(CR.score_mutation(pos, ref_aa, alt_aa, allele))

    return scores

# Plot scoring distribution
def ScoreDistributionPlot(df, x_col, hue_col, bins=20, ax=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(4,3), dpi=300)
    sns.histplot(data=df, x=x_col, hue=hue_col, stat='probability', common_norm=False, multiple='layer', kde=True, bins=bins, ax=ax)
    pval = NonParamTest(df, x_col, hue_col) # statistical test
    _ = ax.text(0.5, 0.5, f'P-value={pval:.2e}', transform=ax.transAxes)