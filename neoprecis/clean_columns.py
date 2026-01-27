#!/bin/python3
# Script Name: clean_columns.py
# Description: Remove unnecessary columns and rename columns
# Author: Kohan

import sys
import pandas as pd

file = sys.argv[1]
df = pd.read_csv(file)

for mhc in ['I', 'II']:
    # drop SubCRD
    if f'SubCRD-{mhc}' in df.columns:
        df = df.drop(columns=[f'SubCRD-{mhc}'])
    # rename Immgen to NP-Immuno
    if f'Immgen-{mhc}' in df.columns:
        df = df.rename(columns={f'Immgen-{mhc}': f'NP-Immuno-{mhc}'})

# save
df.to_csv(file, index=False)