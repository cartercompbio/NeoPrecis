#!/bin/python3
# Script Name: clean_columns.py
# Description: Remove unnecessary columns and rename columns
# Author: Kohan

import sys
import pandas as pd

file = sys.argv[1]
df = pd.read_csv(file)

# drop SubCRD, PeptCRD
df = df.drop(columns=['SubCRD-I', 'SubCRD-II', 'PeptCRD-I', 'PeptCRD-II'])

# rename Immgen to NP-Immuno
df = df.rename(columns={'Immgen-I': 'NP-Immuno-I', 'Immgen-II': 'NP-Immuno-II'})

# save
df.to_csv(file, index=False)