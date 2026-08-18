#!/bin/bash
# Script Name: run_bam_readcount.sh
# Description: Run the bam-readcount for RNA AF annotation
# Author: Kohan


source ~/.bashrc
mamba activate base

MUT_FILE=$1
BAM_FILE=$2
OUT_BASENAME=$3
CONFIG=$4

source ${CONFIG}

# generate site file
# Look the columns up by name: generate_peptides.py writes an unnamed pandas
# index plus original_idx ahead of #CHROM/POS, so fixed positions drift.
awk -F, 'BEGIN { OFS="\t" }
    NR == 1 { for (i = 1; i <= NF; i++) { if ($i == "#CHROM") c = i; if ($i == "POS") p = i }
              if (!c || !p) { print "ERROR: #CHROM/POS not found in " FILENAME > "/dev/stderr"; exit 1 }
              next }
    { print $c, $p, $p }' ${MUT_FILE} > ${OUT_BASENAME}.site.tsv
#awk -F, 'BEGIN { OFS="\t" } NR > 1 { max_value = ($3 > $3 + length($4) - 1) ? $3 : $3 + length($4) - 1; print $2, $3, max_value }' ${MUT_FILE} > ${OUT_BASENAME}.site.tsv

# bam-readcount
${BAM_READCOUNT_DIR}/build/bin/bam-readcount \
    -l ${OUT_BASENAME}.site.tsv \
    -f ${GENOME} \
    -w1 \
    ${BAM_FILE} > ${OUT_BASENAME}.readcount.tsv

# parse output
python3 ${BAM_READCOUNT_DIR}/tutorial/scripts/parse_brc.py \
    ${OUT_BASENAME}.readcount.tsv > ${OUT_BASENAME}.readcount.parsed.tsv