#!/bin/bash
# Script Name: neoagfinder.sh
# Description: Run the neoantigen immunogenicity prediction
# Author: Kohan

### Input Arguments
usage() {
  cat <<EOF 
Usage: $(basename "${BASH_SOURCE[0]}") <args>

This script runs the neoantigen immunogenicity prediction

Available options:
-c  config file [REQUIRED]
-m  VEP-annotated mutation file in VCF format [REQUIRED]
-a  MHC allele file (HLA-HD final result or allele list) [REQUIRED]
-r  STAR-aligned RNA-seq BAM file [OPTIONAL]
-g  RSEM result file of gene expression [OPTIONAL]
-t  sample name used to specify the tumor sample in the VCF file (default: *tumor*) [OPTIONAL]
-o  output directory [REQUIRED]
EOF
  exit
}

while getopts "c:m:a:r:g:t:o:h" opt; do
    case "${opt}" in
        c) CONFIG=${OPTARG};;
        m) MUTATION_FILE=${OPTARG};;
        a) MHC_FILE=${OPTARG};;
        r) RNA_BAM_FILE=${OPTARG};;
        g) RNA_EXP_FILE=${OPTARG};;
        t) TUMOR_NAME=${OPTARG};;
        o) OUTDIR=${OPTARG};;
        h) usage;;
    esac
done

if [ -z "${CONFIG}" ] || [ -z "${MUTATION_FILE}" ] || [ -z "${MHC_FILE}" ] || [ -z "${OUTDIR}" ]; then
    usage
fi

if [ ! -f "${CONFIG}" ]; then
    echo "Error: The file ${CONFIG} does not exist."
    exit 1
fi

if [ ! -f "${MUTATION_FILE}" ]; then
    echo "Error: The file ${MUTATION_FILE} does not exist."
    exit 1
fi

if [ ! -f "${MHC_FILE}" ]; then
    echo "Error: The file ${MHC_FILE} does not exist."
    exit 1
fi

if [ ! -f "${RNA_BAM_FILE}" ]; then
    echo "RNA aligned BAM file isn't available"
    RNA_BAM_FILE=""
fi

if [ ! -f "${RNA_EXP_FILE}" ]; then
    echo "RNA expression file isn't available"
    RNA_EXP_FILE=""
fi

if [ ! -d "${OUTDIR}" ]; then
    mkdir -p "${OUTDIR}"
fi

echo "
#####################################
#####           Paths           #####
#####################################
"
echo "Config file: ${CONFIG}"
echo "Mutation file: ${MUTATION_FILE}"
echo "MHC allele file: ${MHC_FILE}"
echo "RNA aligned file: ${RNA_BAM_FILE}"
echo "RNA expression file: ${RNA_EXP_FILE}"
echo "Ouput directory: ${OUTDIR}"
echo

# Assign variables
source "${CONFIG}"
REAL_PATH=$(realpath $0)
SRC_DIR=$(dirname ${REAL_PATH})
SRC_DIR=${SRC_DIR}/src
SAMPLE_NAME=$(basename ${MUTATION_FILE})
SAMPLE_NAME=${SAMPLE_NAME%%.*}


### Peptide generation
echo "
#####################################
#####   1. Peptide Generation   #####
#####################################
"
mkdir -p "${OUTDIR}/peptides"

python3 "${SRC_DIR}/generate_peptides.py" \
    "${MUTATION_FILE}" \
    "${OUTDIR}/peptides/${SAMPLE_NAME}" \
    --cdna_file "${CDNA_FASTA}" \
    --cds_file "${CDS_FASTA}"
echo


### Binding prediction
echo "
#####################################
#####   2. Binding Prediction   #####
#####################################
"
mkdir -p "${OUTDIR}/mhcbinds"

# parse MHC file
echo "Parsing MHC allele file ..."
python3 "${SRC_DIR}/parse_mhc.py" \
    "${MHC_FILE}" \
    "${OUTDIR}/${SAMPLE_NAME}.mhc.txt"
MHC_FILE=${OUTDIR}/${SAMPLE_NAME}.mhc.txt
echo

# MHC-I
echo "Predicting MHC-I binding ..."
python3 "${SRC_DIR}/run_mhc_bind_pred.py" \
    "${MHC_FILE}" \
    "${OUTDIR}/peptides/${SAMPLE_NAME}.peptide.mhci.txt" \
    "${OUTDIR}/mhcbinds/${SAMPLE_NAME}" \
    --mhc_class I \
    --exec_file "${NETMHCPAN_EXEC}"
echo

# MHC-II
echo "Predicting MHC-II binding ..."
python3 "${SRC_DIR}/run_mhc_bind_pred.py" \
    "${MHC_FILE}" \
    "${OUTDIR}/peptides/${SAMPLE_NAME}.peptide.mhcii.txt" \
    "${OUTDIR}/mhcbinds/${SAMPLE_NAME}" \
    --mhc_class II \
    --exec_file "${NETMHCIIPAN_EXEC}"
echo


### Abundance annotation
echo "
#####################################
#####  3. Abundance Annotation  #####
#####################################
"
mkdir -p "${OUTDIR}/abundance"

# bam-readcount
if [ -f "${RNA_BAM_FILE}" ]; then
    echo "Running bam-readcount ..."
    bash "${SRC_DIR}/run_bam_readcount.sh" \
        "${OUTDIR}/peptides/${SAMPLE_NAME}.mut.csv" \
        "${RNA_BAM_FILE}" \
        "${OUTDIR}/abundance/${SAMPLE_NAME}" \
        "${CONFIG}"
    RNA_AF_FILE=${OUTDIR}/abundance/${SAMPLE_NAME}.readcount.parsed.tsv
else
    echo "RNA bam file isn't available. Skip bam-readcount"
    RNA_AF_FILE=""
fi
echo

# annotate abundance
echo "Annotating abundance metrics ..."
python3 "${SRC_DIR}/annotate_abundance.py" \
    "${OUTDIR}/peptides/${SAMPLE_NAME}.mut.csv" \
    "${OUTDIR}/abundance/${SAMPLE_NAME}.mut.abd.csv" \
    --tumor_colname "${TUMOR_NAME}" \
    --rna_rsem_file "${RNA_EXP_FILE}" \
    --rna_af_file "${RNA_AF_FILE}"
echo


### Functional measurement
echo "
#####################################
##### 4. Functional Measurement #####
#####################################
"
mkdir -p "${OUTDIR}/metrics"

echo "Updating reference ..."
python3 "${SRC_DIR}/CRD/update_reference.py" \
    "${SRC_DIR}/CRD/ref.h5" \
    "${MHC_FILE}" \
    --mhci_pred_exec "${NETMHCPAN_EXEC}" \
    --mhcii_pred_exec "${NETMHCIIPAN_EXEC}" \
    --mhci_peptide_file "${SRC_DIR}/CRD/mhci_random_peptides.txt" \
    --mhcii_peptide_file "${SRC_DIR}/CRD/mhcii_random_peptides.txt"
echo

echo "Calculating metrics ..."
python3 "${SRC_DIR}/calculate_metrics.py" \
    "${MHC_FILE}" \
    "${OUTDIR}/abundance/${SAMPLE_NAME}.mut.abd.csv" \
    "${OUTDIR}/peptides/${SAMPLE_NAME}.peptide" \
    "${OUTDIR}/metrics/${SAMPLE_NAME}" \
    --mhci_pred_file "${OUTDIR}/mhcbinds/${SAMPLE_NAME}.pred.mhci.csv" \
    --mhcii_pred_file "${OUTDIR}/mhcbinds/${SAMPLE_NAME}.pred.mhcii.csv"

mv ${OUTDIR}/metrics/${SAMPLE_NAME}.metrics.csv ${OUTDIR}/${SAMPLE_NAME}.neoantigen.csv