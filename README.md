# NeoPrecis
NeoPrecis is an advanced tool designed to enhance cancer immunotherapy by accurately predicting neoantigens.


## Overview

The immune system plays a crucial role in combating tumors by recognizing tumor-specific antigens. Among these, neoantigens—derived from somatic mutations—represent a key category due to their tumor-specific nature and high immunogenic potential. However, identifying immunogenic neoantigens remains a significant challenge.

NeoPrecis addresses this challenge through two key functionalities:
1. **Neoantigen Immunogenicity Prediction**: Enables the identification of neoantigens for designing cancer vaccines.
2. **Neoantigen Landscape Construction**: Facilitates the prediction of immune checkpoint inhibitor response by mapping the overall neoantigen profile.


## Setup

### External Tools
Please install the following external tools:
- [NetMHCpan-4.0](https://services.healthtech.dtu.dk/services/NetMHCpan-4.0/): Predicts peptide binding to MHC-I molecules.
- [NetMHCIIpan-4.3](https://services.healthtech.dtu.dk/services/NetMHCIIpan-4.3/): Predicts peptide binding to MHC-II molecules.
- [bam-readcount](https://github.com/genome/bam-readcount): Calculates base-level allele frequencies from BAM files.
- [PyClone](https://github.com/Roth-Lab/pyclone): Identifies clonal and subclonal tumor populations for clonality analysis.

### Reference Files
Please download the following reference files:
- Ensembl CDS and cDNA reference sequences (FASTA)
  - Ensembl FASTA files are available at: [https://ftp.ensembl.org/pub/](https://ftp.ensembl.org/pub/)
  - Ensure the Ensembl version matches the VEP version used. For example, if using GRCh38 v113:
    - `https://ftp.ensembl.org/pub/release-113/fasta/homo_sapiens/cdna/Homo_sapiens.GRCh38.cdna.all.fa.gz`
    - `https://ftp.ensembl.org/pub/release-113/fasta/homo_sapiens/cds/Homo_sapiens.GRCh38.cds.all.fa.gz`
- The genome reference file used for RNA-seq alignment with STAR.

### Required Packages
To install the necessary dependencies, you can use one of the following methods:
- Python Required Packages: Install the packages listed in `setup/requirements.txt` using `pip`.
  ```bash
  pip3 install -r setup/requirements.txt
  ```
- Conda Environment: Create a Conda environment using the file `setup/env.yml`
  ```bash
  conda env create -f setup/env.yml
  conda activate neoprecis
  ```


## Prediction of Immunogenic Neoantigens

### Input Preparation
To predict immunogenic neoantigens, the following inputs are required:

- Somatic Mutations
  - A VEP-annotated `VCF` file (`examples/case.vep.vcf`)
  - Ensure the following VEP options are enabled: --symbol, --hgvs, --canonical
  - Record the VEP version used, as it is required for subsequent analyses
- MHC Alleles
  - An allele list file (`examples/case.mhc.txt`) or an HLA-HD output file (`examples/case_final.result.txt`)
  - It is recommended to use the default IMGT-HLA version provided by HLA-HD to ensure compatibility with MHC-binding predictors. Updated IMGT-HLA versions may include rare alleles that are not supported by all predictors
- RNA Expression (Optional)
  - A STAR-aligned `BAM` file (`examples/caseAligned.sortedByCoord.out.bam`) and an RSEM result file (`examples/case.genes.results`)
  - These files are used to calculate RNA expression levels and RNA allele frequencies, providing additional insights for neoantigen prioritization

### Usage
To run the script, first prepare a `config.conf` file to define the paths to external tools and reference files. An example configuration file can be found at `config.conf`. Then, execute the command as follows:
```bash
bash neoagfinder.sh \
  -c config.conf \
  -m examples/case.vep.vcf \
  -a examples/case.mhc.txt \
  -r examples/caseAligned.sortedByCoord.out.bam \
  -g examples/examples/case.genes.results \
  -t case_tumor \
  -o outdir
```

Description of arguments:
- `-c config.conf`: Configuration file
- `-m examples/case.vep.vcf`: VEP-annotated VCF file
- `-a examples/case.mhc.txt`: Input file listing MHC alleles
- `-r examples/caseAligned.sortedByCoord.out.bam` (optional): STAR-aligned RNA-seq BAM file
- `-g examples/case.genes.results` (optional): RSEM result file of gene expression
- `-t case_tumor` (optional): Sample identifier used to specify the tumor sample in the VCF file. If not provided, the script will default to capturing a name containing the word "tumor".
- `-o outdir`: Output folder

### Interpretation
The output folder contains a main output file, `outdir/neoags.csv`, along with intermediate files generated during the analysis.
This main output file includes the following columns:
- Coordinates: Genomic coordinates of the mutation
- Annotations: Fields annotated by VEP
- Immunogenicity-related metrics
  - DNA_AF: DNA allele frequency
  - RNA_AF: RNA allele frequency
  - RNA_EXP: RNA expression level (in TPM)
  - RNA_EXP_QTL: RNA expression level quantile (1 = minimal, 4 = maximal)
  - Robustness-{MHC}: Number of alleles presenting the neoantigen
  - PHBR-{MHC}: Patient harmonic-mean best rank for evaluating MHC-binding affinity (range: 0–100; lower values indicate stronger binding)
  - CRD-{MHC}: Cross-reactivity distance for assessing TCR recognition (higher values indicate stronger recognition)
  - Agretopicity-{MHC}: Log-scale binding score ratio between mutated and wild-type peptides (>0 indicates stronger binding of the mutated peptide)
- Immunogenicity-{MHC}: Overall immunogenicity score

Note: {MHC} in the column names refers to the MHC class type, either I or II. Metrics are calculated separately for MHC class I and class II alleles.

## Construction of Neoantigen Landscape

### Input Preparation

### Usage

### Interpretation
