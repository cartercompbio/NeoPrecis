# NeoPrecis
NeoPrecis is an advanced tool designed to enhance cancer immunotherapy by accurately predicting neoantigens.


## Overview

The immune system plays a crucial role in combating tumors by recognizing tumor-specific antigens. Among these, neoantigens—derived from somatic mutations—represent a key category due to their tumor-specific nature and high immunogenic potential. However, identifying immunogenic neoantigens remains a significant challenge.

NeoPrecis addresses this challenge through two key functionalities:
1. **Neoantigen Immunogenicity Prediction**: Enables the identification of neoantigens for designing cancer vaccines.
2. **Neoantigen Landscape Construction**: Facilitates the prediction of immune checkpoint inhibitor response by mapping the overall neoantigen profile.


## Setup

### External Tools
#### MHC-Binding Prediction
- [NetMHCpan-4.0](https://services.healthtech.dtu.dk/services/NetMHCpan-4.0/): Used for predicting peptide binding to MHC-I molecule
- [NetMHCIIpan-4.3](https://services.healthtech.dtu.dk/services/NetMHCIIpan-4.3/): Used for predicting peptide binding to MHC-II molecule

#### Allele frequency calculation
- [bam-readcount](https://github.com/genome/bam-readcount): Computes base-level allele frequencies for BAM files

#### Tumor clonality analysis
- [PyClone](https://github.com/Roth-Lab/pyclone): Performs tumor clonality analysis to identify clonal and subclonal populations

### Reference Files
#### Ensembl Reference

#### Genome Reference


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

#### Somatic Mutations
- A VEP-annotated `VCF` file (`examples/case.vep.vcf`)
- Ensure the following VEP options are enabled: --symbol, --hgvs, --canonical
- Record the VEP version used, as it is required for subsequent analyses

#### MHC Alleles
- An allele list file (`examples/case.mhc.txt`) or an HLA-HD output file (`examples/case_final.result.txt`)
- It is recommended to use the default IMGT-HLA version provided by HLA-HD to ensure compatibility with MHC-binding predictors. Updated IMGT-HLA versions may include rare alleles that are not supported by all predictors

#### RNA Expression (Optional)
- An aligned `BAM` file (`examples/caseAligned.sortedByCoord.out.bam`) and an RSEM result file (`examples/case.genes.results`)
- These files are used to calculate RNA expression levels and RNA allele frequencies, providing additional insights for neoantigen prioritization


### Usage

### Interpretation


## Construction of Neoantigen Landscape

### Input Preparation

### Usage

### Interpretation
