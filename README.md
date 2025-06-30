# NeoPrecis
NeoPrecis is an computational framework designed to advance personalized cancer immunotherapy by assessing neoantigen immunogenicity.


## Overview

The immune system plays a crucial role in combating tumors by recognizing tumor-specific antigens. Among these, neoantigens—derived from somatic mutations—represent a key category due to their tumor-specific nature and high immunogenic potential. However, identifying immunogenic neoantigens and evaluating individual neoantigen landscape remain significant challenges.

NeoPrecis addresses this challenge through two key functionalities:
1. **Neoantigen Immunogenicity Prediction**: Enables the identification of immunogenic neoantigens for designing cancer vaccines.
2. **Neoantigen Landscape Construction**: Facilitates the prediction of immune checkpoint inhibitor response by mapping the overall neoantigen profile.


## Setup

### External Tools
Please install the following external tools:
- [NetMHCpan-4.1](https://services.healthtech.dtu.dk/services/NetMHCpan-4.1/): Predicts peptide binding to MHC-I molecules.
- [NetMHCIIpan-4.3](https://services.healthtech.dtu.dk/services/NetMHCIIpan-4.3/): Predicts peptide binding to MHC-II molecules.
- [bam-readcount](https://github.com/genome/bam-readcount): Calculates base-level allele frequencies from BAM files.

### Reference Files
Please download the following reference files:
- Ensembl CDS and cDNA reference sequences (FASTA)
  - Ensembl FASTA files are available at: [https://ftp.ensembl.org/pub/](https://ftp.ensembl.org/pub/)
  - Ensure the Ensembl version matches the VEP version used. For example, if using GRCh38 v113:
    - [cDNA](https://ftp.ensembl.org/pub/release-113/fasta/homo_sapiens/cdna/Homo_sapiens.GRCh38.cdna.all.fa.gz) `https://ftp.ensembl.org/pub/release-113/fasta/homo_sapiens/cdna/Homo_sapiens.GRCh38.cdna.all.fa.gz`
    - [CDS](https://ftp.ensembl.org/pub/release-113/fasta/homo_sapiens/cds/Homo_sapiens.GRCh38.cds.all.fa.gz) `https://ftp.ensembl.org/pub/release-113/fasta/homo_sapiens/cds/Homo_sapiens.GRCh38.cds.all.fa.gz`
- The genome reference file used for RNA-seq alignment with STAR.

### Required Packages
To install the necessary dependencies, choose one of the following methods. Installation should take no more than 10 minutes.
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
  - A VEP-annotated `VCF` file (e.g. `examples/case.vep.vcf`)
  - Ensure the following VEP options are enabled: `--symbol`, `--hgvs`, `--canonical`
  - Record the VEP version used, as it is required for subsequent analyses
- MHC Alleles
  - An allele list file (e.g. `examples/case.mhc.txt`) or an HLA-HD output file (e.g. `examples/case_final.result.txt`)
  - It is recommended to use the default IMGT-HLA version provided by HLA-HD to ensure compatibility with MHC-binding predictors. Updated IMGT-HLA versions may include rare alleles that are not supported by all predictors.
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
  -o examples
```

Description of arguments:
- `-c`: Configuration file
- `-m`: VEP-annotated VCF file
- `-a`: Input file listing MHC alleles
- `-r` (optional): STAR-aligned RNA-seq BAM file
- `-g` (optional): RSEM result file of gene expression
- `-t` (optional): Sample identifier used to specify the tumor sample in the VCF file. If not provided, the script will default to capturing a name containing the substring "tumor".
- `-o`: Output folder

### Interpretation
The output folder contains a main output file, `${outdir}/${basename}.neoantigen.csv` (e.g. `examples/case.neoantigen.csv`), along with intermediate files generated during the analysis. This main output file includes the following columns:
- Coordinates: Genomic coordinates of the mutation
- Annotations: Fields annotated by VEP
- Immunogenicity-related metrics
  | Metric | Description |
  | --- | --- |
  | DNA_AF | DNA allele frequency |
  | RNA_AF | RNA allele frequency |
  | RNA_DEPTH | RNA read coverage |
  | RNA_EXP | RNA expression level (in TPM) |
  | RNA_EXP_QRT | RNA expression level quartile (1 = minimal, 4 = maximal) |
  | Robustness-{MHC} | Number of alleles presenting the neoantigen |
  | PHBR-{MHC} | Patient harmonic-mean best rank for evaluating MHC-binding affinity (range: 0–100; lower values indicate stronger binding) |
  | Agretopicity-{MHC} | Log-scale binding score ratio between mutated and wild-type peptides (>0 indicates stronger binding of the mutated peptide) |
  | NP-Immuno-{MHC} | Immunogenicity prediction focusing on T-cell recognition |
  | NP-Integrated-{MHC} | Immunogenicity prediction considering antigen abundance, MHC presentation, and T-cell recognition (only applied if RNA is available) |

**Notes**
- {MHC} in the column names refers to the MHC class type, either I or II. Metrics are calculated separately for MHC class I and class II alleles.
- NP-Immuno is trained on TCR-pMHC binding data from [IEDB](https://www.iedb.org/) and [VDJdb](https://vdjdb.cdr3.net/), as well as T-cell activation assay data from [CEDAR](https://cedar.iedb.org/).
- NP-Integrated is a logistic regression model incorporating five features–DNA_AF, RNA_AF, RNA_EXP_QTL, PHBR, and NP-Immuno–trained on the [NCI GI cancer cohort](https://www.ncbi.nlm.nih.gov/projects/gap/cgi-bin/study.cgi?study_id=phs001003.v3.p1).


## Construction of Neoantigen Landscape

### Input Preparation
To construct the neoantigen landscape, the following inputs are required:

- Neoantigen Predictions: Perform neoantigen immunogenicity predictions using the provided pipeline
- Tumor Clonality Analysis
  - Provide a cluster TSV file (`examples/cluster.tsv`) and a loci TSV file (`examples/loci.tsv`) generated by [PyClone](https://github.com/Roth-Lab/pyclone)
  - An example script for running PyClone is available at `src/run_pyclone`
  - To accelerate the process, the `--max_clusters` parameter can be set to 100

### Usage

```bash
bash neoagland.sh \
  -i examples/case.neoantigen.csv \
  -c examples/case.cluster.tsv \
  -l examples/case.loci.tsv \
  -o examples/case.landscape.csv
```

Description of arguments:
- `-i`: Neoantigen prediction file
- `-c`: Cluster file generated by PyClone
- `-l`: Loci file generated by PyClone
- `-o`: Output path

### Interpretation
The output file provides tumor-centric scores, aggregated from mutation-centric scores based on immunogenicity predictions.

Each row represents a specific metric:

| Metric | Description |
| --- | --- |
| TMB | Tumor mutation burden (non-synonymous mutations) |
| TNB | Tumor neoantigen burden (PHBR-I ≤ 2) |
| NPB | NeoPrecis burden (product of NP-Immuno-I and NP-Immuno-II ≥ 0.16)
| NP-LandscapeSum | Sum of NP-Immuno scores across all neoantigens |
| NP-LandscapeCCF | Weighted sum of NP-Immuno scores across all neoantigens, adjusted by the cancer cell fraction (CCF) of mutations |
| NP-LandscapeClone | Aggregated NP-Immuno scores across all neoantigens based on mutation clusters |

Each column represents a score scale:

| Scale | Description |
| --- | --- |
| value | Raw metric value |
| percentile_melanoma | normalized percentile within the melanoma cohort (n=277) |
| percentile_NSCLC | normalized percentile within the NSCLC cohort (n=248) |

**Notes**
- Normalized percentile (between 0 to 1) within the melanoma and NSCLC cohorts are provided to contextualize the metric values and enhance interpretability.


## Analyses for the Manuscript
The analyses for the manuscript are detailed in the `manuscript` directory. Please refer to its contents for reproducing the results presented in our manuscript.


## Reference
- netMHCpan-4.1: Reynisson, B., Alvarez, B., Paul, S., Peters, B. & Nielsen, M. NetMHCpan-4.1 and NetMHCIIpan-4.0: improved predictions of MHC antigen presentation by concurrent motif deconvolution and integration of MS MHC eluted ligand data. Nucleic Acids Res. 48, 449–454 (2020).
- NetMHCIIpan-4.3: Nilsson, J. B. et al. Accurate prediction of HLA class II antigen presentation across all loci using tailored data acquisition and refined machine learning. Sci. Adv. 9, eadj6367 (2023).
- bam-readcount: Khanna, A. et al. Bam-readcount - rapid generation of basepair-resolution sequence metrics. J. Open Source Softw. 7, 3722 (2022).
- PyClone: Roth, A. et al. PyClone: statistical inference of clonal population structure in cancer. Nat. Methods 11, 396–398 (2014).
- NCI cohort: Parkhurst, M. R. et al. Unique Neoantigens Arise from Somatic Mutations in Patients with Gastrointestinal Cancers. Cancer Discov. 9, 1022–1035 (2019).
- IEDB: Vita, R. et al. The Immune Epitope Database (IEDB): 2018 update. Nucleic Acids Res. 47, D339–D343 (2019).
- VDJdb: Goncharov, M. et al. VDJdb in the pandemic era: a compendium of T cell receptors specific for SARS-CoV-2. Nat. Methods 19, 1017–1019 (2022).
- CEDAR: Koşaloğlu-Yalçın, Z. et al. The Cancer Epitope Database and Analysis Resource (CEDAR). Nucleic Acids Res. 51, D845–D852 (2023).

**Please refer to the NeoPrecis manuscript for the complete list of references**

## Citation
