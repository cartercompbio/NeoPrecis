"""PHBR: the harmonic mean of the best ranks of the sample's alleles with predictions, binders or
not; an allele without predictions, and a locus not typed, are left out. Run from the repository
root, with pytest installed (setup/ does not install it): python -m pytest tests"""

import numpy as np
import pandas as pd
import pytest

from neoprecis.api import BestEpi, EpiMetrics

PEPTIDE = "AAAAAAAAA"
# A sample with HLA-C not typed: 4 of the 6 class I alleles, each with its rank for PEPTIDE
NO_C = {"A*01:01": 0.5, "A*02:01": 1.0, "B*07:02": 1.5, "B*08:01": 2.0}
# A sample with DP not typed: 6 of the 10 class II entries, the DRB1 alleles and the four DQ pairs
NO_DP = {"DRB1*03:01": 1.5, "DRB1*15:01": 3.0,
         "DQA1*01:02_DQB1*06:02": 5.0, "DQA1*01:02_DQB1*02:01": 7.0,
         "DQA1*05:01_DQB1*06:02": 9.0, "DQA1*05:01_DQB1*02:01": 10.0}


@pytest.fixture(scope="module")
def class_i():
    return EpiMetrics("i", None)


@pytest.fixture(scope="module")
def class_ii():
    return EpiMetrics("ii", None)


def phbr(epi_metrics, alleles, ranks, bind_threshold=2):
    """PHBR of a mutation with one peptide, given the sample's alleles and the peptide's rank for
    each allele with predictions."""
    binds = pd.DataFrame([(PEPTIDE, allele, PEPTIDE, rank, 0.5) for allele, rank in ranks.items()],
                         columns=["Peptide", "MHC", "Core", "Rank", "Score"])
    peptides = pd.DataFrame({"MT_epitope": [PEPTIDE], "WT_epitope_left_aligned": [np.nan],
                             "WT_epitope_right_aligned": [np.nan], "WT_epitope_both_aligned": [np.nan]})
    epi_metrics.alleles = alleles
    best = BestEpi(alleles, binds)(peptides)
    return epi_metrics(best, bind_threshold, metrics=["PHBR"])[1]["PHBR"]


def test_a_locus_not_typed_is_left_out(class_i):
    phbr_i = phbr(class_i, list(NO_C), NO_C)
    assert phbr_i == pytest.approx(4 / (1 / 0.501 + 1 / 1.001 + 1 / 1.501 + 1 / 2.001))


def test_an_allele_without_predictions_is_left_out(class_i):
    ranks = {**NO_C, "C*07:01": 1.8}  # C*07:02N is a null allele, which no predictor scores
    phbr_i = phbr(class_i, list(ranks) + ["C*07:02N"], ranks)
    assert phbr_i == pytest.approx(5 / (1 / 0.501 + 1 / 1.001 + 1 / 1.501 + 1 / 2.001 + 1 / 1.801))


def test_a_predicted_non_binder_counts(class_i):
    ranks = {**NO_C, "C*07:01": 4.0, "C*07:02": 80.0}  # 4 and 80 are above the threshold
    phbr_i = phbr(class_i, list(ranks), ranks)
    assert phbr_i == pytest.approx(
        6 / (1 / 0.501 + 1 / 1.001 + 1 / 1.501 + 1 / 2.001 + 1 / 4.001 + 1 / 80.001))


def test_no_allele_with_predictions_gives_no_phbr(class_i):
    assert np.isnan(phbr(class_i, list(NO_C), {}))


def test_class_ii_leaves_out_a_locus_not_typed(class_ii):
    phbr_ii = phbr(class_ii, list(NO_DP), NO_DP, bind_threshold=10)
    assert phbr_ii == pytest.approx(
        6 / (1 / 1.501 + 1 / 3.001 + 1 / 5.001 + 1 / 7.001 + 1 / 9.001 + 1 / 10.001))
