"""Allele names: two fields with the allele's expression letter, each allele named as the predictor
lists it, and predictions named back by the sample's own alleles. Run from the repository root,
with pytest installed (setup/ does not install it): python -m pytest tests"""

from pathlib import Path

import pandas as pd
import pytest

from neoprecis import run_mhc_bind_pred as step
from neoprecis.api import MHC
from neoprecis.CRD.update_reference import GetNewAlleles
from neoprecis.parse_mhc import ParseHLAHD

CLASS_I = ["HLA-A*01:01", "HLA-A*02:01", "HLA-A*30:14L"]  # netMHCpan lists A*30:14L only with its L
CLASS_II = ["DRB1_0101", "DRB5_0108N", "HLA-DQA10107-DQB10602", "HLA-DQA10501-DQB10602",
            "HLA-DPA10103-DPB110401"]  # DRB5_0108N: the one null allele netMHCIIpan 4.3 lists


def mhc(tmp_path, *alleles):
    path = tmp_path / "s.mhc.txt"
    path.write_text("".join(f"{allele}\n" for allele in alleles))
    return MHC(str(path))


@pytest.fixture
def predictors(tmp_path):
    """netMHCpan and netMHCIIpan executables whose data/ lists name the alleles above."""
    (tmp_path / "netMHCpan" / "data").mkdir(parents=True)
    (tmp_path / "netMHCpan" / "data" / "allelenames").write_text(
        "".join(f"{name.replace('*', '')} {name}\n" for name in CLASS_I))
    (tmp_path / "netMHCIIpan" / "data").mkdir(parents=True)
    (tmp_path / "netMHCIIpan" / "data" / "allelelist.txt").write_text("".join(f"{name} {name}\n" for name in CLASS_II))
    return str(tmp_path / "netMHCpan" / "netMHCpan"), str(tmp_path / "netMHCIIpan" / "netMHCIIpan")


def test_hlahd_reads_two_fields_and_the_alleles_letter(tmp_path):
    path = tmp_path / "s_final.result.txt"
    path.write_text("A\tHLA-A*02:01:14Q\tHLA-A*30:14L\nDQB1\tHLA-DQB1*06:02:01\tHLA-DQB1*02:20:01:01N\n")
    assert ParseHLAHD(str(path), genes=["A", "DQB1"]) == {
        "A_1": "A*02:01Q", "A_2": "A*30:14L", "DQB1_1": "DQB1*06:02", "DQB1_2": "DQB1*02:20N"}


def test_an_allele_is_named_as_the_predictor_lists_it(tmp_path):
    sample = mhc(tmp_path, "A*02:01:14Q", "A*30:14L", "A*01:01:01:02N", "DRB1*01:01N", "DRB5*01:08:01N",
                 "DQA1*01:07Q", "DQA1*05:01:01:02N", "DQB1*06:02")
    # bare name first, then with the letter; a null allele only under its N name, even where the
    # predictor lists its bare name (HLA-A*01:01, DRB1_0101, the DQA10501 pair)
    assert sample.predictable("i", "netmhcpan", CLASS_I) == [("HLA-A*02:01", "A*02:01"), ("HLA-A*30:14L", "A*30:14")]
    assert sample.predictable("ii", "netmhcpan", CLASS_II) == [("DRB5_0108N", "DRB5*01:08N"),
                                                               ("HLA-DQA10107-DQB10602", "DQA1*01:07_DQB1*06:02")]
    assert sample.name_dict["MHC-I"]["standard"] == ["A*02:01", "A*30:14", "A*01:01N"]
    assert sample.name_dict["MHC-II"]["standard"] == ["DRB1*01:01N", "DRB5*01:08N", "DQA1*01:07_DQB1*06:02",
                                                     "DQA1*05:01N_DQB1*06:02"]
    assert sample.name_dict["MHC-I"]["mixmhcpred"] == [["A0201", "A0201Q"], ["A3014", "A3014L"], ["A0101N"]]


def test_new_alleles_skip_those_in_the_reference(tmp_path, predictors):
    mhci, mhcii = predictors
    sample = mhc(tmp_path, "A*02:01", "A*30:14L", "DQA1*01:07Q", "DQB1*06:02", "DPA1*01:03", "DPB1*104:01")
    assert GetNewAlleles(sample, ["A*02:01"], "i", "netmhcpan", mhci) == (["HLA-A*30:14L"], ["A*30:14"])
    assert GetNewAlleles(sample, ["DQA1*01:07_DQB1*06:02"], "ii", "netmhcpan", mhcii) == (
        ["HLA-DPA10103-DPB110401"], ["DPA1*01:03_DPB1*104:01"])


@pytest.mark.parametrize("mhc_class, alleles, expected", [
    ("I", ["A*30:14L", "A*02:01"], ["A*02:01", "A*30:14"]),
    ("II", ["DQA1*01:07Q", "DQB1*06:02"], ["DQA1*01:07_DQB1*06:02"]),
])
def test_binding_step_names_predictions_by_the_samples_alleles(tmp_path, predictors, monkeypatch,
                                                               mhc_class, alleles, expected):
    mhc(tmp_path, *alleles)
    (tmp_path / "p.txt").write_text("PEPTIDEAAAAAAAA\n")

    def fake_netmhc(names, peptide_file, out_file, exe_path):  # one row per allele, named as given
        rows = "".join(f" {i} {name} PEPTIDEAAAAAAAA 0 PEPTIDEAA 1.0 0.5\n" for i, name in enumerate(names.split(",")))
        Path(out_file).write_text(f"-----\n Pos MHC Peptide Of Core %Rank_EL Score_EL\n-----\n{rows}-----\n")

    monkeypatch.setattr(step, "RunNetMHCpan", fake_netmhc)
    exe = predictors[0] if mhc_class == "I" else predictors[1]
    args = step.ArgumentParser().parse_args([str(tmp_path / "s.mhc.txt"), str(tmp_path / "p.txt"),
                                             str(tmp_path / "s"), "--mhc_class", mhc_class, "--exec_file", exe])
    step.Main(args)
    predictions = pd.read_csv(tmp_path / f"s.pred.mhc{mhc_class.lower()}.csv")
    assert sorted(predictions["MHC"]) == expected
    # nothing to predict is not an error: a rerun, then no allele the predictor lists, then no peptide
    for listed, peptides in ((alleles, "PEPTIDEAAAAAAAA\n"), (["A*99:99"], "PEPTIDEAAAAAAAA\n"), (alleles, "")):
        mhc(tmp_path, *listed)
        (tmp_path / "p.txt").write_text(peptides)
        with pytest.raises(SystemExit) as skip:
            step.Main(args)
        assert skip.value.code == 0
