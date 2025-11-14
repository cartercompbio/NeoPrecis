"""
NeoPrecis: Neoantigen prediction and analysis toolkit

A comprehensive toolkit for neoantigen prediction, peptide generation,
MHC binding prediction, and immunogenicity scoring.
"""

__version__ = '1.4.0'
__author__ = 'Kohan'

# Import main API classes for direct access
from .api import (
    # Classes
    Blosum62,
    MHC,
    PepGen,
    EpiGen,
    BestEpi,
    EpiMetrics,

    # I/O functions
    ReadVCF,
    ReadMAF,
    ReadVEPannotTXT,
    ReadTranscriptFASTA,
    WritePeptideTXT,

    # Prediction functions
    ReadNetMHCpan,
    ReadMixMHCpred,
    RunNetMHCpan,
    RunMixMHCpred,

    # Utility functions
    MHCIAlleleTransform,
    MHCIIAlleleTransform,
    LoadAllowedAlleles,
)

# Core CRD models
from .CRD import SubCRD, PeptCRD

# Export commonly used classes and functions
__all__ = [
    # Version info
    '__version__',
    '__author__',

    # CRD models
    'SubCRD',
    'PeptCRD',

    # Main API classes
    'MHC',
    'PepGen',
    'EpiGen',
    'BestEpi',
    'EpiMetrics',
    'Blosum62',

    # I/O functions
    'ReadVCF',
    'ReadMAF',
    'ReadVEPannotTXT',
    'ReadTranscriptFASTA',
    'WritePeptideTXT',

    # Prediction functions
    'ReadNetMHCpan',
    'ReadMixMHCpred',
    'RunNetMHCpan',
    'RunMixMHCpred',

    # Utility functions
    'MHCIAlleleTransform',
    'MHCIIAlleleTransform',
    'LoadAllowedAlleles',
]
