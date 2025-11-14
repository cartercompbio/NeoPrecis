"""
CRD: Cross-Reactivity Distance models for neoantigen prediction

This module provides models for calculating cross-reactivity distance
between peptides based on substitution patterns and positional factors.
"""

from .CRD import SubCRD, PeptCRD

__all__ = ['SubCRD', 'PeptCRD']
__version__ = '1.4.0'
