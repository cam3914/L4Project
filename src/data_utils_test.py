from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize
import numpy as np
import pandas as pd
from pyteomics import mgf
import pytest
from data_utils import *

melatonin_smiles = 'CC(=O)NCCC1=CNc2c1cc(OC)cc2CC(=O)NCCc1c[nH]c2ccc(OC)cc12'

class TestMgf:
    data = pd.DataFrame(mgf.read("../data/raw/example.mgf"))
    def test_one(self):
        assert self.data.count()[0] == 5

    def test_two(self):
        assert self.data.iloc[0]['params']['title'] == 'scan=986 profile data'

class TestRdkit:
    def test_one(self):
        assert Chem.MolStandardize.rdMolStandardize.ValidateSmiles(melatonin_smiles) == []

    def test_two(self):
        with pytest.raises(ValueError):
            Chem.MolStandardize.rdMolStandardize.ValidateSmiles("N/A")

    def test_three(self):
        with pytest.raises(ValueError):
            Chem.MolStandardize.rdMolStandardize.ValidateSmiles(melatonin_smiles[:-1])

class TestGetFingerprint:
    def test_one(self):
        assert len(getFingerprint(melatonin_smiles)) == 2048

    def test_two(self):
        with pytest.raises(Exception):
            getFingerprint(melatonin_smiles[:-1])
        
class TestNormalize:
    def test_one(self):
        test_array = np.array([ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        normalized = normalize(test_array)
        assert normalized[0] == 0.0 and normalized[10] == 1.0

class TestBinning:
    data = pd.DataFrame({
            'intensity': np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0]),
            'm/z': np.array([0.8, 1.2, 5.0, 8.0, 100.0, 500.0])
        })
    
    def test_one(self):
        binned = bin_spectra(self.data, 1)
        assert binned[1] == 0.5 and binned[500] == 1.0

    def test_two(self):
        binned = bin_spectra(self.data, 10)
        assert binned[1] == 1.0 and binned[10] == 0.5
