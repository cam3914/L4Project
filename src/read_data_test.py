from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize
import numpy as np
import pandas as pd
from pyteomics import mgf
import pytest

mgf_data = mgf.read("../data/raw/example.mgf")
data = pd.DataFrame(mgf_data)

class TestMgf:
    def test_one(self):
        assert data.count()[0] == 5

    def test_two(self):
        assert data.iloc[0]['params']['title'] == 'scan=986 profile data'

class TestRdkit:
    def test_one(self):
        assert Chem.MolStandardize.rdMolStandardize.ValidateSmiles("O[C@H]1C2=C(C=C(OC)C(OC)=C2)C3=C([C@]1([H])[N+](C)([O-])CC4)C4=CC5=C3OCO5") == []

    def test_two(self):
        with pytest.raises(ValueError):
            Chem.MolStandardize.rdMolStandardize.ValidateSmiles("N/A") 