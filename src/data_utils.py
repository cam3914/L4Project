from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
import pandas as pd
from pyteomics import mgf


def getData(filename):
    """ Use pyteomics to read in data from /data/raw/ folder, check SMILES strung and convert to dataframe

        Args: filename as string
    """
    raw = mgf.read("../data/raw/" + filename)
    
    print("num molecules in raw file: ", len(raw))
    # create dataframe with name and SMILES string

    mols = []

    for i, mol in enumerate(raw):  
        smiles = mol['params']['smiles']

        if len(smiles) > 0 or smiles != "N/A":


            molecule = {'Name': mol['params']['name'],
                        'title': mol['params']['title'],
                        'SMILES': mol['params']['smiles'],
                        'pepmass': mol['params']['pepmass'],
                        'm/z': mol['m/z array'],
                        'intensity': mol['intensity array'],
                        'charge array': mol['charge array'],
                        'charge': mol['params']['charge'],
                        #'fingerprint': getFingerprint(mol['params']['smiles'])
                        }

            try:
                fp = getFingerprint(molecule['SMILES'])
                molecule['fingerprint'] = np.array(fp)
                mols.append(molecule)
            except:
                print("unable to generate fp for entry " + str(i))
           

    df = pd.DataFrame(mols)

    print("num parsed molecules: ", len(df))

    return df

def getFingerprint(smiles):
    """ Create rdkit molecule and get Morgan fingerprint
    
    """
    mol = Chem.MolFromSmiles(smiles)
    info = {}
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, bitInfo=info)

    return fp


def normalize(array):
    """ Divides each value in array by the max or the array, normalizing to 1

        Args: numpy array of floats
    """ 
    
    return array / max(array)



def bin_spectra(data, bin_size):
    """ Creates bins of specified size over range 0 - 600, sums intensities within bin ranges and 
        normalizes results to 1
    
        Args: dataframe of mols and bin size
    """ 
    if bin_size % 2 != 0 and bin_size != 1:
        raise Exception("bin size must be 1 or even")

    bins = [0]

    for n in range(1, int(600/bin_size)):
        bins.append(n * bin_size + 0.5)

    bins.append(600)

    digitized = np.digitize(data['m/z'], bins)
    bin_sums = [data['intensity'][digitized == i].sum() for i in range(len(bins))]


    return normalize(bin_sums)
