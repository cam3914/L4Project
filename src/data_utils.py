import numpy as np
import pandas as pd
from pyteomics import mgf
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
import spectrum_utils.plot as sup
import spectrum_utils.spectrum as sus
import matplotlib.pyplot as plt
import os.path


def getData(filename):
    """ Use pyteomics to read in data from /data/raw/ folder, check SMILES string and convert to dataframe

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
                        }

            molecule['binned'] = np.array(bin_spectra(molecule, 1))
            molecule['normed_binned'] = normalize(molecule['binned'])

            try:
                fp = getFingerprint(molecule['SMILES'])
                molecule['fingerprint'] = fp
                mols.append(molecule)
            except:
                print("unable to generate fp for entry " + str(i))
           

    df = pd.DataFrame(mols)

    print("num parsed molecules: ", len(df))

    return df

def fetch_data(tag):
    
    data = {
        'unique': {
            'filename': 'MS_data_allGNPS_uniqueInchikey14_191101.mgf',
            'path': os.path.dirname(os.path.abspath("")) + "/data/processed/unique8239.csv",
            'size': 8239
        },
        'large': {
            'filename': 'MS_data_allGNPS_smiles_191101.mgf',
            'path': os.path.dirname(os.path.abspath("")) + "/data/processed/large39467.csv",
            'size': 39467
        },
    } 
    
    path = data[tag]['path']
    size = data[tag]['size']
    df = False
    
    df = getData(data[tag]['filename'])
    print("finished")
    
    return df

def getFingerprint(smiles):
    """ Create rdkit molecule and get Morgan fingerprint
    
        Args: SMILES string

        Taken from Wei et al (2019)
    """ 
    mol = Chem.MolFromSmiles(smiles)
    fp = AllChem.GetHashedMorganFingerprint(mol, 2, nBits=4096)
    fp_arr = np.zeros(1)
    DataStructs.ConvertToNumpyArray(fp, fp_arr)
    return fp_arr


def normalize(array):
    """ Divides each value in array by the max of the array and multiplies each value by 1000, normalizing to 1000

        Args: numpy array of floats
    """ 
    if max(array) == 0:
        return array
    else:
        return np.array((array * 1000 / max(array)), dtype=np.float64)
    

def bin_spectra(data, bin_size):
    """ Creates bins of specified size over range 0 - 2000, sums intensities within bin ranges
    
        Args: dataframe of mols and bin size
    """ 
    if bin_size % 2 != 0 and bin_size != 1:
        raise Exception("bin size must be 1 or even")

    bins = []

    for n in range(0, int(2000/bin_size) + 1):
        bins.append(n * bin_size + 0.5)

    digitized = np.digitize(data['m/z'], bins)
    bin_sums = [data['intensity'][digitized == i].sum() for i in range(len(bins))]

    return np.array(bin_sums, dtype=np.float64)


def fingerprint_match(fp1, fp2):
    """ Compares two fingerprints and returns a percentage of how many values are equal
    
        Args: two numpy arrays of molecule fingerprints
    """ 
    match_array = fp1 == fp2

    num_matched = np.sum(match_array)

    if num_matched == 0:
        return 0
    else:
        return (num_matched / len(fp1)) * 100.0


def get_weighted_cosine_similarity(mol_q, mol_l):
    """ Takes the binned intensity arrays of two molecules and computes the weighted cosine similarity
    
        Args: two molecules, with m/z and intensity arrays for the spectrum and a mass
    """ 

    # find max m/z values
    Mq = 0
    Ml = 0

    for i, val in enumerate(mol_q['binned']):
        if val > 0:
            Mq = i + 1

    for i, val in enumerate(mol_l['binned']):
        if val > 0:
            Ml = i + 1

    Mmax = max([Mq, Ml])

    # create m/z bin index
    M_index = np.arange(1, Mmax + 1)

    # create array of I^0.5
    Iq_sqrt = np.sqrt(mol_q['binned'][:Mmax])
    Il_sqrt = np.sqrt(mol_l['binned'][:Mmax])

    # multiply I^0.5 by m
    Iq_M = Iq_sqrt * M_index
    Il_M = Il_sqrt * M_index

    dot_prod = Iq_M @ Il_M

    # calculate norm of left and right I^0.5xm vectors
    l_norm = np.sqrt(np.sum(np.square(Iq_M[:Mq])))
    r_norm = np.sqrt(np.sum(np.square(Il_M[:Ml])))

    denom = l_norm * r_norm

    return np.around(dot_prod / denom, decimals=3)

def get_cosine_similarity(mol_q, mol_l):
    """ Takes the binned intensity arrays of two molecules and computes the cosine similarity
    
        Args: two molecules, with m/z and intensity arrays for the spectrum and a mass
    """ 

    # find max m/z values
    Mq = 0
    Ml = 0

    for i, val in enumerate(mol_q['binned']):
        if val > 0:
            Mq = i + 1

    for i, val in enumerate(mol_l['binned']):
        if val > 0:
            Ml = i + 1

    Mmax = max([Mq, Ml])

    Iq = mol_q['binned'][:Mmax]
    Il = mol_l['binned'][:Mmax]

    dot_prod = Iq @ Il

    l_norm = np.sqrt(np.sum(np.square(Iq[:Mq])))
    r_norm = np.sqrt(np.sum(np.square(Il[:Ml])))

    denom = l_norm * r_norm

    return np.around(dot_prod / denom, decimals=3)

def show_spectrum(spectrum_dict, xlim=None):
    
    identifier = spectrum_dict['title']
    precursor_mz = spectrum_dict['pepmass'][0]
    precursor_charge = spectrum_dict['charge'][0]
    mz = spectrum_dict['m/z']
    intensity = spectrum_dict['intensity']

    # Create the MS/MS spectrum.
    spectrum = sus.MsmsSpectrum(identifier, precursor_mz, precursor_charge, mz, intensity)
    spectrum.filter_intensity(0.01)
    spectrum.scale_intensity('root')

    fig, ax = plt.subplots(figsize=(12, 6))
    sup.spectrum(spectrum, ax=ax)
    
    if xlim is not None:
        plt.xlim(right=xlim)
    plt.show()
    plt.close()

def compare_spectrum(spectrum1, spectrum2):
    spectrum_top = sus.MsmsSpectrum(spectrum1['title'], spectrum1['pepmass'][0], spectrum1['charge'][0], spectrum1['m/z'], spectrum1['intensity'])
    spectrum_top.filter_intensity(0.01)
    
    spectrum_bottom = sus.MsmsSpectrum(spectrum2['title'], spectrum2['pepmass'][0], spectrum2['charge'][0], spectrum2['m/z'], spectrum2['intensity'])
    spectrum_bottom.filter_intensity(0.01)

    _ , ax = plt.subplots(figsize=(12, 6))
    sup.mirror(spectrum_top, spectrum_bottom, ax=ax)
    
    plt.show()
    plt.close()

def get_top_cosine_similarity(mol_q, lib, n=10):
    similarities = []

    def sort_sim(val):
        return val[0]

    for i, mol in lib.iterrows():
        similarities.append((get_cosine_similarity(mol_q, mol), i))

    similarities.sort(reverse=True, key=sort_sim)
    
    return similarities[:n]

def compare_bins(true, predicted):

    if not np.all(true < 0.1) or not np.all(predicted < 0.1):
        fig, axs = plt.subplots(2, 1)
        a = np.arange(len(true))
        b = true

        x = np.arange(len(predicted))
        y = predicted

        true = axs[0].bar(a, b)
        pred = axs[1].bar(x, y, color='red')

        axs[0].set_ylim(0,1)
        axs[0].xaxis.set_visible(False)
        axs[1].set_ylim(0,1)
        axs[1].invert_yaxis()
        yticks = axs[1].yaxis.get_major_ticks()
        yticks[0].label1.set_visible(False)

        fig.subplots_adjust(hspace=0)
        plt.show()
        plt.close()

def cosine_sim(a, b):
    """ Takes the binned intensity arrays with the same length and computes the cosine similarity
    
        Args: two molecules, with m/z and intensity arrays for the spectrum and a mass
    """ 
    dot = a @ b
    
    if dot != 0:
        denom = np.sqrt(np.sum(np.square(a))) * np.sqrt(np.sum(np.square(b)))
        return dot/denom
    else:
        return 0

def get_best_fp_match(fp, lib, n=10):
    similarities = []

    def sort_sim(val):
        return val[0]

    for i, lib_fp in enumerate(lib['fingerprint']):
        similarities.append((fingerprint_match(fp, lib_fp), i))

    similarities.sort(reverse=True, key=sort_sim)
    
    return similarities[:n]


def get_top_cosine_similarity(vector, lib, n=10):
    similarities = []

    def sort_sim(val):
        return val[0]

    for i, mol in enumerate(lib):
        similarities.append((cosine_sim(vector, mol), i))

    similarities.sort(reverse=True, key=sort_sim)
    
    return similarities[:n]

def found_in_top_n(index, vector, lib, n):
    
    found = False
    similarities = get_top_cosine_similarity(vector, lib, n=n)
    for _, i in similarities:
        if index == i:
            found = True
    
    return found

def get_rank_at_n_score(predictions, lib, n):
    
    in_top_n = 0
    
    for i, vector in enumerate(predictions):
        if found_in_top_n(i, vector, lib, n):
            in_top_n += 1
        
    return (in_top_n/len(predictions)) * 100