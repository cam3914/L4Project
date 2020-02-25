# Project meetings

## 8th October

* Discussed project options
* Sent a few paper and things to research

## 18th October

* Confirmed Project title
* Meeting format (slides, questions, etc)
* Went through student/supervisor questionnaire 
* Mentieoned different technologies (Pytorch/Keras), no preference
* Emphasis on uncertainty of predictions
* Might be able to do transfer learning from google brain paper code
* Cost function by distance in space of spectra
* Ideally trace predictions back to SMILES string
* Discussed data source
* Possibility of coordination with other projects, not until later on

## 25th October

* Update on progress
* Confirmed GitHub VCS
* Confirmed data formatting in (.mgf) but only need SMILES
* Discussed semester plan (optimistic) for week 6-12
* Early steps: predict presence of spectra accurately first

## 1st November

* Data transfer
* RMT research overlap, choose ML in metabolomics/MS to deepen understanding, be careful not to overlap too much
* Research Sirius 4 paper and it's citations - best results (metfrag) 
* CFM-ID and google brain paper are to only two methods of prediction so far
* Sirius/CSI:FingerID wininng CASMI
* Once read in and formatted data, need to bin spectra (integers?)
* Next meeting on Friday at 1.30pm
* Sent link to github repository

## 8th November

* More data, start with unique spectra
* New data has mass tags cleaned up
* Binning spectra - dont start at 1 as masses tend towards whole numbers so a split at integers would split grouped data points
* Can get inchikey from smiles, not other way around
* Chem spider - database of structures to search for SMILES if needed
* spectrum_utils for displaying spectra (or just do it with matplotlib etc)
* Next meeting Tuesday 12th 15:00 in 1028 lab

## 12th November

* Haven't had time to do much since te last meeting
* Having rdkit and package issues
* CI good for grades but need to get started on the ML
* Next meeting 22nd Nov at 1.30pm

## 22nd November

* Confirmed specifics, project is using data from ESI-MS\MS
* Google Brain paper does not use tandem MS but does not matter as it is higher energy and produces fragments, same result as ESI-MS/MS
* No feasible way to evaluate unidentified spectra, create a training and test split in the data
* Metfrag is the best performing method of predicting structure from spectra, still not great and slow
* Structure to spectra is the preferred method to augment spectral libraries with synthetic data
* CFM-ID created bad data which had to be removed from METLIN
* Next meeting 2.30 on Friday the 29th

## 29th November

* Brief meeting, not much to discuss as not much progress

## 21st January

* Update on progress
* Need to find difference between ECFPs and Morgan fingerprints
* Clarification on predicting single spectra 
* Check how google brain paper deals with lost masses

## 28th January

* No meeting

## 4th February

* Morgan fingerprints appear to be interchangeable with ECFPs
* Confirmed binning method is correct, summing intensities in bins
* Bins need to be max size 1 to be effective
* Molecules with rdkit read errors can be ignored, probably poor data
* Need to start thinking about evaluation, possibly an intelligent guesser based on most similar peaks
* Could normalize by log

## 12 February

* No meeting, email progress update

## 19th February

* baseline for evaluation: function to find similarity of fingerprint in library
* need to understand uniqueness of fingerprints, how well a library search method does
* can use cosine similarity of whole spectra or rmse of each bin
* paired statistical test for baseline scores and neural net scores
* use different sizes of data in training to compare network performance
* start basic, make sure network trains before going too complex where it is easy to run into issues