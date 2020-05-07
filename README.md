# COVID-related datasets for PyTorch-Geometric

This repo contains some useful `InMemoryDataset`s for [PyTorch-Geometric](https://pytorch-geometric.readthedocs.io/en/latest/) to be used for COVID-related tasks.

To use these dataset you need the following packages:
```
torch
torch-geometric
numpy
pandas
rdkit
```

## `DrugRepurposing` dataset

This dataset provides 

 - a set of drug structures, with user-defined graph-, node-, and edge-features;
 - a graph representing the Human protein-protein interactions; 
 - the drugs interactions with the Human proteins, as a bipartite graph;
 - a set of Human viruses, and their interactions with the Human interactome. A virus can be selected using the `virus` parameter, which can be one of `'HCV'`, `'HHV-1'` to `'HHV-8'`, `'HIV-1'`, `'HIV-2'`, `'HPV-6'`, `'HPV-10'`, `'HPV-16'`, `'MERS-CoV'`, `'SARS-CoV'`, `'SARS-CoV-2'`, `'SV-40'`, and `'VACV'`.
 
### Basic usage

```python
In[1]: from covid import DrugRepurposing
In[2]: dr = DrugRepurposing(root='./data/DR/', virus='SARS-CoV-2')
Downloading https://www.drugbank.ca/releases/5-1-5/downloads/all-open-structures
Extracting DATA/DR/raw/all-open-structures
Downloading https://static-content.springer.com/esm/art%3A10.1038%2Fs41467-019-09186-x/MediaObjects/41467_2019_9186_MOESM4_ESM.xlsx
Downloading https://static-content.springer.com/esm/art%3A10.1038%2Fs41467-019-09186-x/MediaObjects/41467_2019_9186_MOESM3_ESM.xlsx
Downloading https://downloads.thebiogrid.org/Download/BioGRID/Latest-Release/BIOGRID-ORGANISM-LATEST.tab3.zip
Processing...
Done!
In[3]: dr
Out[3]: DrugRepurposing(4142)
In[4]: dr[0]
Out[4]: Data(edge_attr=[320], edge_index=[2, 320])
In[5]: ppi = dr.get_human_interactome()
In[6]: ppi
Out[6]: Data(edge_index=[2, 431056])
In[7]: ppi.num_nodes
Out[7]: 15970
In[8]: dr.get_virus_interactions()
Out[8]: 
tensor([[    3,     3,     3,  ...,    1,    1,     1],
        [ 5397,  7906,  3938,  ..., 3212, 4693, 11010]])
In[9]: dr.get_drugs_interactions()
Out[9]: 
tensor([[   9,    9,    9,  ..., 4116, 4116, 4116],
        [1981, 3793, 9185,  ..., 6996, 7305, 7324]])
```
 
## `SARSCoV` dataset

This class provides a set datasets for classification and unsupervised tasks. The data is retrieved from [AI Cures SARSCoV-related datasets](https://www.aicures.mit.edu/data), where a full description of the data is provided. The dataset can be specified in the `name` argument, which can be set as one of the following:

 - `'AID1706'`: In-vitro assay that detects inhibition of SARS-CoV 3CL protease via fluorescence.
 - `'AID1706_eval'`: Evaluation set for `'AID1706'`.
 - `'AID1706_full'`: Both `'AID1706'` and `'AID1706_eval'`.
 - `'PLpro'`: Bioassay that detects activity against SARS-CoV in yeast models via PL protease inhibition.
 - `'Mpro'`: Fragments screened for 3CL protease binding using crystallography techniques.
 - `'PubChemIdex'`: FDA-approved drugs that are mentioned in generic coronavirus literature.
 - `'BRHLib'`: Compounds from the Broad Repurposing Hub, many of which are FDA-approved.
 - `'ExtLib'`: A set of FDA-approved drugs.
 - `'ExpExtLib'`: A larger set of FDA-approved drugs, but not a strict superset of `'ExtLib'`.
 - `'EColi'`: Compounds which have been screened for inhibitory activity against E. coli.

**Note:** `'PubChemIdex'`, `'BRHLib'`, `'ExtLib'`, and `'ExpExtLib'` have no target. The other datasets have an `'activity'` (binary) target.

### Basic usage

```python
In[1]: from covid import SARSCoV
In[2]: ds = SARSCoV(root='./data/', name='Mpro')
Downloading https://github.com/yangkevin2/coronavirus_data/raw/master/data/mpro_xchem.csv
Processing...
Done!
In[3]: ds
Out[3]: SARSCoV(880)
In[4]: ds[0]
Out[4]: Data(edge_index=[2, 40], y=[1])
```

## `Pseudomonas` dataset

This dataset provides a set of molecules for the [Pseudomonas Aeruginosa Open-Task of AI Cures](https://www.aicures.mit.edu/data).

### Basic usage

```python
In[1]: from covid import Pseudomonas
In[2]: ds = Pseudomonas(root='./data/', url='https://<...>.zip')
Downloading https://<...>.zip
Processing...
Done!
In[3]: ds
Out[3]: Pseudomonas(2097)
In[4]: ds[0]
Out[4]: Data(edge_index=[2, 16], y=[1])
```
