import os
import torch
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import PandasTools
from torch_geometric.data import Data, InMemoryDataset, download_url, extract_zip
from torch_geometric.utils import to_undirected


class DrugRepurposing(InMemoryDataset):
    r"""Drug-Repurposing Dataset.

    Args:
        root (string): Root directory where the dataset should be saved.
        virus (string, optional): The name of the virus to be used. Can be one
            of the following (default: :obj:`'SARS-CoV-2'`):

            Name       Alias for
            ---------- ----------------------------------------------------
            HCV        Hepatitus C Virus
            HHV-1      Human Herpesvirus 1
            HHV-2      Human Herpesvirus 2
            HSV-1      Human Herpesvirus 1
            HSV-2      Human Herpesvirus 2
            HHV-3      Human Herpesvirus 3
            VZV        Human Herpesvirus 3
            HHV-4      Human Herpesvirus 4
            HBV        Human Herpesvirus 4
            HHV-5      Human Herpesvirus 5
            HCMV       Human Herpesvirus 5
            HHV-6A     Human Herpesvirus 6A
            HHV-6B     Human Herpesvirus 6B
            HHV-7      Human Herpesvirus 7
            HHV-8      Human Herpesvirus 8
            KSHV       Human Herpesvirus 8
            HIV        Human Immunodeficiency Virus 1
            HIV-1      Human Immunodeficiency Virus 1
            HIV-2      Human Immunodeficiency Virus 2
            HPV-6      Human Papillomavirus 6B
            HPV-10     Human Papillomavirus 10
            HPV-16     Human Papillomavirus 16
            MERS-CoV   Middle-East Respiratory Syndrome-Related Coronavirus
            SARS-CoV   Severe Acute Respiratory Syndrome Coronavirus
            SARS-CoV-2 Severe Acute Respiratory Syndrome Coronavirus 2
            SV-40      Simian Virus 40
            VACV       Vaccinia Virus
            VV         Vaccinia Virus

        add_missing_proteins (bool, optional): If any Host-Host interaction is
            present in the virus dataset, add it to the main interactome
            (default: :obj:`False`)
        restrict_to (int, optional): Reduce the human interactome to the
            induced subgraph of the proteins contained in the :math:`k`-hop
            neighborhood of the proteins affected by the virus (default:
            :obj:`None`)
        feature_extractor (callable, optional): A function that maps a RDKit
            molecule (:obj:`Mol`) to a :obj:`dict` of features to be include
            into the :obj:`Data` object. If :obj:`None`, the graph will be
            represented only by its topology (default :obj:`None`).
        protein_features (callable, optional): A function that maps an array
            of Entrez ID strings to a single :obj:`dict` of features of the
            same size (along the first dimension). These features will be
            included as node features in the human interactome (default
            :obj:`None`).
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in an
            :obj:`torch_geometric.data.Data` object and returns a boolean
            value, indicating whether the data object should be included in the
            final dataset. (default: :obj:`None`)
    """

    base_url = "https://raw.githubusercontent.com/CLAIRE-COVID-T4/covid-data/master/data/"

    # Note: Some viruses have multiple aliases (e.g., 'HIV' and 'HIV-1')
    virus_aliases = {
        'HCV': 'Hepatitus_C_Virus',
        'HHV-1': 'Human_Herpesvirus_1',
        'HHV-2': 'Human_Herpesvirus_2',
        'HSV-1': 'Human_Herpesvirus_1',
        'HSV-2': 'Human_Herpesvirus_2',
        'HHV-3': 'Human_Herpesvirus_3',
        'VZV': 'Human_Herpesvirus_3',
        'HHV-4': 'Human_Herpesvirus_4',
        'HBV': 'Human_Herpesvirus_4',
        'HHV-5': 'Human_Herpesvirus_5',
        'HCMV': 'Human_Herpesvirus_5',
        'HHV-6A': 'Human_Herpesvirus_6A',
        'HHV-6B': 'Human_Herpesvirus_6B',
        'HHV-7': 'Human_Herpesvirus_7',
        'HHV-8': 'Human_Herpesvirus_8',
        'KSHV': 'Human_Herpesvirus_8',
        'HIV': 'Human_Immunodeficiency_Virus_1',
        'HIV-1': 'Human_Immunodeficiency_Virus_1',
        'HIV-2': 'Human_Immunodeficiency_Virus_2',
        'HPV-6': 'Human_papillomavirus_6b',
        'HPV-10': 'Human_papillomavirus_10',
        'HPV-16': 'Human_papillomavirus_16',
        'MERS-CoV': 'Middle-East_Respiratory_Syndrome-related_Coronavirus',
        'SARS-CoV': 'Severe_acute_respiratory_syndrome_coronavirus',
        'SARS-CoV-2': 'Severe_acute_respiratory_syndrome_coronavirus_2',
        'SV-40': 'Simian_Virus_40',
        'VACV': 'Vaccinia_Virus',
        'VV': 'Vaccinia_Virus'
    }

    def __init__(self, root, virus='SARS-CoV-2', add_missing_proteins=False,
                 restrict_to=None, feature_extractor=None, protein_features=None,
                 transform=None, pre_transform=None, pre_filter=None):
        assert restrict_to is None or restrict_to >= 0, "Not a valid hop number."

        self.feature_extractor = feature_extractor
        self.protein_features = protein_features
        self.restrict_to = restrict_to
        self.virus = self.virus_aliases.get(virus, virus)
        self.add_missing_proteins = add_missing_proteins

        super(DrugRepurposing, self).__init__(root, transform, pre_transform, pre_filter)

        self.data, self.slices = torch.load(self.processed_paths[0])
        self.entrez_ids, self.virus_ids, self.drug_ids = torch.load(self.processed_paths[-1])

    @property
    def raw_file_names(self):
        return ['drug-structures.sdf', 'drug-host.tab', 'protein-protein.tab', f'{self.virus}.tab']

    @property
    def processed_file_names(self):
        return ['drug-structures.pt', 'drug-host.pt', 'host-host.pt',
                'virus-host.pt', 'virus-virus.pt', 'id_maps.pt']

    @property
    def processed_dir(self):
        sub_dir = 'all' if self.restrict_to is None else f'{self.restrict_to}_hops'
        return os.path.join(self.root, 'processed', self.virus, sub_dir)

    def download(self):
        for name in self.raw_file_names[:-1]:
            download_url(self.base_url + name, self.raw_dir)

        download_url(self.base_url + 'virus-host/' + self.raw_file_names[-1], self.raw_dir)

    def process(self):
        # Host-Host
        hh = pd.read_table(self.raw_paths[2]).values.T

        # Virus-Virus
        vv = pd.read_table(self.raw_paths[3], usecols=[
            'EntrezGeneID_InteractorA', 'EntrezGeneID_InteractorB',
            'OrganismID_InteractorA', 'OrganismID_InteractorB'
        ], na_values='-').fillna(-1).convert_dtypes(int).values

        human_only = np.all(vv[:, 2:] == 9606, axis=-1)  # Homo Sapiens

        if self.add_missing_proteins:
            hh = np.concatenate((hh, vv[human_only, :2].T))

        entrez_ids, interactome = np.unique(hh, return_inverse=True)
        interactome = interactome.reshape((2, -1))
        pid2pos = {idx: pos for pos, idx in enumerate(entrez_ids)}

        vv = vv[~human_only]
        mask = vv.T[2:] == 9606
        vv = vv.T[:2]

        virus_ids, vv[~mask] = np.unique(vv[~mask], return_inverse=True)
        vv[mask] = [pid2pos.get(pid, -1) for pid in vv[mask]]

        # Virus-Host
        human_related = np.any(mask, axis=0)
        vv = np.where(mask[:1], vv[::-1], vv)
        vh = vv[:, human_related & (vv[1] != -1)]
        vv = vv[:, ~human_related]

        # K-Hop Restriction
        if self.restrict_to is not None:
            mask = np.zeros(entrez_ids.shape[0], dtype=bool)
            row, col = interactome
            mask[vh[1].astype(int)] = True

            for _ in range(self.restrict_to):
                last = mask.copy()
                mask[col] |= last[row]
                mask[row] |= last[col]

            interactome = interactome[:, mask[row] & mask[col]]
            old_ids, interactome = np.unique(interactome, return_inverse=True)
            interactome = interactome.reshape((2, -1))
            pid2pos = {idx: pos for pos, idx in enumerate(entrez_ids[mask])}
            vh[1] = [pid2pos[entrez_ids[pid]] for pid in vh[1]]
            entrez_ids = entrez_ids[mask]

        # Drug Structures
        drugs = PandasTools.LoadSDF(self.raw_paths[0], idName='RowID')[['RowID', 'ROMol']]
        drugs.iloc[:, 0] = drugs.iloc[:, 0].apply(lambda did: int(did[2:]))
        drugs = drugs.set_index('RowID')

        # Drugs-Host
        dh = pd.read_table(self.raw_paths[1], converters={
            '#DrugBankID': lambda did: int(did[2:]),
            'EntrezGeneID': lambda pid: int(pid2pos.get(int(pid), -1))
        })

        dh = dh[dh.iloc[:, 1] != -1].join(drugs[[]], on='#DrugBankID', how='inner').values.T
        drug_ids, dh[0] = np.unique(dh[0], return_inverse=True)
        drugs = drugs.loc[drug_ids, 'ROMol']

        data_list = []

        for mol in drugs:
            if self.feature_extractor is None:
                adj = Chem.GetAdjacencyMatrix(mol) * Chem.Get3DDistanceMatrix(mol)
                nz = np.nonzero(adj)

                features = {
                    'num_nodes': adj.shape[0],
                    'edge_index': torch.LongTensor(np.stack(nz)),
                    'edge_attr': torch.FloatTensor(adj[nz])
                }
            else:
                features = self.feature_extractor(mol)

            data = Data(**features)

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            if self.pre_filter is None or self.pre_filter(data):
                data_list.append(data)

        self.data, self.slices = self.collate(data_list)
        num_proteins = entrez_ids.shape[0]
        num_virus = virus_ids.shape[0]

        host_host = {
            'num_nodes': num_proteins,
            'edge_index': to_undirected(torch.LongTensor(interactome),
                                        num_nodes=num_proteins)
        }

        virus_virus = {
            'num_nodes': num_virus,
            'edge_index': to_undirected(torch.LongTensor(vv.astype(int)),
                                        num_nodes=num_virus)
        }

        if self.protein_features is not None:
            host_host.update(self.protein_features(entrez_ids))
            # virus_virus.update(self.protein_features(virus_ids))

        virus_host = torch.LongTensor(vh.astype(int))
        drug_host = torch.LongTensor(dh.astype(int))
        self.entrez_ids = torch.LongTensor(entrez_ids.astype(int))
        self.virus_ids = torch.LongTensor(virus_ids.astype(int))
        self.drug_ids = torch.LongTensor(drug_ids.astype(int))

        torch.save((self.data, self.slices), self.processed_paths[0])
        torch.save(drug_host, self.processed_paths[1])
        torch.save(host_host, self.processed_paths[2])
        torch.save(virus_host, self.processed_paths[3])
        torch.save(virus_virus, self.processed_paths[4])
        torch.save((self.entrez_ids, self.virus_ids, self.drug_ids), self.processed_paths[5])

    def get_entrez_id(self, index, virus=False):
        r"""Retrieve the original EntrezID of a protein.

        Args:
             index (int, list or Tensor): The protein index (or indices) in
                the interactome.
             virus (bool, optional): Whether the indices refer to the human
                or viral interactome.
        """
        if virus:
            return self.virus_ids[index].tolist()

        return self.entrez_ids[index].tolist()

    def get_drugbank_id(self, index):
        r"""Retrieve the original DrugBankID of a molecule.

        Args:
             index (int, list or Tensor): The molecule position in the
                dataset.
        """
        return [f'DB{i:05d}' for i in self.drug_ids[index].tolist()]

    def get_drugs_interactions(self):
        r"""Return the drug-host interactions.

        Returns a :obj:`torch.LongTensor` containing drug indices in the first
        row, referring to the position of a specific drug in the dataset, and
        protein indices in the second one, as protein position in the human
        interactome. The two rows form a sparse adjacency matrix of bipartite
        graph of drug-host interactions.
        """
        return torch.load(self.processed_paths[1])

    def get_human_interactome(self):
        r"""Return the human interactome.

        Returns a :obj:`torch_geometric.data.Data` containing the graph
        representation of the human interactome.
        """
        return Data(**torch.load(self.processed_paths[2]))

    def get_virus_interactions(self):
        r"""Return the virus-host interactions.

        Returns a :obj:`torch.LongTensor` containing protein indices
        referring, in the first row, to the position of a specific viral
        protein in the viral interactome, while, in the second one, to protein
        positions in the human interactome. The two rows form a sparse
        adjacency matrix of bipartite graph of virus-host interactions.
        """
        return torch.load(self.processed_paths[3])

    def get_virus_interactome(self):
        r"""Return the viral interactome.

        Returns a :obj:`torch_geometric.data.Data` containing the graph
        representation of the viral interactome.
        """
        return Data(**torch.load(self.processed_paths[4]))


class SARSCoV(InMemoryDataset):
    r"""SARSCoV datasets.

    Args:
        root (string): Root directory where the dataset should be saved.
        name (string): Name of the dataset. Possible values:

            - :obj:`'AID1706'`: In-vitro assay that detects inhibition of
                                SARS-CoV 3CL protease via fluorescence
                                (default value).
            - :obj:`'AID1706_eval'`: Evaluation set for :obj:`'AID1706'`.
            - :obj:`'AID1706_full'`: Both :obj:`'AID1706'` and
                                     :obj:`'AID1706_eval'`
            - :obj:`'PLpro'`: Bioassay that detects activity against SARS-CoV
                              in yeast models via PL protease inhibition.
            - :obj:`'Mpro'`: Fragments screened for 3CL protease binding using
                             crystallography techniques.
            - :obj:`'PubChemIdex'`: FDA-approved drugs that are mentioned in
                                    generic coronavirus literature.
            - :obj:`'BRHLib'`: Compounds from the Broad Repurposing Hub, many
                               of which are FDA-approved.
            - :obj:`'ExtLib'`: A set of FDA-approved drugs.
            - :obj:`'ExpExtLib'`: A larger set of FDA-approved drugs, but not
                                  a strict superset of :obj:`'ExtLib'`.
            - :obj:`'EColi'`: Compounds which have been screened for
                              inhibitory activity against E. coli.

            Note:
                :obj:`'PubChemIdex'`, :obj:`'BRHLib'`, :obj:`'ExtLib'`, and
                :obj:`'ExpExtLib'` have no target. The other datasets have an
                "activity" (binary) target.

        features_extractor (callable, optional): A function that maps a RDKit
            molecule (:obj:`Mol`) to a :obj:`dict` of features to be include
            into the :obj:`Data` object. If :obj:`None`, the graph will be
            represented only by its topology (default :obj:`None`).
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in an
            :obj:`torch_geometric.data.Data` object and returns a boolean
            value, indicating whether the data object should be included in the
            final dataset. (default: :obj:`None`)
    """

    base_url = 'https://github.com/yangkevin2/coronavirus_data/raw/master/data/'
    urls = {
        'AID1706': 'AID1706_binarized_sars.csv',
        'AID1706_eval': 'evaluation_set_v2.csv',
        'AID1706_full': 'AID1706_binarized_sars_full_eval_actives.csv',
        'PLpro': 'PLpro.csv',
        'Mpro': 'mpro_xchem.csv',
        'PubChemIdex': 'corona_literature_idex.csv',
        'BRHLib': 'broad_repurposing_library.csv',
        'ExtLib': 'external_library.csv',
        'ExpExtLib': 'expanded_external_library.csv',
        'EColi': 'ecoli.csv'
    }

    def __init__(self, root, name='AID1706', feature_extractor=None,
                 transform=None, pre_transform=None, pre_filter=None):
        assert name in self.urls, f'Dataset {name} not found. Available datasets: ' + \
                                  ', '.join(self.urls.keys())
        self.name = name
        self.feature_extractor = feature_extractor
        super(SARSCoV, self).__init__(os.path.join(root, name),
                                      transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [self.urls[self.name]]

    @property
    def processed_file_names(self):
        return [self.name + '.pt']

    def download(self):
        download_url(self.base_url + self.urls[self.name], self.raw_dir)

    def process(self):
        df = pd.read_csv(self.raw_paths[0])
        data_list = []

        for row in df.itertuples(False, None):
            data = self._process_row(*row)

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            if self.pre_filter is None or self.pre_filter(data):
                data_list.append(data)

        self.data, self.slices = self.collate(data_list)
        torch.save((self.data, self.slices), self.processed_paths[0])

    def _process_row(self, smiles, label=None):
        mol = Chem.MolFromSmiles(smiles)

        if self.feature_extractor is None:
            adj = Chem.GetAdjacencyMatrix(mol)
            features = {
                'num_nodes': adj.shape[0],
                'edge_index': torch.LongTensor(np.stack(np.nonzero(adj)))
            }
        else:
            features = self.feature_extractor(mol)

        return Data(y=label, **features)


class Pseudomonas(InMemoryDataset):
    r"""Pseudomonas aeruginosa dataset.

    Args:
        root (string): Root directory where the dataset should be saved.
        url (sting, optional): The url of the Pseudomonas dataset, provided by
            `AI Cures. <https://www.aicures.mit.edu/>`_ This parameter should
            be assigned the at the first execution of the class initializer.
            Once the zip file is downloaded, the parameter becomes optional
            (it is ignored if already downloaded).
        split (string): Split of the dataset. Can be either of :obj:`'train'`
            or :obj:`'test'` (default :obj:`'train'`).
        feature_extractor (callable, optional): A function that maps a RDKit
            molecule (:obj:`Mol`) to a :obj:`dict` of features to be include
            into the :obj:`Data` object. If :obj:`None`, the graph will be
            represented only by its topology (default :obj:`None`).
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in an
            :obj:`torch_geometric.data.Data` object and returns a boolean
            value, indicating whether the data object should be included in the
            final dataset. (default: :obj:`None`)
    """

    splits = ['train', 'test']

    def __init__(self, root, url=None, split='train', feature_extractor=None,
                 transform=None, pre_transform=None, pre_filter=None):
        assert split in self.splits, f'Wrong split "{split}". Can be either "train" or "test".'
        self.url = url
        self.split = split
        self.feature_extractor = feature_extractor
        super(Pseudomonas, self).__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(os.path.join(self.processed_dir, self.split + '.pt'))

    @property
    def raw_file_names(self):
        return ['pseudomonas.zip']

    @property
    def processed_file_names(self):
        return [split + '.pt' for split in self.splits] + ['splits.pt']

    def download(self):
        assert self.url is not None, "The dataset URL must be provided at least one time."

        csv_path = download_url(self.url, self.raw_dir)
        os.rename(csv_path, self.raw_paths[0])

    def process(self):
        extract_zip(self.raw_paths[0], self.raw_dir, log=False)
        dir_path = os.path.join(self.raw_dir, 'pseudomonas')
        smiles_idx = {}
        data_lists = {}

        for i, split in enumerate(self.splits):
            df = pd.read_csv(os.path.join(dir_path, split + '.csv')).drop('id', axis=1, errors='ignore').dropna(1)
            data_lists[split] = []

            for row in df.itertuples(False, None):
                smiles_idx.setdefault(row[0], len(smiles_idx))
                data = self._process_row(*row)

                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                if self.pre_filter is None or self.pre_filter(data):
                    data_lists[split].append(data)

            data, slices = self.collate(data_lists[split])
            torch.save((data, slices), self.processed_paths[i])

        splits = []

        for fold in range(10):
            idx = []

            for split in ['train', 'dev', 'test']:
                df = pd.read_csv(os.path.join(dir_path, 'train_cv', f'fold_{fold}', split + '.csv'))
                idx.append(torch.LongTensor(df['smiles'].map(lambda s: smiles_idx[s])))

            splits.append(tuple(idx))

        torch.save(tuple(splits), self.processed_paths[-1])

    def get_splits(self):
        for train_idx, val_idx, test_idx in torch.load(self.processed_paths[-1]):
            yield train_idx, val_idx, test_idx

    def _process_row(self, smiles, label=None):
        mol = Chem.MolFromSmiles(smiles)

        if self.feature_extractor is None:
            adj = Chem.GetAdjacencyMatrix(mol)
            features = {
                'num_nodes': adj.shape[0],
                'edge_index': torch.LongTensor(np.stack(np.nonzero(adj)))
            }
        else:
            features = self.feature_extractor(mol)

        return Data(y=label, **features)
