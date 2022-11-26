# This file creates the dataset that Spektral can then use for Spektral layers.
# It was written by Dr. Peter St. John (https://github.com/pstjohn) and edited by myself to fit the base model.

import json
import numpy as np
import pandas as pd

from pymatgen.core.structure import Structure
from spektral.data import Dataset, Graph
from spektral.utils import sparse
from tqdm import tqdm

from preprocessor import GVPPreprocessor

############
# Saving graphs
############

# The data is saved in a JSON file that contains the crystal structures as values, and the corresponding ICSD ID
# as the key.

input_path = "/path/to/json/file.json"

structures = {}
with open(input_path) as f:
    for key, struct in json.load(f).items():
        structures[key] = Structure.from_dict(struct)

preprocessor = GVPPreprocessor()
structure_inputs = {
    key: preprocessor(structure) for key, structure in tqdm(structures.items())
}

############
# Saving labels
############

# This (publicly available) dataset contains the target values.
# We will only use the max possible value of kappa_L for the base model.
kappaL = pd.read_csv(
    "https://github.com/prashungorai/anisotropy-atlas/raw/master/cm2020-kappaL/kappaL-tensors-layered.csv"
)

valid = kappaL.sample(50, random_state=1)
train = kappaL[~kappaL.index.isin(valid)].sample(frac=1.0, random_state=1)

############
# Creating dataset
############

class KappaLDataset(Dataset):
    def __init__(self, data):
        self.data = data
        super().__init__()

    @staticmethod
    def inputs_to_graph(inputs, y):
        a, e = sparse.edge_index_to_matrix(
            edge_index=inputs["connectivity"],
            edge_weight=np.ones(len(inputs["connectivity"])),
            edge_features=inputs["distance"],
        )

        x = inputs["site"][:, np.newaxis]
        return Graph(x=x, a=a, e=e, y=y)

    def read(self):
        return [
            self.inputs_to_graph(structure_inputs[row.icsd], row.kappaLmax)
            for _, row in tqdm(self.data.iterrows(), total=len(self.data))
            if row.icsd in structure_inputs
        ]

print("Loading data!")
train_data = KappaLDataset(train)
valid_data = KappaLDataset(valid)
