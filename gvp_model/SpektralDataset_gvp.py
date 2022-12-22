# This file creates the dataset that Spektral can then use for Spektral layers.
# It was written by Dr. Peter St. John (https://github.com/pstjohn) and edited by myself to fit the base model.

import json
import numpy as np
import pandas as pd

from pymatgen.core.structure import Structure
from spektral.data import Dataset, Graph
from spektral.utils import sparse 
from tensorflow.math import log
from tqdm import tqdm

from preprocessor_gvp import GVPPreprocessor

from tensorflow import concat, expand_dims, convert_to_tensor, shape
from tensorflow.linalg import eigh

############
# Saving graphs
############

input_path = "/PATH/TO/DATASET/"

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

kappaL = pd.read_csv(
    "https://github.com/prashungorai/anisotropy-atlas/raw/master/cm2020-kappaL/kappaL-tensors-layered.csv"
)
kappaL.drop(index = [27, 28, 29, 30, 1225], inplace=True)
kappaL.reset_index(inplace=True)
kappaL = kappaL.sample(frac = 1) # Shuffle rows

# Create tensors
first_row = expand_dims(convert_to_tensor(kappaL[["T11", "T12", "T13"]]), axis = 1)
second_row = expand_dims(convert_to_tensor(kappaL[["T12", "T22", "T23"]]), axis = 1)
third_row = expand_dims(convert_to_tensor(kappaL[["T13", "T23", "T33"]]), axis = 1)
T = concat(
    [first_row, second_row, third_row], axis = -2
)

# Calculate SVD decomposition
e, v = eigh(T) # eigenvalue, eigenvector
e = expand_dims(e, axis = -1)
labels = concat([v, e], axis = -1)

# Separate data
valid_csv = kappaL.iloc[:50]
train_csv = kappaL.iloc[50:]

class KappaLDataset(Dataset):
    def __init__(self, data_csv, data_T):
        self.data_csv = data_csv
        self.data_T = data_T
        super().__init__()

    @staticmethod
    def inputs_to_graph(inputs, y):
        a, e = sparse.edge_index_to_matrix( 
            edge_index=inputs["connectivity"],
            edge_weight=np.ones(len(inputs["connectivity"])),
            edge_features=inputs["distance"],
        )
        assert a.getnnz() == e.shape[0], "checking for multigraph"

        x = inputs["site"][:, np.newaxis]
        return Graph(x=x, a=a, e=e, y=y)

    def read(self):
        return [
            self.inputs_to_graph(structure_inputs[row.icsd], self.data_T[i])
            for i, row in tqdm(self.data_csv.iterrows(), total=len(self.data_csv))
            if row.icsd in structure_inputs
        ]

print("Loading data!")
train_data = KappaLDataset(train_csv, labels)
valid_data = KappaLDataset(valid_csv, labels)

from spektral.data import DisjointLoader
from spektral.data.utils import to_tf_signature
from tensorflow import SparseTensorSpec, TensorSpec, as_dtype, int64

# We modify the loader only to be able to use tf_signatures
class modified_DisjointLoader(DisjointLoader):
    def tf_signature(self):
        signature = self.dataset.signature
        if "y" in signature:
            signature["y"]["shape"] = (None, 3, 4) #prepend_none(signature["y"]["shape"])
        if "a" in signature:
            signature["a"]["spec"] = SparseTensorSpec

        signature["i"] = dict()
        signature["i"]["spec"] = TensorSpec
        signature["i"]["shape"] = (None,)
        signature["i"]["dtype"] = as_dtype(int64)

        signature ["e"]["shape"] = (None, 4)

        return to_tf_signature(signature)
