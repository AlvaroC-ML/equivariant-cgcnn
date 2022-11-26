# This file creates the preprocessor that transforms an ICSD file into a Networkx graph to be taken as input by 
# SpektralDataset.py. It uses the NFP (Neuralfingerprint) library, it was written by Dr. Peter St. John
# (https://github.com/pstjohn), and it was slightly edited by me to fit the GVP model.

from typing import Dict

import networkx as nx
import numpy as np
from nfp.preprocessing import PymatgenPreprocessor
from pymatgen.core.periodic_table import Element

class GVPPreprocessor(PymatgenPreprocessor):
    def __init__(self, max_atomic_num=83, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.site_tokenizer = lambda x: Element(x).Z
        self._max_atomic_num = max_atomic_num

    @property
    def site_classes(self):
        return self._max_atomic_num

    def create_nx_graph(self, crystal, **kwargs) -> nx.MultiDiGraph:
        """crystal should be a pymatgen.core.Structure object."""
        g = nx.MultiDiGraph(crystal=crystal)
        g.add_nodes_from(((i, {"site": site}) for i, site in enumerate(crystal.sites)))

        if self.radius is None:
            # Get the expected number of sites / volume, then find a radius
            # expected to yield 2x the desired number of neighbors
            desired_vol = (crystal.volume / crystal.num_sites) * self.num_neighbors
            radius = 2 * (desired_vol / (4 * np.pi / 3)) ** (1 / 3)
        else:
            radius = self.radius

        for i, neighbors in enumerate(crystal.get_all_neighbors(radius)):
            # if len(neighbors) < self.num_neighbors:
            #     raise RuntimeError(f"Only {len(neighbors)} neighbors for site {i}")

            sorted_neighbors = sorted(neighbors, key=lambda x: x[1])[
                : self.num_neighbors
            ]

            visited = set()
            for _, distance, j, _ in sorted_neighbors:
                if j not in visited:
                    direction = crystal.sites[j].coords - crystal.sites[i].coords
                    g.add_edge(
                        i,
                        j,
                        distance= (np.append(direction, np.array([distance]))),
                    )
                    visited.add(j)
        return g

    def get_edge_features(
        self, edge_data: list, max_num_edges
    ) -> Dict[str, np.ndarray]:

        edge_feature_matrix = np.empty((max_num_edges, 4), dtype="float32")
        edge_feature_matrix[:] = np.nan  # Initialize distances with nans

        for n, (source_index, target_index, edge_dict) in enumerate(edge_data):
            edge_feature_matrix[n] = edge_dict["distance"]

        return {"distance": edge_feature_matrix}
