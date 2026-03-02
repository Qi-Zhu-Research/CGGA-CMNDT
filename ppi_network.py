import numpy as np

from numba import njit
import math
import pandas as pd


class PPINetwork:
    def __init__(self, network_fname = "TCGA_Data/networkFile/mergeNet.txt"):
        self.NetworkMap = self.generate_network_map(network_fname)

    @staticmethod
    def generate_network_map(network_fname):
        ret_map = {}
        import os

        with open(os.path.join(network_fname), "r") as file:
            lines = file.readlines()
            for line in lines:
                ppi_network_list = line.strip().split("\t")
                if len(ppi_network_list) > 1:
                    tmp_arr = []
                    for i in range(1, len(ppi_network_list)):
                        tmp_arr.append(ppi_network_list[i])
                    if ppi_network_list[0] in ret_map:
                        print("already_contains_ppi_node_key: ", ppi_network_list[0])
                        ret_map[ppi_network_list[0]].extend(tmp_arr)
                    else:
                        ret_map[ppi_network_list[0]] = tmp_arr

        return ret_map

    def get_network_map(self):
        return self.NetworkMap

    def calc_connectivity_by_gene_name_list(self, gene_name_list):
        from itertools import product

        K = len(gene_name_list)

        connectivity = 0.0
        # Calculates the original connectivity
        for gname_i, gname_j in product(gene_name_list, repeat = 2):
            if gname_i in self.NetworkMap and gname_j in self.NetworkMap[gname_i]:
                # print(f"{gname_i}, {gname_j}")
                connectivity += 1

        # Calculates the normalized connectivity
        connectivity_norm_factor = 2 * math.comb(K, 2)
        connectivity_norm = (1 + connectivity / connectivity_norm_factor) if K >= 2 else 1.0
        # connectivity_norm = (0.5 + connectivity / connectivity_norm_factor) if K >= 2 else 1.0
        # connectivity_norm = (0.2 + connectivity / connectivity_norm_factor) if K >= 2 else 1.0
        # return connectivity
        return round(connectivity_norm, 3) - 1.0

    def generate_adjacency_matrix(self, gene_names = None):
        if gene_names is None:
            gene_names = list(self.NetworkMap.keys())

        adjacency_matrix = pd.DataFrame(0, index = gene_names, columns = gene_names)

        for node, neighbors in self.NetworkMap.items():
            for neighbor in neighbors:
                if node in gene_names and neighbor in gene_names:
                    adjacency_matrix.loc[node, neighbor] = 1

        return adjacency_matrix


class PPINetworkString(PPINetwork):
    def __init__(self, network_fname, threshold = 0.0):
        self.threshold_value = threshold
        self.ppi_df = pd.read_csv(
            network_fname,
            delimiter = "\t", header = 0
        )
        self.NetworkMap = self.generate_network_map()
        self.gene_set = set(self.ppi_df["#node1"]).union(set(self.ppi_df["node2"]))

    @property
    def threshold(self):
        return self.threshold_value

    @property
    def num_edges(self):
        return self.ppi_df.shape[0]

    @property
    def num_genes(self):
        return len(self.gene_set)

    @property
    def genes(self):
        return self.gene_set

    def calc_weighted_connectivity_by_gene_name_list(self, gene_name_list):
        from itertools import product

        K = len(gene_name_list)

        connectivity = 0.0
        # Calculates the original connectivity
        for gname_i, gname_j in product(gene_name_list, repeat = 2):
            if gname_i in self.NetworkMap and gname_j in self.NetworkMap[gname_i]:
                connectivity += float(self.get_edge_score(gname_i, gname_j))

        # Calculates the normalized connectivity
        connectivity_norm_factor = math.comb(K, 2)
        connectivity_norm = 1 + connectivity / connectivity_norm_factor

        return round(connectivity_norm, 2)

    def get_edge_score(self, node1, node2):
        node1 = node1.strip();
        node2 = node2.strip()
        cond1 = self.ppi_df["#node1"].isin([node1]) & self.ppi_df["node2"].isin([node2])
        cond2 = self.ppi_df["node2"].isin([node1]) & self.ppi_df["#node1"].isin([node2])

        if cond1.any():
            return self.ppi_df.loc[cond1, "combined_score"].astype(np.float_)
        elif cond2.any():
            return self.ppi_df.loc[cond2, "combined_score"].astype(np.float_)
        else:
            return "None Edges"

    def generate_network_map(self):
        ret_map = {}

        # Iterate over each row in the DataFrame
        for index, row in self.ppi_df.iterrows():
            if row["combined_score"] < self.threshold:
                continue

            node1 = row["#node1"]
            node2 = row["node2"]

            # Add node2 to the list of neighbors for node1
            if node1 in ret_map:
                ret_map[node1].append(node2)
            else:
                ret_map[node1] = [node2]

        return ret_map


if __name__ == '__main__':
    ppi_network = PPINetwork("../TCGA_Data/networkFile/mergeNet.txt")

    testing_gene_name_list = ['TP53', 'CDKN2A', 'CDK4', 'ERBB2', 'EGFR', 'PIK3R1', 'VAV1', 'RB1', 'MDM2', 'STAT1']
    testing_gene_name_list = ['MYC', 'BRCA1', 'CCNE1', 'KRAS', 'GRB2', 'PIK3R1', 'BRCA2', 'FANCA', 'BRD4', 'UBC']

    connectivity = ppi_network.calc_connectivity_by_gene_name_list(testing_gene_name_list)
    print(connectivity)
