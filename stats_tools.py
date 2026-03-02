import numpy as np
import pandas as pd
import time
from typing import Union


class GeneStatsCalculator:
    def __init__(self, tumor_somatic_df: pd.DataFrame, d_matrix_df: pd.DataFrame, ppi: str,
                 num_testing: int = 10000, num_genes: int = 5, seed: int = 50):
        """

        Initialize the GeneStatsCalculator class.
        
        Parameters:
        tumor_somatic_df (pd.DataFrame): DataFrame containing tumor somatic data.
        d_matrix_df (pd.DataFrame): DataFrame containing d matrix data.
        ppi (str): PPI (Protein-Protein Interaction) data.
        NUM_TESTING (int): Number of testing iterations (default is 10000).
        K (int): Number of random numbers to generate (default is 5).
        SEED (int): Seed value for random number generation (default is 50).
        """
        self.res_stats_df = None
        self.tumor_somatic_df = tumor_somatic_df
        self.d_matrix_df = d_matrix_df
        assert self.tumor_somatic_df.columns.equals(self.d_matrix_df.columns), ("The columns of tumor_mut_bin and "
                                                                                "d_matrix_df are not equal.")
        self.ppi = ppi
        self.NUM_TESTING = num_testing
        self.K = num_genes
        self.SEED = seed
        self.gene_idxs = self.generate_random_numbers()

    @property
    def stats_result(self):
        return self.res_stats_df

    def generate_random_numbers(self) -> list[list[int]]:
        """
        Generate random numbers.
        
        Returns:
        list[list[int]]: list of randomly generated numbers.
        """
        np.random.seed(self.SEED)
        results = []
        for _ in range(self.NUM_TESTING):
            random_numbers = np.random.choice(self.d_matrix_df.shape[1], size = self.K, replace = False)
            results.append(random_numbers.tolist())
        return results

    def _get_columns_by_index(self, gene_idxs: list[list[int]]) -> list[list[Union[str, float]]]:
        """
        Get columns by index.
        
        Parameters:
        gene_idxs (list[list[int]]): list of gene indices.
        
        Returns:
        list[list[Union[str, float]]]: list of columns by index.
        """
        results = []
        for group in gene_idxs:
            row_res = []
            selected_columns = self.d_matrix_df.iloc[:, group]
            gene_names = selected_columns.columns.tolist()
            for name in gene_names:
                row_res.append(name)
            score = self.calc_score(gene_names)
            for s in score:
                row_res.append(s)
            results.append(row_res)
        return results

    def calc_score(self, genes):
        return GeneStatsCalculator.default_calc_score(
            genes, self.tumor_somatic_df,
            self.d_matrix_df, self.ppi
        )

    @staticmethod
    def default_calc_score(self, genes, tumor_somatic_df, d_matrix_df, ppi):
        print("Please setting the default_calc_score method outside the class")
        return None

    def add_test_genes(self, genes: list[str]) -> None:
        """
        Add test genes to the result statistics DataFrame.
        
        Parameters:
        genes (list[str]): list of gene names.
        """
        scores = self.calc_score(genes)
        stats_row = [genes, scores]
        stats_row = [item for sublist in stats_row for item in sublist]
        stats_row_df = pd.DataFrame([stats_row], columns = self.res_stats_df.columns)
        self.res_stats_df = pd.concat([stats_row_df, self.res_stats_df], ignore_index = True)
        self.res_stats_df = self.res_stats_df \
            .sort_values(by = ["score", "co", "me"], ascending = False) \
            .reset_index(drop = True)

    def calculate_stats(self) -> pd.DataFrame:
        """
        Calculate the statistics.

        Returns:
        pd.DataFrame: The updated DataFrame containing the statistics.
        """
        # Measure execution time
        start_time = time.time()

        column_names = [f"gene_{i}" for i in range(self.K)] + ["co", "me", "ppi_conn", "dt", "score"]
        results = self._get_columns_by_index(self.gene_idxs)
        self.res_stats_df = pd.DataFrame(results, columns = column_names)
        self.res_stats_df = self.res_stats_df[self.res_stats_df["score"] > 0.0]
        self.res_stats_df = self.res_stats_df \
            .sort_values(by = ["score", "co", "me"], ascending = False) \
            .reset_index(drop = True)

        # Calculate execution time
        execution_time = time.time() - start_time
        print("Execution time:", execution_time, " s")

        return self.res_stats_df

    def print_top_result(self, k=0):
        if self.stats_result is None:
            self.calculate_stats()

        for g in self.stats_result.iloc[k, :self.K]:
            print(g)
