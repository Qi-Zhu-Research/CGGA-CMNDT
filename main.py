import pandas as pd
import numpy as np
from numba import njit, jit
from numba import jit, prange

import os
import sys

from data_read_tools import load_from_pickle, save_to_pickle

# 导入配置参数
from config import *

# 数据加载
tumor_mut_bin = load_from_pickle(f"./data/{CANCER_TYPE}/tumor_mut_bin.pickle")
d_matrix_df = load_from_pickle(f"./data/{CANCER_TYPE}/d_matrix_df.pickle")
ppi = load_from_pickle(f"./data/{CANCER_TYPE}/ppi.pickle")

from genetic_algorithm_method import genetic_algorithm
from genetic_algorithm_method import fitness_fn
from genetic_algorithm_method import get_gene_names_by_idxs

from methods import calc_score

d_matrix_trim_df = d_matrix_df.mask(
    d_matrix_df > 0.0, 1.0
)

# 遗传算法执行
best_fitness_val, best_sol, fitness_obs = genetic_algorithm(
    fitness_fn=fitness_fn,
    population_size=POPULATION_SIZE,
    chromosome_total_length=tumor_mut_bin.shape[1],
    chromosome_length=K,
    elite_size=ELITE_SIZE,
    mutation_rate=MUTATION_RATE,
    crossover_probability=CROSSOVER_PROB,
    tumor_mut_bin=tumor_mut_bin,
    gname_list=tumor_mut_bin.columns,
    max_generations=MAX_GEN,
    fitness_max_min_type="max",
    global_env=globals(),
)