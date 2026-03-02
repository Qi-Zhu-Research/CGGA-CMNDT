import sys

import numpy as np
from numba import jit, prange
from numba import njit
import pandas as pd


@jit(parallel = True)
def mean_numba(a):
    res = []
    for i in prange(a.shape[0]):
        res.append(a[i, :].mean())

    return np.array(res)


@njit
def calc_d_matrix_nb(tumor_arr, normal_arr):
    normal_arr_mean = mean_numba(normal_arr)
    ret_arr = np.zeros(tumor_arr.shape)
    for i in range(tumor_arr.shape[0]):
        for j in range(tumor_arr.shape[1]):
            if tumor_arr[i][j] == 0 or normal_arr_mean[j] == 0:
                ret_arr[i][j] == 0
            else:
                abs_diff = np.absolute(np.log2(tumor_arr[i][j]) - np.log2(normal_arr_mean[j]))
                ret_arr[i][j] = abs_diff if abs_diff >= 1 else 0
    return ret_arr

def calc_d_matrix(tumor_df, normal_df):
    return pd.DataFrame(
        data=calc_d_matrix_nb(tumor_df.values, normal_df.values),
        index=tumor_df.index,
        columns=tumor_df.columns,
    )

# 类间差异性计算

@njit
def calc_intra_cluster_distance_nb(d_matrix, cell_type):
    unique_cell_types = np.unique(cell_type)
    ret_arr = np.zeros((unique_cell_types.size, d_matrix.shape[1]))
    for c in range(ret_arr.shape[0]):
        for j in range(ret_arr.shape[1]):
            type_c_idx = np.where(cell_type == c)[0]
            non_type_c_idx = np.where(cell_type != c)[0]
            d_cj = np.mean(d_matrix[type_c_idx, j]) - np.mean(d_matrix[non_type_c_idx, j])
            ret_arr[c, j] = d_cj
    mask = np.isnan(ret_arr) | ~np.isfinite(ret_arr)
    ret_arr = np.where(mask, 0, ret_arr)
    return ret_arr

def calc_intra_cluster_distance(d_matrix_df, reverse_ct_dict):
    intra_class_dis_matrix_vals = calc_intra_cluster_distance_nb(
        d_matrix_df.values,
        d_matrix_df.index.get_level_values("cell_type_encoded").values
    )

    dis_matrix_df = pd.DataFrame(
        intra_class_dis_matrix_vals,
        columns=d_matrix_df.columns,
    )

    dis_matrix_df.index.name="cell_type_encoded"
    dis_matrix_df["cell_type"] = np.array(list(reverse_ct_dict.values()))
    dis_matrix_df.set_index(['cell_type'], append=True, inplace=True)
    return dis_matrix_df

# 细胞间差异性计算

# 原版公式
@njit
def calc_dl_matrix_ori(d_sub_matrix, thres = 0):
    dl_matrix = np.zeros(d_sub_matrix.shape[0])

    for i in range(d_sub_matrix.shape[0]):
        max_val = np.max(d_sub_matrix[i, :])
        # 同时高表达
        # max_val = np.sum(d_sub_matrix[i, :])
        val = 1 if max_val > thres else 0
        dl_matrix[i] = val

    return dl_matrix


# 修改体现K个基因在一个细胞子集里面差异一致
@njit
def calc_dl_matrix(d_sub_matrix, thres = 0):
    # if d_sub_matrix.ndim == 1:
    #     d_sub_matrix = d_sub_matrix.reshape(-1, 1)

    dl_matrix = np.zeros(d_sub_matrix.shape[0])

    for i in range(d_sub_matrix.shape[0]):
        max_val = np.max(d_sub_matrix[i, :]) if d_sub_matrix.ndim > 1 else d_sub_matrix[i]
        # 同时高表达
        # max_val = np.sum(d_sub_matrix[i, :])
        val = 1 if max_val > thres else 0
        dl_matrix[i] = val

    return dl_matrix


@njit
def calc_dt_matrix_nb(d_sub_matrix, cell_type, thres = 0, if_print = False):
    max_diff_k = 0.0;
    min_diff_k = np.inf

    for c in np.unique(cell_type):
        type_c_idx = np.where(cell_type == c)[0]
        non_type_c_idx = np.where(cell_type != c)[0]
        c_matrix, non_c_matrix = d_sub_matrix[type_c_idx, :], d_sub_matrix[non_type_c_idx, :]
        c_dl_matrix = calc_dl_matrix(c_matrix, thres)
        non_c_dl_matrix = calc_dl_matrix(non_c_matrix, thres)
        c_k = np.sum(c_dl_matrix) / c_dl_matrix.shape[0]
        non_c_k = np.sum(non_c_dl_matrix) / non_c_dl_matrix.shape[0]
        # c_k = np.mean(c_dl_matrix)
        # non_c_k = np.mean(non_c_dl_matrix)
        diff_k = c_k - non_c_k
        if if_print:
            print(c, ", ", c_k, ", ", non_c_k)
        max_diff_k = diff_k if diff_k >= max_diff_k else max_diff_k
        min_diff_k = diff_k if diff_k <= min_diff_k else min_diff_k

    return max_diff_k


@njit
def calc_dt_matrix_v2_nb(d_sub_matrix, cell_type, thres = 0, if_print = False):
    max_c_k = 0.0;
    min_c_k = np.inf

    for c in np.unique(cell_type):
        type_c_idx = np.where(cell_type == c)[0]
        c_matrix = d_sub_matrix[type_c_idx, :]
        c_dl_matrix = calc_dl_matrix(c_matrix, thres)
        c_k = np.sum(c_dl_matrix) / c_dl_matrix.shape[0]
        c_k = np.mean(c_matrix)
        if if_print:
            print(c, ", ", c_k)
        max_c_k = c_k if c_k >= max_c_k else max_c_k
        min_c_k = c_k if c_k <= min_c_k else min_c_k

    return max_c_k - min_c_k


# 修改体现K个基因在一个细胞子集里面差异一致
# 每一条基因找最大差异，然后乘起来
@njit
def calc_dt_matrix_v3_nb(d_sub_matrix, cell_type, thres = 0, if_print = False):
    num_cells, num_genes = d_sub_matrix.shape

    max_cell_type_diff = 0.0;
    max_cell_type = np.inf;

    for c in np.unique(cell_type):

        cell_max_diff_res = 0.0

        for j in range(num_genes):
            # 找最大差异
            max_diff = 0.0;
            min_diff = np.inf;

            type_c_idx = np.where(cell_type == c)[0]
            c_matrix = d_sub_matrix[type_c_idx, j]
            c_dl_matrix = calc_dl_matrix(c_matrix, thres)

            non_type_c_idx = np.where(cell_type != c)[0]
            non_c_matrix = d_sub_matrix[non_type_c_idx, j]
            non_c_dl_matrix = calc_dl_matrix(non_c_matrix, thres)

            diff = np.sum(c_dl_matrix) / c_dl_matrix.shape[0] - np.sum(non_c_dl_matrix) / non_c_dl_matrix.shape[0]
            if if_print:
                # print(j, ", ", c, ", ", np.sum(c_dl_matrix))
                print(c, ", ", j, ", ", diff)
            max_diff = diff if diff >= max_diff else max_diff
            min_diff = diff if diff <= min_diff else min_diff

            # 第一种算法:
            # cell_max_diff_res += max_diff

            # 第二种算法(加起来,直接加diff):
            # cell_max_diff_res += diff

            # 2.2 给个list然后算总体熵 (详见草稿):

            # 2.4 大于0的个数:
            if if_print and c == 9:
                print("step1")
                print(diff)
                print(cell_max_diff_res)
            if diff > 0.3:
                diff = 1
            else:
                diff = 0
            cell_max_diff_res += diff
            if if_print and c == 9:
                print("after_step_1")
                print(diff)
                print(cell_max_diff_res)

            # 第三种算法(熵最大):
            # prob1 = np.sum(c_dl_matrix) / c_dl_matrix.shape[0]
            # prob2 = np.sum(non_c_dl_matrix) / non_c_dl_matrix.shape[0]
            # kl_divergence = prob1 * np.log2(prob1 / prob2) + (1 - prob1) * np.log2((1 - prob1) / (1 - prob2))

            # 第三种算法(熵最大) 3.1 除零错:
        #             epsilon = 1e-10

        #             prob1 = np.sum(c_dl_matrix) / c_dl_matrix.shape[0]
        #             prob2 = np.sum(non_c_dl_matrix) / non_c_dl_matrix.shape[0]

        #             # Add a small epsilon to avoid division by zero
        #             prob1 = prob1 + epsilon
        #             prob2 = prob2 + epsilon

        #             kl_divergence = prob1 * np.log2(prob1 / prob2) + (1 - prob1) * np.log2((1 - prob1) / (1 - prob2))

        #             cell_max_diff_res += kl_divergence

        # TODO: Continuing Thinking: 继续看kl_divergence的熵，二阶熵

        if cell_max_diff_res >= max_cell_type_diff:
            max_cell_type = c
        max_cell_type_diff = cell_max_diff_res if cell_max_diff_res >= max_cell_type_diff else max_cell_type_diff
    if if_print:
        print()
        print(max_cell_type)
        print(max_cell_type_diff)

    return max_cell_type_diff

# 覆盖度与互斥度计算

@jit(nopython = True, parallel = True, fastmath = True)
def calc_co_me_fitness_nb(so_mut_matrix):
    n_rows = so_mut_matrix.shape[0]
    n_cols = so_mut_matrix.shape[1]

    # 计算 eta 数组 (每行最大值)
    eta = np.empty(n_rows)
    for i in prange(n_rows):
        max_val = so_mut_matrix[i, 0]
        for j in range(1, n_cols):
            if so_mut_matrix[i, j] > max_val:
                max_val = so_mut_matrix[i, j]
        eta[i] = max_val

    # 计算 co
    eta_sum = 0.0
    for i in range(n_rows):
        eta_sum += eta[i]
    co = eta_sum / n_rows

    # 特殊情况处理
    if eta_sum == 0.0:
        return 0.0, 0.0
    if n_cols == 1:
        return co, co

    # 计算 me_sum (根据列数选择计算方式)
    me_sum = 0.0
    if n_cols == 2:
        for i in prange(n_rows):
            phi_i = 0.0
            for j in range(n_cols):
                phi_i += so_mut_matrix[i, j]
            me_sum += phi_i / 2.0  # 分母 = 2
    else:
        denom = n_cols * (n_cols - 1.0)  # 浮点分母
        for i in prange(n_rows):
            phi_i = 0.0
            for j in range(n_cols):
                phi_i += so_mut_matrix[i, j]
            me_sum += phi_i * (phi_i - 1.0) / denom

    me = me_sum / n_rows
    return co, me

# @njit
def calc_co_me_fitness(so_mut_matrix):
    eta = np.zeros(so_mut_matrix.shape[0])
    for i in range(so_mut_matrix.shape[0]):
        max_val_i = np.max(so_mut_matrix[i])
        eta[i] = max_val_i

    eta_sum = np.sum(eta)
    co = eta_sum / eta.shape[0]
    K = so_mut_matrix.shape[1]

    if eta_sum == 0:
        return 0.0, 0.0
    if K == 1:
        return co, co

    me_sum = 0.0
    for i in range(so_mut_matrix.shape[0]):  # Loop through each row
        phi_i = np.sum(so_mut_matrix[i, :])
        me_i = phi_i * (phi_i - 1) / (K * (K - 1))
        if K == 2:
            me_i = phi_i / (K * (K-1))
        me_sum += me_i
    me = me_sum / eta.shape[0]
    return co, me

def calc_score(genes, tumor_somatic_df, d_matrix_df, d_matrix_trim_df, ppi):
    co, me = calc_co_me_fitness(tumor_somatic_df[genes].values)
    ppi_conn = ppi.calc_connectivity_by_gene_name_list(
        genes
    )
    try:
        if all(gene in d_matrix_df.columns for gene in genes):
            dt = calc_dt_matrix_nb(
                d_matrix_df[genes].values,
                d_matrix_df.index.get_level_values("cell_type_encoded").values,
                2,
            )
        else:
            dt = 0.0
    except ZeroDivisionError as e:
        print(f"ZeroDivisionError occurred with genes: {genes}")

    total_score = (co - me) +  2 * ppi_conn + 1 * dt
    return co, me, ppi_conn, dt, total_score#, co_c, me_c

