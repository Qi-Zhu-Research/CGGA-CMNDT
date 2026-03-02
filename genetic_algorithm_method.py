import numpy as np
from collections import defaultdict
import time
import pandas as pd
from numba import njit
import numpy as np
import pandas as pd
from numba import njit
from datetime import timedelta

from methods import calc_score


def init_choice_population(
    population_size,
    chromosome_total_length, chromosome_length,
):

    # Initialize a population with zeros
    population = np.zeros((population_size, chromosome_length), dtype = np.int64)

    for i in range(population_size):
        population[i] = np.random.choice(range(chromosome_total_length), size = chromosome_length, replace = False)

    return population

def elite_strategy(population, elite_size, fitness_fn, fitness_max_min_type = "max"):
    population_size = len(population)
    fitness_values = np.empty(population_size, dtype = np.float64)

    for i in range(population_size):
        try:
            fitness_values[i] = fitness_fn(population[i])
        except Exception as e:
            print(f"Exception occurred with population[{i}]: {population[i]}")
            print(e)
            raise

    if fitness_max_min_type == "max":
        elite_indices = np.argsort(fitness_values)[::-1][:elite_size]
    elif fitness_max_min_type == "min":
        elite_indices = np.argsort(fitness_values)[:elite_size]

    elite_population = np.empty((elite_size, population.shape[1]), dtype = population.dtype)
    for i in range(elite_size):
        elite_population[i] = population[elite_indices[i]]
    return elite_population, fitness_values


def calc_fitness_sum(population, fitness_fn, gname_list):
    fitness_sum = 0
    for individual in population:
        fitness_sum += fitness_fn(individual, gname_list)
    return fitness_sum


def roulette_wheel_selection(population, fitness_sum, fitness_fn):
    random_num = np.random.uniform(0, fitness_sum)
    current_sum = 0
    for i in range(len(population)):
        current_sum += fitness_fn(population[i])
        if current_sum > random_num:
            return population[i]
    return population[len(population) - 1]


def crossover_fn(parent1, parent2, crossover_probability, chromosome_total_length):
    if np.random.rand() >= crossover_probability:
        return parent1, parent2

    # Step 1: Find common elements, assume the length is L_comm
    common_elements = np.intersect1d(parent1, parent2)

    # Step 2: Find elements in parent1 not in parent2, and vice versa
    parent1_diff = np.setdiff1d(parent1, common_elements)
    parent2_diff = np.setdiff1d(parent2, common_elements)

    # Determine the number of elements to choose, which is floor((L - L_comm) / 2)
    num_elements = (len(parent1) - len(common_elements)) // 2
    if num_elements == 0:
        return parent1, parent2

    # Randomly choose elements from the different parts for the remaining part of the offspring
    if parent1_diff.size > 0 and parent2_diff.size > 0:
        chosen_from_parent1 = np.random.choice(parent1_diff, num_elements, replace = False)
        chosen_from_parent2 = np.random.choice(parent2_diff, num_elements, replace = False)
        remaining_parent1 = np.setdiff1d(parent1_diff, chosen_from_parent1)
        remaining_parent2 = np.setdiff1d(parent2_diff, chosen_from_parent2)
        offspring1 = np.concatenate([common_elements, chosen_from_parent1, remaining_parent1])
        offspring2 = np.concatenate([common_elements, chosen_from_parent2, remaining_parent2])
    else:
        offspring1 = common_elements
        offspring2 = common_elements

    # Check for invalid solutions
    for offspring in [offspring1, offspring2]:
        for digit in offspring:
            if digit < 0 or digit >= chromosome_total_length:
                print(f"Invalid solution in crossover: {digit} with parent <{parent1}, {parent2}>")
                # You can handle this error as you see fit
    return offspring1, offspring2


def mutation_choice_fn(individual, mutation_rate,
                       chromosome_total_length,
                       seed = np.NINF):
    if seed != np.NINF:
        np.random.seed(seed)

    for i in range(0, len(individual)):
        if np.random.rand() < mutation_rate:
            choice_digit = np.random.randint(0, chromosome_total_length - 1)
            while True:
                if choice_digit not in individual:
                    break
                choice_digit = np.random.randint(0, chromosome_total_length - 1)

            individual[i] = choice_digit
            # Check for invalid solutions
            if choice_digit < 0 or choice_digit >= chromosome_total_length:
                print(f"Invalid solution in mutation: {choice_digit}")
                # You can handle this error as you see fit
            break

    return individual


def common_best_pop_pattern_search(origin_pop, thres_prob, chromosome_total_length):
    count_dict = defaultdict(int)
    total = len(origin_pop)
    individual_size = origin_pop[0].size

    # Count the occurrences of each number in the first two positions
    for individual in origin_pop:
        for digit in individual:
            count_dict[digit] += 1

    # Calculate the probability of each number and select the numbers with a probability greater than thres_prob
    common_pattern = [number for number, count in count_dict.items() if count / total >= thres_prob]

    # Determine the number of random numbers to generate
    m = individual_size - len(common_pattern)

    # Create an array of all possible numbers excluding the common_pattern
    possible_numbers = np.setdiff1d(np.arange(chromosome_total_length), common_pattern)

    # Generate the new individual
    new_individual = common_pattern + list(np.random.choice(possible_numbers, m, replace = False))

    return new_individual


def gen_common_best_pop_pattern_search_pop(origin_pop, thres_prob, theta_out, chromosome_total_length):
    count_dict = defaultdict(int)
    total = len(origin_pop)
    individual_size = origin_pop[0].size

    # Count the occurrences of each number in the first two positions
    for individual in origin_pop:
        for digit in individual:
            count_dict[digit] += 1

    # Calculate the probability of each number and select the numbers with a probability greater than thres_prob
    common_pattern = [number for number, count in count_dict.items() if
                      count / total >= thres_prob and 0 <= number < chromosome_total_length]

    # Determine the number of random numbers to generate
    m = individual_size - len(common_pattern)

    # Create an array of all possible numbers excluding the common_pattern
    possible_numbers = np.setdiff1d(np.arange(chromosome_total_length), common_pattern)

    # Generate the new population
    new_population = []
    for _ in range(int(theta_out * len(origin_pop))):
        new_individual = common_pattern + list(np.random.choice(possible_numbers, m, replace = False))
        new_population.append(new_individual)
    new_population = np.array(new_population)
    return new_population


def generate_offspring(population, population_size, elite_size, offspring_origin,
                       gen, maxgen,
                       from_index, end_index,
                       fitness_sum, fitness_fn,
                       crossover_probability, mutation_rate,
                       chromosome_total_length, chromosome_length,
                       roulette_wheel_selection_fn, crossover_fn, mutation_fn,
                       tumor_mut_bin, gname_list, global_env
    ):
    offspring = offspring_origin.copy()
    offspring_index, offspring_end_index = from_index, end_index
    while offspring_index < offspring_end_index:
        start_time = time.time()

        indices = np.random.randint(0, population.shape[0], 2)
        parent1, parent2 = population[indices]

        children = crossover_fn(parent1, parent2, crossover_probability, chromosome_total_length)

        for child in children:
            # for digit in child:
            #     if digit > chromosome_total_length or digit < 0:
            #         print(f"{digit} error before mutation")
            child = mutation_fn(child, mutation_rate, chromosome_total_length, np.NINF)
            # for digit in child:
            #     if digit > chromosome_total_length or digit < 0:
            #         print(f"{digit} error after mutation")
            offspring[offspring_index] = child
            offspring_index += 1
            if offspring_index >= offspring_end_index:
                break

    # Calculate the sampling ratio for offspring_origin population
    sampling_ratio = 0.2 + 0.4 * (gen / maxgen)

    # Sample from the offspring_origin population
    num_samples = int(len(offspring_origin) * sampling_ratio)
    row_indices = np.random.choice(offspring_origin.shape[0], size = num_samples, replace = False)
    pop_into_comm = offspring_origin[row_indices]

    # Calculate the output ratio for gen_common_best_pop_pattern_search
    output_ratio = 0.6 - 0.4 * (gen / maxgen)
    # Generate the common best population pattern search
    pop_comm = gen_common_best_pop_pattern_search_pop(
        pop_into_comm, 0.5, output_ratio, chromosome_total_length,
    )

    # Combine offspring and pop_comm, calculate fitness for the combined population,
    # get indices of individuals with top fitness values, and update offspring with top individuals

    # Find the best chromosome from the original population and add it to the offspring
    elite_pop = population[:elite_size]
    # print(elite_pop)
    best_fitness_val, best_chromosome = fitness_fn(population[0], tumor_mut_bin, gname_list, global_env), population[0]
    combined_population = np.concatenate((offspring, elite_pop, pop_comm, [best_chromosome]))
    for i in range(len(combined_population)):
        for digit in range(len(combined_population[i])):
            if combined_population[i][digit] < 0 or combined_population[i][digit] >= chromosome_total_length:
                print(f"{i}-th chrome for {digit} with value {combined_population[i][digit]}")
    fitness_values = np.array([fitness_fn(individual, tumor_mut_bin, gname_list, global_env) for individual in combined_population])
    top_indices = np.argsort(fitness_values)[::-1][:population_size]
    offspring = combined_population[top_indices]

    return offspring


def calc_max_fitness_val(population, fitness_fn, tumor_mut_bin, gname_list):
    max_fitness_val = fitness_fn(population[0], tumor_mut_bin, gname_list, global_env)
    max_fitness_individual = population[0]
    for i in range(1, population.shape[0]):
        try:
            cur_fitness = fitness_fn(population[i], tumor_mut_bin, gname_list, global_env)
        except Exception as e:
            print(f"{i}-th solution error.")
        if cur_fitness > max_fitness_val:
            max_fitness_val = cur_fitness
            max_fitness_individual = population[i]
    return max_fitness_val, max_fitness_individual


def genetic_algorithm(
    fitness_fn,
    population_size, chromosome_total_length, chromosome_length,
    elite_size, mutation_rate, crossover_probability,
    max_generations, tumor_mut_bin, gname_list, fitness_max_min_type = "max",
    global_env = None,
):
    np.random.seed(179)

    population = init_choice_population(
        population_size, chromosome_total_length, chromosome_length
    )
    fitness_values = np.array([fitness_fn(individual, tumor_mut_bin, gname_list, global_env) for individual in population])
    sorted_indices = np.argsort(fitness_values)[::-1]
    population = population[sorted_indices]

    fitness_every_gen_observer = np.zeros((max_generations, ), dtype = np.float64)

    def train_fn(max_generations, population, maxt):
        # 进行遗传算法的迭代
        best_fitness = None
        best_fitness_streak = 0
        current_best_fitness = 0

        # 进行遗传算法的迭代
        gen = 0
        for gen in range(max_generations):

            offspring = np.empty_like(population)
            fitness_sum = -1

            offspring_from_index, offspring_end_index = 0, population_size
            offspring_start_time = time.time()
            offspring = generate_offspring(population, population.shape[0], elite_size, offspring,
                                           gen, max_generations,
                                           offspring_from_index, offspring_end_index,
                                           fitness_sum, fitness_fn,
                                           crossover_probability, mutation_rate,
                                           chromosome_total_length, chromosome_length,
                                           roulette_wheel_selection, crossover_fn, mutation_choice_fn,
                                           tumor_mut_bin, gname_list, global_env
                                           )

            population = offspring

            fitness_every_gen_observer[gen], max_fitness_individual = fitness_fn(population[0], tumor_mut_bin, gname_list, global_env), population[0]

            current_best_fitness = fitness_every_gen_observer[gen]
            # 检查最优适应度值是否在连续的 maxt 代中保持不变
            if best_fitness is None or current_best_fitness != best_fitness:
                best_fitness = current_best_fitness
                best_fitness_streak = 0
            else:
                best_fitness_streak += 1
            if best_fitness_streak >= maxt:

                break

        print(
            f"Individual {gname_list[max_fitness_individual.astype(int)]} converges on fitness: {fitness_every_gen_observer[gen]} ..."
        )
        return population

    population = train_fn(max_generations, population, maxt = 20)
    population_size = len(population)
    fitness_values = np.empty((population_size, chromosome_length + 1))

    for i in range(population_size):
        fitness_value = fitness_fn(population[i], tumor_mut_bin, gname_list, global_env)
        fitness_values[i] = np.concatenate((np.array([fitness_value]), population[i]))

    if fitness_max_min_type == "max":
        sorted_indices = np.argsort(fitness_values[:, 0])[::-1]
    elif fitness_max_min_type == "min":
        sorted_indices = np.argsort(fitness_values[:, 0])
    fitness_values = fitness_values[sorted_indices]

    return fitness_values[0][0], fitness_values[0][1:], fitness_every_gen_observer

def get_gene_names_by_idxs(gene_idxs, gname_list):
    gene_idxs = np.array(gene_idxs).flatten()
    return list(gname_list[gene_idxs])

def fitness_fn(genes, tumor_mut_bin, gname_list, global_env):
    genes = get_gene_names_by_idxs(genes, gname_list)
    d_matrix_df = global_env["d_matrix_df"]
    d_matrix_trim_df = global_env["d_matrix_trim_df"]
    ppi = global_env["ppi"]

    co, me, ppi_conn, dt, total_score = calc_score(
        genes, tumor_mut_bin, d_matrix_df, d_matrix_trim_df, ppi
    )
    return total_score
