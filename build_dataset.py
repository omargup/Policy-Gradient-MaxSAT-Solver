# Building the dataset for the paper
#
# This script generates the following instances.
# For k=3 and n=[20, 30, 40]:
# 5 Uniform random instances with r=2.0
# 5 Uniform random instances with r=2.5
# 5 Uniform random instances with r=3.0
# 5 Uniform random instances with r=3.5
# 5 Uniform random instances with r=4.0
# 5 Uniform random instances with r=4.5
#
# For n=[20, 30, 40]:
# 5 SR instances with k = 1 + B(0.7) + G(0.4)
# ==============================================================================

from src.sat_generator import URGenerator, SRGenerator
import numpy as np
from tqdm import tqdm


# Build Uniform random instaces
dir_name = 'data'
n_list = [20, 30, 40]
r_list = [2.0, 2.5, 3.0, 3.5, 4.0, 4.5]
k = 3
num_instances = 5

np.random.seed(98702)

for n in tqdm(n_list):
    for r in r_list:
        # Instantiate a sat generator
        sat_gen = URGenerator(min_n = n,
                              max_n = n,
                              min_k = k,
                              max_k = k,
                              min_m = int(n * r),
                              max_m = int(n * r))
        for i in range(1, num_instances + 1):
            # Create a uniform random sat formula
            n, m, r, formula = sat_gen.generate_formula()

            # Saving the formula
            filename = sat_gen.get_filename(dir_name, i)
            sat_gen.save(n, formula, filename)


# Build SR random instances
dir_name = 'data'
n_list = [20, 30, 40]
p_bernoulli = 0.7 
p_geometric = 0.4
num_instances = 5

np.random.seed(14650)

for n in tqdm(n_list):
    # Instantiate a sat generator
    sat_gen = SRGenerator(n = n,
                          p_bernoulli = p_bernoulli,
                          p_geometric = p_geometric)
    
    for i in range(1, num_instances + 1):
        # Create a uniform random sat formula
        n, m, r, [formula_unsat, formula_sat] = sat_gen.generate_formula()

        # Saving the sat formula. Unsat is useless.
        filename = sat_gen.get_filename(dir_name, True, i)
        sat_gen.save(n, formula_sat, filename)