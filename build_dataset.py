from src.sat_generator import URGenerator, SRGenerator
from src.solvers import minisat_solver

import numpy as np
from tqdm import tqdm


def toy_dataset():
    """
    Builds a toy dataset with Satisfiable Random SAT Formulas.
    This function generates 5 uniform random sat instances
    for each of the following combinations of parameters:
        k=3
        n=[5, 10, 15]
        r=[1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]:
    """
    print("Building toy_dataset")

    # Build Uniform random instances
    dir_name = 'data/toy'
    data_name = 'toy'
    n_list = [5, 10, 15]
    r_list = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
    k = 3
    num_instances = 5

    np.random.seed(855104)

    for n in tqdm(n_list):
        for r in r_list:
            m = int(np.around(n * r))
            # Instantiate a sat generator
            sat_gen = URGenerator(min_n=n,
                                  max_n=n,
                                  min_k=k,
                                  max_k=k,
                                  min_m=m,
                                  max_m=m)

            sat_clauses = 0
            while sat_clauses < num_instances:
                # Create a uniform random sat formula
                n, m, r, formula = sat_gen.generate_formula()

                # Using minisat_solver to check satifiability
                assignment, is_sat = minisat_solver(n, formula)

                if is_sat:
                    sat_clauses += 1

                    # Saving the formula
                    filename = sat_gen.get_filename(dir_name, data_name, sat_clauses)
                    sat_gen.save(n, formula, filename)


def rand_dataset():
    """
    This function generates the following instances:
    For k=3 and n=[20, 30, ..., 100]:
        - 11 Uniform random instances with r=[1.0, 1.5, ..., 5.0]

    """
    print("Building rand_dataset")

    # Build Uniform random instances
    dir_name = 'data/rand'
    data_name = 'rand'
    n_list = [20, 30, 40, 50, 60, 70, 80, 90, 100]
    r_list = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.2, 4.4, 4.6, 4.8, 5.0]
    k = 3
    num_instances = 11

    np.random.seed(98702)

    for n in tqdm(n_list):
        for r in r_list:
            m = int(np.around(n * r))
            dir_name1 = f'{dir_name}/{n:04d}/{m:04d}'
            
            # Instantiate a sat generator
            sat_gen = URGenerator(min_n=n,
                                max_n=n,
                                min_k=k,
                                max_k=k,
                                min_m=m,
                                max_m=m)

            for i in range(1, num_instances + 1):
                # Create a uniform random sat formula
                n, m, r, formula = sat_gen.generate_formula()

                # Saving the formula
                filename = sat_gen.get_filename(dir_name1, data_name, i)
                sat_gen.save(n, formula, filename)


def sat_rand_dataset():
    """
    Builds a dataset with Satisfiable Random SAT Formulas.
    This function generates 5 uniform random sat instances
    for each of the following combinations of parameters:
        k=3
        n=[20, 30, 40]
        r=[2.0, 2.5, 3.0, 3.5, 4.0, 4.5]:
    """
    print("Building sat_rand_dataset")

    # Build Uniform random instances
    dir_name = 'data/sat_rand'
    data_name = 'sat_rand'
    n_list = [20, 30, 40, 50, 60, 70, 80, 90, 100]
    r_list = [2.0, 2.5, 3.0, 3.5, 4.0, 4.5]
    k = 3
    num_instances = 5

    np.random.seed(104873)

    for n in tqdm(n_list):
        for r in r_list:
            m = int(np.around(n * r))
            # Instantiate a sat generator
            sat_gen = URGenerator(min_n=n,
                                  max_n=n,
                                  min_k=k,
                                  max_k=k,
                                  min_m=m,
                                  max_m=m)

            sat_clauses = 0
            while sat_clauses < num_instances:
                # Create a uniform random sat formula
                n, m, r, formula = sat_gen.generate_formula()

                # Using minisat_solver to check satifiability
                assignment, is_sat = minisat_solver(n, formula)

                if is_sat:
                    sat_clauses += 1

                    # Saving the formula
                    filename = sat_gen.get_filename(dir_name, data_name, sat_clauses)
                    sat_gen.save(n, formula, filename)


def sr_dataset():
    """
    This function generates the following instances:
    For n=[20, 30, 40]:
        - 5 SR instances with k = 1 + B(0.7) + G(0.4)
    """
    print("Building sr_dataset")

    # Build SR random instances
    dir_name = 'data/sr'
    data_name = 'sr'
    n_list = [20, 30, 40]
    p_bernoulli = 0.7 
    p_geometric = 0.4
    num_instances = 5

    np.random.seed(14650)

    for n in tqdm(n_list):
        # Instantiate a sat generator
        sat_gen = SRGenerator(n=n,
                            p_bernoulli=p_bernoulli,
                            p_geometric=p_geometric)
        
        for i in range(1, num_instances + 1):
            # Create a uniform random sat formula
            n, m, r, [formula_unsat, formula_sat] = sat_gen.generate_formula()

            # Saving the sat formula.
            filename = sat_gen.get_filename(dir_name, data_name, True, i)
            sat_gen.save(n, formula_sat, filename)
            
            # Saving the unsat formula
            filename = sat_gen.get_filename(dir_name, data_name, False, i)
            sat_gen.save(n, formula_unsat, filename)


if __name__ == '__main__':
    toy_dataset()
    rand_dataset()
    sat_rand_dataset()
    sr_dataset()