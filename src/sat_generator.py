import src.utils as utils
import PyMiniSolvers.minisolvers as minisolvers
import numpy as np
import os


class BaseCNFGenerator():
    """The base class for all CNF generators."""

    def generate_k_clause(self, *args):
        "Generates a k-clause"
        raise NotImplementedError

    def generate_formula(self, *args):
        "Generates unlabeled CNF instances."
        raise NotImplementedError
    
    def get_filename(self, *args):
        "Generates the file name of an instance."
        raise NotImplementedError
    
    def save(self, n, formula, filename):
        # create folders
        path = os.path.dirname(filename)
        os.makedirs(path, exist_ok=True)

        with open(filename, 'w') as f:
            f.write(f"p cnf {n} {len(formula)}\n")
            for clause in formula:
                for literal in clause:
                    f.write(f"{literal} ")
                f.write("0\n")
            f.write(f"%")


class URGenerator(BaseCNFGenerator):
    """Implements the uniformly random CNF generator."""
    def __init__(self, min_n=20, max_n=20, min_k=3, max_k=3, min_m=80, max_m=80):
        self.min_n = min_n  # The minimum number of variables for a random CNF instance
        self.max_n = max_n  # The maximum number of variables for a random CNF instance
        self.min_k = min_k  # The minimum clause size for a random CNF instance
        self.max_k = max_k  # The maximun clause size for a random CNF instance
        self.min_m = min_m  # The minimum number of clauses for a random CNF instance
        self.max_m = max_m  # The maximum number of clauses for a random CNF instance
  
    def generate_k_clause(self, n, k):
        vars = np.random.choice(n, size=k, replace=False)
        return [v + 1 if np.random.rand() < 0.5 else -(v + 1) for v in vars]

    def generate_formula(self):
        n = np.random.randint(self.min_n, self.max_n + 1)  # Number of variables
        m = np.random.randint(self.min_m, self.max_m + 1)  # Number of clauses

        formula = []
        for _ in range(m):
            k = np.random.randint(self.min_k, min(self.max_k, n)+1)
            clause = self.generate_k_clause(n, k)
            formula.append(clause)

        r = m/float(n)
        
        return n, m, r, formula
    

    def get_filename(self, dir_name, data_name, i):
        """
        dir_name : Name of the directory.
        data_name : Name of the dataset.
        i : Number of the instance.
        """
        #dir_name = f"{dir_name}/{data_name}/"
        
        if self.min_n == self.max_n:
            n_name = f"_n={self.min_n:04d}"
        else:
            n_name = f"_minn={self.min_n:04d}_maxn={self.max_n:04d}"
        
        if self.min_k == self.max_k:
            k_name = f"_k={self.min_k:02d}"
        else:
            k_name = f"_mink={self.min_k:02d}_maxk={self.max_k:02d}"
        
        if self.min_m == self.max_m:
            m_name = f"_m={self.min_m:04d}"
        else:
            m_name = f"_minm={self.min_m:04d}_maxm={self.max_m:04d}"

        filename = dir_name + "/" + data_name + n_name + k_name + m_name + f"_i={i}" + ".cnf"
        
        return filename



class SRGenerator(BaseCNFGenerator):
    """Implements the SR(n) distribution over pairs of random SAT problems on n variables.
    Selsam, D., et al. Learning SAT Solvers from a Simple Bit Supervision. 2019.
    https://arxiv.org/pdf/1802.03685.pdf"""

    def __init__(self, n=20, p_bernoulli=0.7, p_geometric=0.4):
        self.n = n
        self.p_b = p_bernoulli
        self.p_g = p_geometric
    
    def generate_k_clause(self, k):
        vars = np.random.choice(self.n, size=k, replace=False)
        return [v + 1 if np.random.rand() < 0.5 else -(v + 1) for v in vars]
    
    def generate_formula(self):
        S = minisolvers.MinisatSolver()
        for _ in range(self.n):
            S.new_var()

        formula_unsat = []
        
        while S.solve():
            k = 1 + np.random.binomial(n=1, p=self.p_b) + np.random.geometric(self.p_g)
            if k > self.n:
                k = self.n
            clause = self.generate_k_clause(k)
            formula_unsat.append(clause)
            S.add_clause(clause)

        formula_sat = formula_unsat.copy()
        clause_sat = clause.copy()
        index =  np.random.randint(0, high=len(clause_sat), size=None, dtype=int)
        clause_sat[index] = -clause_sat[index]
        formula_sat[-1] = clause_sat
  
        m = len(formula_sat)
        r = m/float(self.n)
        formula = [formula_unsat, formula_sat]

        return self.n, m, r, formula      
    

    def get_filename(self, dir_name, data_name, sat, i):
        """
        dir_name : Name of the directory.
        data_name : Name of the dataset.
        sat : Whether the clause is sat or not. 
        i : Number of the instance.
        """
        #dir_name = f"{dir_name}/{data_name}/"
        n_name = f"_n={self.n:04d}"
        b_name = f"_pb={self.p_b:.2f}"
        g_name = f"_pg={self.p_g:.2f}"
        if sat:
            s_name = f"_s=T"
        else:
            s_name = f"_s=F"

        filename = dir_name + "/" + data_name + n_name + "_1" + b_name + g_name + s_name + f"_i={i}" + ".cnf"
        
        return filename

    