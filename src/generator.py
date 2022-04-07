import src.utils as utils
from PyMiniSolvers import minisolvers
import numpy as np


class BaseCNFGenerator():
    """The base class for all CNF generators."""

    def generate_k_clause(self, *args):
        "Generates a k-clause"
        raise NotImplementedError

    def generate_formula(self, *args):
        "Generates unlabeled CNF instances."
        raise NotImplementedError


class UniformGenerator(BaseCNFGenerator):
    """Implements the uniformly random CNF generator."""
    def __init__(self, min_n, max_n, min_k, max_k, min_r, max_r):
        self.min_n = min_n  # The minimum number of variables for a random CNF instance
        self.max_n = max_n  # The maximum number of variables for a random CNF instance
        self.min_k = min_k  # The minimum clause size for a random CNF instance
        self.max_k = max_k  # The maximun clause size for a random CNF instance
        self.min_r = min_r  # The minimum radius for a random CNF instance
        self.max_r = max_r  # The maximum radius for a random CNF instance
  
    def generate_k_clause(self, n, k):
        vars = np.random.choice(n, size=k, replace=False)
        return [v + 1 if np.random.rand() < 0.5 else -(v + 1) for v in vars]

    def generate_formula(self):
        n = np.random.randint(self.min_n, self.max_n + 1)  # Number of variables
        r = np.random.uniform(self.min_r, self.max_r)  # raduis
        m = int(n * r)  # Number of clauses

        formula = []
        for _ in range(m):
            k = np.random.randint(self.min_k, min(self.max_k, n)+1)
            clause = self.generate_k_clause(n, k)
            formula.append(clause)

        return n, r, m, formula


class SRnGenerator(BaseCNFGenerator):
    """Implements the SR(n) distribution over pairs of random SAT problems on n variables.
    Selsam, D., et al. Learning SAT Solvers from a Simple Bit Supervision. 2019.
    https://arxiv.org/pdf/1802.03685.pdf"""

    def __init__(self, n, p_bernoulli=0.7, p_geometric=0.4):
        self.n = n
        self.p_b = p_bernoulli
        self.p_g = p_geometric
        self.S = minisolvers.MinisatSolver()
        for _ in range(n):
            self.S.new_var()
    
    def generate_k_clause(self, k):
        vars = np.random.choice(self.n, size=k, replace=False)
        return [v + 1 if np.random.rand() < 0.5 else -(v + 1) for v in vars]
    
    def generate_formula(self):
        formula_unsat = []
        while self.S.solve():
            k = 1 + np.random.binomial(n=1, p=self.p_b) + np.random.geometric(self.p_g)
            if k > self.n:
                k = self.n
            clause = self.generate_k_clause(k)
            formula_unsat.append(clause)
            self.S.add_clause(clause)
        
        formula_sat = formula_unsat.copy()
        clause_sat = clause.copy()
        index =  np.random.randint(0, high=len(clause_sat), size=None, dtype=int)
        clause_sat[index] = -clause_sat[index]
        formula_sat[-1] = clause_sat
  
        m = len(formula_sat)
        r = m/float(self.n)
        formula = [formula_unsat, formula_sat]

        return self.n, r, m, formula


    