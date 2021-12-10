import numpy as np

class BaseCNFGenerator(object):
    "The base class for all CNF generators."

    def __init__(self, min_n, max_n, min_k, max_k, min_r, max_r):
        self.min_n = min_n #The minimum number of variables for a random CNF instance
        self.max_n = max_n #The maximum number of variables for a random CNF instance
        self.min_k =min_k #The minimum clause size for a random CNF instance
        self.max_k =max_k #The maximun clause size for a random CNF instance
        self.min_r = min_r #The minimum radius for a random CNF instance
        self.max_r = max_r # The maximum radius for a random CNF instance
        
    def generate_formula(self):
        "Generates unlabeled CNF instances."
        pass


class UniformCNFGenerator(BaseCNFGenerator):
    "Implements the uniformly random CNF generator."

    def __init__(self, min_n, max_n, min_k, max_k, min_r, max_r):
        
        super(UniformCNFGenerator, self).__init__(min_n, max_n, min_k, max_k, min_r, max_r)
  
    def generate_k_clause(self, n, k):
        vars = np.random.choice(n, size=k, replace=False)
        return [v + 1 if np.random.rand() < 0.5 else -(v + 1) for v in vars]

    def generate_formula(self):
        n = np.random.randint(self.min_n, self.max_n + 1) #Number of variables
        r = np.random.uniform(self.min_r, self.max_r) #raduis
        m = int(n * r) #Number of clauses

        formula = []
        for _ in range(m):
            k = np.random.randint(self.min_k, min(self.max_k, n)+1)
            clause = self.generate_k_clause(n, k)
            formula.append(clause)

        return n, r, m, formula
