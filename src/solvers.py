import src.utils as utils
from PyMiniSolvers import minisolvers


def random_solver(n, formula):
    # Create a random assignment
    assignment = utils.random_assignment(n=n)
    # Verifying the number of satisfied clauses
    is_sat, num_sat, eval_formula = utils.num_sat_clauses(formula, assignment)
    return assignment, num_sat


def minisat_solver(n, formula):
    S = minisolvers.MinisatSolver()
    for i in range(n):
        S.new_var()
    for clause in formula:
        S.add_clause(clause)
    assignment = None
    is_sat = S.solve()
    if is_sat:
        assignment = list(S.get_model())
        #is_sat, num_sat, eval_formula = utils.num_sat_clauses(formula, assignment)
    
    return assignment, is_sat
        
