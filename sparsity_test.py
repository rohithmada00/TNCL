from contextlib import redirect_stdout
from models import Solver, SolverArgs
import pandas as pd
import numpy as np
import random

def sparsity_test():
    np.random.seed(42)
    random.seed(42)
    p_list = [25, 50, 75, 100]
    d_list = [0.2, 0.4, 0.6, 0.8]
    lambda_val = 0.001
    rho_val = 0.1

    results = []

    for p in p_list:
        for d in d_list:
            args = SolverArgs(p=p, d=int(d * p), lambda_param=lambda_val, rho=rho_val, num_rep=10)
            solver = Solver(args)
            data = solver.solve()
            metrics = solver.evaluate(data)
            
            # Flatten and add identifiers
            row = {'p': p, 'd': d}
            row.update(metrics)
            results.append(row)

    # Convert to DataFrame and save as CSV
    df = pd.DataFrame(results)
    df.to_csv('sparsity_test_rho_same.csv', index=False)
    print("Sparsity test is complete and results are saved to sparsity_test.csv")



if __name__ == "__main__":
    with open("sparsity_test.txt", "w") as f:
        with redirect_stdout(f):
            sparsity_test()