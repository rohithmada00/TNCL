from contextlib import redirect_stdout
import csv
import itertools
import numpy as np
import random
from models import Solver, SolverArgs


def grid_search():
    np.random.seed(42)
    random.seed(42)
    p_values = [25, 50, 75, 100]
    lambda_grid = [0.01, 0.05, 0.1, 0.2]
    rho_grid = [0.1, 0.5, 1.0, 2.0]

    output_file = 'grid_search_1037_results.csv'
    is_first = True  # Write header only once

    for p in p_values:
        best_loss = float('inf')
        best_params = None

        for idx, (lamb, rho) in enumerate(itertools.product(lambda_grid, rho_grid)):

            args = SolverArgs(
                p=p,
                lambda_param=lamb,
                rho=rho,
                num_rep=10,
                n_samples=1037
            )
            solver = Solver(args)
            data = solver.solve()
            metrics = solver.evaluate(data)

            result_row = {
                'p': p,
                'lambda': lamb,
                'rho': rho,
                'd': args.d,
                'n_samples': args.n_samples,
                **metrics
            }

            # Append to CSV immediately for fault tolerance
            with open(output_file, mode='a', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=result_row.keys())
                if is_first:
                    writer.writeheader()
                    is_first = False
                writer.writerow(result_row)

            if metrics['avg_kll'] < best_loss:
                best_loss = metrics['avg_kll']
                best_params = result_row

        print(f"Best for p={p}:", best_params)

    print("Grid search completed. Results saved to:", output_file)

if __name__ == "__main__":
    with open("grid_search_reg_logs.txt", "w") as f:
        with redirect_stdout(f):
            grid_search()