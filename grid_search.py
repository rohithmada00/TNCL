from contextlib import redirect_stdout
import csv
import itertools

from models import Solver, SolverArgs


def grid_search():
    p_values = [25, 50, 75, 100]
    lambda_grid = [0.01, 0.05, 0.1, 0.2]
    rho_grid = [0.5, 1.0, 2.0]

    results = []

    for p in p_values:
        best_loss = float('inf')
        best_params = None

        for lamb, rho in itertools.product(lambda_grid, rho_grid):
            args = SolverArgs(p=p, lambda_param=lamb, rho=rho, num_rep=10, n_samples=2000)
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
            results.append(result_row)

            if metrics['avg_kll'] < best_loss:
                best_loss = metrics['avg_kll']
                best_params = result_row

        print(f"Best for p={p}:", best_params)

    # Save to CSV
    with open('grid_search_results.csv', mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print("Results saved to 'grid_search_results.csv'")


if __name__ == "__main__":
    with open("grid_search_logs.txt", "w") as f:
        with redirect_stdout(f):
            grid_search()