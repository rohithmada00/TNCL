import pandas as pd
from contextlib import redirect_stdout
from helper_functions import Solver, SolverArgs, evaluate, save_to_pickle
import numpy as np
import random

def scale_test():
    p_list = [25, 30]
    tau = [0.1, 1, 5, 10, 25, 50, 100]
    d_fixed = 7
    lambda_val = 0.1
    rho_val = 0.1

    f1_results_ratio = {p: [] for p in p_list}
    data_list = {p: [] for p in p_list}

    for p in p_list:
        args = SolverArgs(p=p, d=d_fixed, lambda_param=lambda_val, rho=rho_val, num_rep=3, iterations=100)
        solver = Solver(args)
        for t in tau:
            solver.args.const = t
            data = solver.solve()
            metrics = evaluate(data)
            data_list[p].append(data)
            f1_results_ratio[p].append(metrics['avg_f1'])

    # plt.figure(figsize=(10, 6))
    # for p in p_list:
    #     plt.plot(tau, f1_results_ratio[p], marker='o', label=f'p={p}')

    # plt.xlabel(r"$\tau = N/(d^2 \log p)$")
    # plt.ylabel('Average F1 Score')
    # plt.title(r'F1 Score vs $\tau$')
    # plt.legend()
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()
    save_to_pickle(data_list, 'scale_data_list.pkl')
    save_to_pickle(f1_results_ratio, 'scale_f1_results.pkl')
    
    print("Scale test is complete and results are saved incrementally to scale_test.csv")


if __name__ == "__main__":
    with open("scale_test_d_varying.txt", "w") as f:
        with redirect_stdout(f):
            scale_test()
