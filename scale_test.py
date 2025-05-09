import pandas as pd
from contextlib import redirect_stdout
from models import Solver, SolverArgs
import numpy as np
import random

def scale_test():
    np.random.seed(42)
    random.seed(42)
    p_list = [25, 50, 75, 100]
    tau_list = [0.01, 0.1, 1, 5, 25, 50, 100]
    d_fixed = 15
    lambda_val = 0.001
    rho_val = 0.1

    output_file = 'scale_test_d_percent.csv'
    is_first = True  # For writing header only once

    for p in p_list:
        for tau in tau_list:
            args = SolverArgs(p=p, d=p*0.8, const=tau, lambda_param=lambda_val, rho=rho_val, num_rep=10)
            solver = Solver(args)
            data = solver.solve()
            metrics = solver.evaluate(data)

            row = {'p': p, 'tau': tau}
            row.update(metrics)
            df_row = pd.DataFrame([row])

            df_row.to_csv(output_file, mode='a', header=is_first, index=False)
            is_first = False

    print("Scale test is complete and results are saved incrementally to scale_test.csv")


if __name__ == "__main__":
    with open("scale_test.txt", "w") as f:
        with redirect_stdout(f):
            scale_test()
