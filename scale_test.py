import pandas as pd
from contextlib import redirect_stdout
from models import Solver, SolverArgs

def scale_test():
    p_list = [25, 50, 75, 100]
    tau_list = [0.01, 0.1, 1, 5, 25, 50, 100]
    d_fixed = 15
    lambda_val = 0.1
    rho_val = 0.1

    results = []

    for p in p_list:
        for tau in tau_list:
            args = SolverArgs(p=p, d=d_fixed, const=tau, lambda_param=lambda_val, rho=rho_val, num_rep=10)
            solver = Solver(args)
            data = solver.solve()
            metrics = solver.evaluate(data)

            # Flatten and tag each metric row
            row = {'p': p, 'tau': tau}
            row.update(metrics)
            results.append(row)

    # Convert to DataFrame and export
    df = pd.DataFrame(results)
    df.to_csv('scale_test.csv', index=False)
    print("Scale test is complete and results are saved to scale_test.csv")



if __name__ == "__main__":
    with open("scale_test.txt", "w") as f:
        with redirect_stdout(f):
            scale_test()

