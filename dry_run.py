from contextlib import redirect_stdout
from models import Solver, SolverArgs

def dry_run():
    defaultArgs = SolverArgs(p=25, num_rep=1, const=5, d=15)
    defaultSolver = Solver(defaultArgs)
    data = defaultSolver.solve()
    print("Results of the run:")
    print(data)

# Capture print output
if __name__ == "__main__":
    with open("dry_run_logs.txt", "w") as f:
        with redirect_stdout(f):
            dry_run()