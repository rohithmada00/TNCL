import pandas as pd
import matplotlib.pyplot as plt

csv_file = 'results/unregularized/scale_test/scale_test_rho_same.csv'  
output_file = 'f1_vs_tau_all_p.png'

df = pd.read_csv(csv_file)

plt.figure(figsize=(10, 7))

for p_value in df['p'].unique():
    subset = df[df['p'] == p_value].sort_values(by='tau')
    plt.errorbar(
        subset['tau'],
        subset['avg_f1'],
        yerr=subset['std_f1'],
        fmt='-o',
        capsize=4,
        label=f'p = {p_value}'
    )

plt.xscale('log')
plt.xlabel('tau (log scale)')
plt.ylabel('F1 Score')
plt.title('F1 Score vs. tau for Different p Values')
plt.legend(title='p')
plt.grid(True)
plt.tight_layout()

plt.savefig(output_file, dpi=300)
print(f'Plot saved as {output_file}')
