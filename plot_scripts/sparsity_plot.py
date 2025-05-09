import pandas as pd
import matplotlib.pyplot as plt

csv_file = 'results/unregularized/sparsity_test/sparsity_test_rho_same.csv'

output_file = 'f1_vs_sparsity_all_p.png'

df = pd.read_csv(csv_file)

plt.figure(figsize=(10, 7))

for p_value in df['p'].unique():
    subset = df[df['p'] == p_value].sort_values(by='d')
    plt.errorbar(
        subset['d'],
        subset['avg_f1'],
        yerr=subset['std_f1'],
        fmt='-o',
        capsize=4,
        label=f'p = {p_value}'
    )

plt.xlabel('d (Sparsity Ratio)')
plt.ylabel('F1 Score')
plt.title('F1 Score vs. Sparsity Ratio (d) for Different p Values')
plt.legend(title='p')
plt.grid(True)
plt.tight_layout()

plt.savefig(output_file, dpi=300)
print(f'Plot saved as {output_file}')
