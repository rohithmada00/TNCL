import pandas as pd
import matplotlib.pyplot as plt

csv_file = 'results/grid_search/with_random_seed/grid_search_results.csv'  
output_file = 'metrics_vs_p.png'
fixed_lambda = 0.01
fixed_rho = 0.1

df = pd.read_csv(csv_file)

filtered_df = df[(df['lambda'] == fixed_lambda) & (df['rho'] == fixed_rho)].copy()
filtered_df = filtered_df.sort_values(by='p')

metrics = ['avg_acc', 'avg_f1', 'avg_rate', 'avg_kll']
std_metrics = ['std_acc', 'std_f1', 'std_rate', 'std_kll']

plt.figure(figsize=(10, 6))
for avg_metric, std_metric in zip(metrics, std_metrics):
    plt.errorbar(
        filtered_df['p'],
        filtered_df[avg_metric],
        yerr=filtered_df[std_metric],
        fmt='-o',
        label=f'{avg_metric} ± std'
    )

plt.xlabel('p (Dimension)')
plt.ylabel('Metric Value')
plt.title(f'Metrics vs. p with Std Devs (lambda={fixed_lambda}, rho={fixed_rho})')
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(output_file, dpi=300)
print(f'Plot saved as {output_file}')