import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure output directories exist
os.makedirs('results/figures', exist_ok=True)
os.makedirs('results/tables', exist_ok=True)

# Prevent pandas from truncating columns when printing to console
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# Load dataset
df = pd.read_csv('data/placental_geneexp.csv')

# Check unique values in SMOKER column to map labels accurately
print("SMOKER value counts:")
print(df['SMOKER'].value_counts(dropna=False))

# Genes to plot in order
genes = ['LGALS14', 'MGST1', 'ATXN3', 'SSU72', 'ICAM2']

# Determine smoker vs non-smoker values dynamically
unique_vals = sorted(df['SMOKER'].dropna().unique())
if set(unique_vals) == {0, 1}:
    non_smoker_val, smoker_val = 0, 1
elif set(unique_vals) == {False, True}:
    non_smoker_val, smoker_val = False, True
else:
    smoker_val = [v for v in unique_vals if 'non' not in str(v).lower()][0]
    non_smoker_val = [v for v in unique_vals if 'non' in str(v).lower()][0]

# Set up a single row of five subplots
fig, axes = plt.subplots(1, 5, figsize=(20, 4), sharey=False)

for idx, gene in enumerate(genes):
    ax = axes[idx]
    
    # Overlay of smoker in red and non-smoker in blue using seaborn kdeplot()
    sns.kdeplot(
        data=df[df['SMOKER'] == non_smoker_val],
        x=gene,
        color='blue',
        fill=True,
        alpha=0.35,
        label='Non-Smoker',
        ax=ax
    )
    sns.kdeplot(
        data=df[df['SMOKER'] == smoker_val],
        x=gene,
        color='red',
        fill=True,
        alpha=0.35,
        label='Smoker',
        ax=ax
    )
    
    ax.set_title(gene, fontsize=12, fontweight='bold')
    ax.set_xlabel('Expression Level', fontsize=10)
    if idx == 0:
        ax.set_ylabel('Density', fontsize=10)
    else:
        ax.set_ylabel('')
    
    # Show legend on the last subplot
    if idx == len(genes) - 1:
        ax.legend(title='Status', loc='upper right')

plt.suptitle('Placental Gene Expression Distributions: Smokers vs. Non-Smokers', fontsize=14, y=1.03)
plt.tight_layout()

# Save figure to results/figures
fig_path = 'results/figures/problem1a_gene_kdeplots.png'
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
print(f"\nFigure saved successfully to {fig_path}")
plt.show()

# Compute summary statistics and arrange column-wise (Smoker vs Non-Smoker as columns)
summary_stats = df.groupby('SMOKER')[genes].agg(['count', 'mean', 'std', 'median', 'min', 'max']).round(4)
summary_table = summary_stats.T
summary_table.index.names = ['Gene', 'Statistic']
summary_table.columns = ['Non-Smoker (0)', 'Smoker (1)']

# Reorder columns so Smoker (1) and Non-Smoker (0) appear cleanly
summary_table = summary_table[['Smoker (1)', 'Non-Smoker (0)']]

table_path = 'results/tables/problem1a_summary_stats.csv'
summary_table.to_csv(table_path)
print(f"Summary table saved successfully to {table_path}")

print("\nFull Summary Statistics (Column-Wise):")
print(summary_table)