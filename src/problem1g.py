import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

# Ensure output directories exist
os.makedirs('results/figures', exist_ok=True)
os.makedirs('results/tables', exist_ok=True)

# Load dataset
df = pd.read_csv('data/placental_geneexp.csv')

features = ['LGALS14', 'MGST1', 'ATXN3', 'SSU72', 'ICAM2']
X = df[features]
y = df['SMOKER']

# 80:20 train:test split (stratified to preserve class balance, random_state=10)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=10, stratify=y
)

print(f"Total samples: {len(df)} | Train samples: {len(X_train)} | Test samples: {len(X_test)}\n")

# Four model types from Problem 1(f) with default hyperparameters
# Note: probability=True is set on SVC so SHAP can explain class-1 (Smoker) probabilities consistently across all 4 models
models = [
    ('Quadratic Discriminant Analysis (QDA)', QuadraticDiscriminantAnalysis()),
    ('SVM (RBF Kernel)', SVC(kernel='rbf', probability=True, random_state=10)),
    ('Gaussian Naïve Bayes', GaussianNB()),
    ('Random Forest (100 Trees)', RandomForestClassifier(n_estimators=100, random_state=10))
]

shap_records = []
shap_values_by_model = {}

for model_name, clf in models:
    clf.fit(X_train, y_train)
    
    # Define prediction function returning P(SMOKER = 1)
    predict_fn = lambda data: clf.predict_proba(pd.DataFrame(data, columns=features))[:, 1]
    
    # Compute exact/permutation Shapley values using X_train as background reference
    explainer = shap.Explainer(predict_fn, X_train, seed=10)
    shap_exp = explainer(X_test)
    
    # Mean absolute SHAP value per feature across the test set
    mean_abs_shap = np.abs(shap_exp.values).mean(axis=0)
    shap_values_by_model[model_name] = mean_abs_shap
    
    # Rank features (1 = most important)
    ranks = pd.Series(mean_abs_shap).rank(ascending=False, method='min').astype(int).values
    
    for feat, val, rk in zip(features, mean_abs_shap, ranks):
        shap_records.append({
            'Model': model_name,
            'Feature': feat,
            'Mean_Abs_SHAP': round(float(val), 5),
            'Rank': int(rk)
        })

shap_df = pd.DataFrame(shap_records)

# Build pivot tables for Mean |SHAP| and Ranks across all 4 models
val_pivot = shap_df.pivot(index='Feature', columns='Model', values='Mean_Abs_SHAP')
rank_pivot = shap_df.pivot(index='Feature', columns='Model', values='Rank')

# Compute average rank and overall consensus rank
summary_table = pd.DataFrame(index=features)
for model_name, _ in models:
    summary_table[f'{model_name} (Mean |SHAP|)'] = val_pivot[model_name]
    summary_table[f'{model_name} (Rank)'] = rank_pivot[model_name]

summary_table['Average_Rank'] = rank_pivot.mean(axis=1).round(2)
summary_table['Overall_Rank'] = summary_table['Average_Rank'].rank(method='min').astype(int)
summary_table = summary_table.sort_values(by=['Average_Rank', 'Overall_Rank'])

# Save summary table to results/tables
table_path = 'results/tables/problem1g_shap_feature_rankings.csv'
summary_table.to_csv(table_path)
print(f"SHAP feature ranking table saved successfully to {table_path}\n")
print("Shapley Value Feature Importance & Rankings Across All 4 Models (80:20 Split):")
print(summary_table.to_string())

# ---------------------------------------------------------
# Visualization: 2-Panel Comparison (Mean |SHAP| & Rank Heatmap)
# ---------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), gridspec_kw={'width_ratios': [1.35, 1.0]})

# Panel A: Grouped Bar Chart of Mean |SHAP| Values
plot_df = shap_df.copy()
feature_order = summary_table.index.tolist()

sns.barplot(
    data=plot_df,
    x='Feature',
    y='Mean_Abs_SHAP',
    hue='Model',
    order=feature_order,
    palette=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'],
    edgecolor='black',
    linewidth=0.7,
    ax=axes[0]
)
axes[0].set_title('Mean Absolute SHAP Value by Feature & Model', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Gene Feature (Ordered by Overall Consensus Rank)', fontsize=11)
axes[0].set_ylabel('Mean |SHAP Value| (Impact on P(Smoker))', fontsize=11)
axes[0].grid(axis='y', linestyle='--', alpha=0.5)
axes[0].legend(title='Classifier', fontsize=9, title_fontsize=9.5, loc='upper right')

# Panel B: Feature Rank Matrix Heatmap (1 = Highest Importance)
rank_display = rank_pivot.reindex(index=feature_order, columns=[m[0] for m in models])
rank_display.columns = ['QDA', 'SVM (RBF)', 'Gaussian NB', 'Random Forest']

sns.heatmap(
    rank_display,
    annot=True,
    fmt='d',
    cmap='YlGnBu_r',
    cbar=False,
    ax=axes[1],
    annot_kws={'size': 13, 'weight': 'bold'},
    linewidths=1.2,
    linecolor='black'
)
axes[1].set_title('Feature Importance Ranks Across Models (1 = Highest)', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Classifier', fontsize=11)
axes[1].set_ylabel('')

plt.suptitle('Shapley (SHAP) Feature Importance Analysis Across Classifiers (80:20 Split)', fontsize=13.5, fontweight='bold', y=1.02)
plt.tight_layout()

fig_path = 'results/figures/problem1g_shap_importance.png'
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
print(f"\nFigure saved successfully to {fig_path}")
plt.show()