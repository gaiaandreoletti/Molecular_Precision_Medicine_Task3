"""
Expert Solution: Molecular Patient Subtyping from Bulk RNA-seq Data
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import (adjusted_rand_score, silhouette_score,
                             classification_report, confusion_matrix)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.feature_selection import SelectKBest, f_classif
from scipy.stats import kruskal
import warnings
warnings.filterwarnings('ignore')


def load_data(expression_csv, clinical_csv):
    expr = pd.read_csv(expression_csv)
    clin = pd.read_csv(clinical_csv)
    merged = clin.merge(expr, on='patient_id')
    gene_cols = [c for c in expr.columns if c.startswith('GENE_')]
    return merged, gene_cols


def select_highly_variable_genes(expr_matrix, top_n=100):
    """Select top-N genes by variance (HVG selection)."""
    variances = expr_matrix.var(axis=0)
    top_genes = variances.nlargest(top_n).index.tolist()
    return top_genes


def run_pca(expr_matrix, n_components=20):
    scaler = StandardScaler()
    scaled = scaler.fit_transform(expr_matrix)
    pca = PCA(n_components=n_components, random_state=42)
    pcs = pca.fit_transform(scaled)
    explained = pca.explained_variance_ratio_
    return pcs, explained, scaler, pca


def cluster_patients(pcs, n_clusters=3):
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
    labels = km.fit_predict(pcs)
    sil = silhouette_score(pcs, labels)
    return labels, sil, km


def evaluate_clustering(pred_labels, true_labels):
    ari = adjusted_rand_score(true_labels, pred_labels)
    return ari


def train_subtype_classifier(expr_hvg, true_labels, n_features=50):
    """Train RF classifier on HVG features with cross-validation."""
    selector = SelectKBest(f_classif, k=n_features)
    X_sel = selector.fit_transform(expr_hvg, true_labels)
    
    clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(clf, X_sel, true_labels, cv=cv, scoring='balanced_accuracy')
    
    clf.fit(X_sel, true_labels)
    return clf, selector, cv_scores


def differential_expression_top_genes(expr_hvg, labels, gene_names, top_n=10):
    """Kruskal-Wallis test per gene across subtypes, return top DE genes."""
    results = []
    for gene in gene_names:
        groups = [expr_hvg[labels == lbl][gene].values 
                  for lbl in np.unique(labels)]
        stat, pval = kruskal(*groups)
        results.append({'gene': gene, 'statistic': stat, 'pval': pval})
    de_df = pd.DataFrame(results).sort_values('pval').head(top_n)
    return de_df


def compute_subtype_summary(merged_df, pred_cluster_col='cluster'):
    """Per-subtype clinical summary."""
    summary = merged_df.groupby(pred_cluster_col).agg(
        n_patients=('patient_id', 'count'),
        mean_age=('age', 'mean'),
        er_positive_pct=('er_status', lambda x: (x == 'Positive').mean() * 100),
        median_os_months=('os_months', 'median'),
        event_rate=('event', 'mean')
    ).round(2)
    return summary


def run_pipeline(expression_csv, clinical_csv, output_prefix=None):
    print(f"\n{'='*60}")
    print(f"Running pipeline: {expression_csv}")
    print('='*60)

    # 1. Load
    merged, gene_cols = load_data(expression_csv, clinical_csv)
    expr_matrix = merged[gene_cols]
    true_labels = merged['molecular_subtype'].values
    print(f"Loaded {len(merged)} patients, {len(gene_cols)} genes")

    # 2. HVG selection
    hvg = select_highly_variable_genes(expr_matrix, top_n=100)
    expr_hvg = expr_matrix[hvg]
    print(f"Selected {len(hvg)} highly variable genes")

    # 3. PCA
    pcs, explained, scaler, pca_model = run_pca(expr_hvg, n_components=15)
    cumvar = np.cumsum(explained)
    n90 = int(np.searchsorted(cumvar, 0.90)) + 1
    print(f"PCs explaining 90% variance: {n90} | PC1 var: {explained[0]:.3f}")

    # 4. Clustering
    pred_labels, silhouette, km_model = cluster_patients(pcs[:, :10], n_clusters=3)
    ari = evaluate_clustering(pred_labels, true_labels)
    merged['cluster'] = pred_labels
    print(f"Clustering → Silhouette: {silhouette:.4f} | ARI: {ari:.4f}")

    # 5. Classifier
    clf, selector, cv_scores = train_subtype_classifier(expr_hvg.values, true_labels)
    mean_ba = cv_scores.mean()
    print(f"RF Classifier 5-CV Balanced Accuracy: {mean_ba:.4f} ± {cv_scores.std():.4f}")

    # 6. DE genes
    de_genes = differential_expression_top_genes(
        expr_hvg.reset_index(drop=True), pred_labels, hvg, top_n=10)
    print(f"\nTop 10 DE Genes across clusters:\n{de_genes[['gene','pval']].to_string(index=False)}")

    # 7. Subtype summary
    summary = compute_subtype_summary(merged, pred_cluster_col='cluster')
    print(f"\nSubtype Clinical Summary:\n{summary.to_string()}")

    # Build result dict
    result = {
        'n_patients': len(merged),
        'n_genes': len(gene_cols),
        'n_hvg': len(hvg),
        'pcs_explaining_90pct_variance': n90,
        'pc1_variance_explained': round(float(explained[0]), 4),
        'silhouette_score': round(float(silhouette), 4),
        'adjusted_rand_index': round(float(ari), 4),
        'rf_cv_balanced_accuracy_mean': round(float(mean_ba), 4),
        'rf_cv_balanced_accuracy_std': round(float(cv_scores.std()), 4),
        'top_de_genes': de_genes['gene'].tolist(),
        'subtype_summary': summary.to_dict()
    }
    return result


if __name__ == '__main__':
    result = run_pipeline(
        '/home/claude/precision_medicine/example_data/example_expression.csv',
        '/home/claude/precision_medicine/example_data/example_clinical.csv',
        output_prefix='example'
    )
    import json
    print("\n--- RESULTS JSON ---")
    print(json.dumps({k: v for k, v in result.items() if k != 'subtype_summary'}, indent=2))

def run_all_tests():
    import json
    test_sets = ['test1', 'test2', 'test3']
    all_results = {}
    for ts in test_sets:
        r = run_pipeline(
            f'/home/claude/precision_medicine/test_data/{ts}_expression.csv',
            f'/home/claude/precision_medicine/test_data/{ts}_clinical.csv'
        )
        all_results[ts] = {k: v for k, v in r.items() if k != 'subtype_summary'}
    print("\n\n=== ALL TEST RESULTS ===")
    print(json.dumps(all_results, indent=2))
    return all_results

run_all_tests()
