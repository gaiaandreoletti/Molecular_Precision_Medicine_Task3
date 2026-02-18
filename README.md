Molecular_Precision_Medicine_Task3.ipynb — The complete Colab notebook with:

Structured task prompt
Step-by-step expert solution code (6 steps)
Automated test case runner with scoring

example_data/ — 120 breast cancer patients, 500 genes

example_expression.csv — bulk RNA-seq matrix
example_clinical.csv — age, stage, ER/HER2 status, molecular subtype, OS

test_data/ — 3 independent cohorts (hidden from LLM)

test1: 100 patients (seed 101)
test2: 150 patients (seed 202)
test3: 90 patients (seed 303)


🧬 Task Summary: Molecular Patient Subtyping
The task asks an LLM to implement a full breast cancer subtyping pipeline:

HVG selection (top 100 by variance)
PCA (StandardScaler → 15 PCs, report 90% cumvar threshold)
K-Means clustering (k=3, evaluate with Silhouette + ARI)
Random Forest classifier (SelectKBest + 5-fold CV balanced accuracy)
Kruskal-Wallis DE genes (top 10 across clusters)
Clinical summary per cluster

7 test checks per dataset × 3 datasets = 21 checks total, scored out of 100 each with a passing threshold of ≥70/100. All seeds are fixed, making evaluation fully reproducible.
