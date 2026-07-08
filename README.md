# Machine Learning Pipeline for Psychometric Scale Analysis
**Self-Directed Independent Project (2026)** — bridging psychometric measurement with computational machine learning.

## Overview
This Jupyter Notebook (`psychometric_ml_pipeline.ipynb`) builds a reproducible machine learning pipeline for high-dimensional psychological scale analysis:
1. Simulated psychometric scale dataset (N=500, 30 latent dimensions)
2. Random Forest & linear SVM classifiers with 5-fold cross-validation (mean accuracy: 85% / 81%)
3. Dual-path feature selection: RF feature importance + Recursive Feature Elimination (RFE) — reducing 30 dimensions to 10 core predictive indicators
4. Predictive feature importance visualization via Matplotlib

The full workflow is directly transferable to real-participant scale data for psychometric and behavioral prediction research.

## Tech Stack
**Python** · scikit-learn · NumPy · pandas · Matplotlib · Jupyter Notebook
