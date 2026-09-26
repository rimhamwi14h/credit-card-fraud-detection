# Credit Card Fraud Detection

**An imbalanced classification project using Python and scikit-learn.**

Identify potentially fraudulent card transactions while balancing two competing goals: catching fraud and avoiding unnecessary alerts.

## At a glance

| Item | Details |
|---|---|
| Task | Binary classification: normal (`0`) or fraud (`1`) |
| Dataset | 284,807 transactions; 492 frauds (0.173%) |
| Features | `V1`–`V28` and `Amount`; the notebook's data has no `Time` column |
| Model | Scaled logistic regression with balanced class weights |
| Selected decision threshold | `0.70`, chosen from a validation split |
| Notebook | [`analyse_fraude.ipynb`](analyse_fraude.ipynb) |

## Why this problem is interesting

Fraud accounts for less than 0.2% of the transactions. A classifier that calls every transaction normal would achieve about **99.83% accuracy** while finding **zero frauds**. For this reason, the project focuses on fraud **precision**, **recall**, and the **confusion matrix**, rather than accuracy alone.

- **Precision:** Of the transactions flagged as fraud, how many actually were fraud?
- **Recall:** Of the actual frauds, how many did the model catch?

## Approach

1. Inspect the data, check for missing values, and measure class imbalance.
2. Separate the features (`X`) from the target (`Class`), then create a stratified 80/20 train/test split.
3. Compare against a `DummyClassifier` that always predicts the most frequent class.
4. Train a `Pipeline` with `StandardScaler` and `LogisticRegression(class_weight="balanced")`. The pipeline fits the scaler on training data only.
5. Split the training portion again to create a validation set. Compare decision thresholds on that validation set, then apply the selected threshold to the test portion.

The `StandardScaler` addresses differences in the **scale of feature values**. Balanced class weights address the **rarity of fraud labels**. These solve different problems.

## Results

| Model / threshold | Fraud precision | Fraud recall | Fraud detected | False alerts | Fraud missed |
|---|---:|---:|---:|---:|---:|
| Always normal baseline | — | 0% | 0 | 0 | 98 |
| Logistic regression, `0.50` | 5.86% | 91.84% | 90 / 98 | 1,447 | 8 |
| Logistic regression, `0.70` | **11.59%** | **89.80%** | **88 / 98** | **671** | **10** |

*The table shows the test split. The threshold was selected using a separate validation split of the training data.*

Raising the threshold from `0.50` to `0.70` reduced false alerts by **776**, while the model missed **two additional frauds**. A higher threshold makes the model more selective. Neither threshold is universally best: the choice depends on the cost of missed fraud versus the cost of reviewing a false alert.

## Run the notebook

1. Create a Python environment and install the notebook dependencies:

   ```bash
   python -m pip install numpy pandas scikit-learn notebook
   ```

2. Open `analyse_fraude.ipynb` in Jupyter or VS Code and select that environment as the kernel.
3. Check the notebook's data-loading cell and make the dataset available at the path it uses. If you use the local CSV, keep it in `data/`, which is excluded from Git.
4. Restart the kernel and run all cells in order.

The public source is the [ULB credit card fraud dataset on Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud). The original CSV may include `Time`; the analysis here uses the 30-column version **without** `Time`.

## Limitations

- The test set was inspected earlier during exploration, so the reported test comparison is **exploratory**, not a fully independent final estimate.
- Only a small set of thresholds and one train/validation split were explored. Results may vary with a different sample.
- Balanced class weights can improve fraud recall while creating many false alerts; threshold selection should reflect actual review costs in a real application.
- This is a learning project, not a production fraud detection system.
