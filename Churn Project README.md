# Customer Churn Prediction

Predicting which customers are likely to churn — and turning that prediction into a **targeted retention strategy** that captures most of the at-risk revenue while avoiding wasteful blanket outreach.

> **Headline result:** A model-driven list of **728 customers captures 76% of all churners**, generating an estimated **~$122K in net value** versus a blanket-outreach approach that would actually *lose* money.

---

## Problem

Customer churn is costly — but so is retention outreach done blindly. Calling every customer wastes money on people who were never going to leave. This project had two goals:

1. **Predict** which customers are at risk of churning.
2. **Act efficiently** — turn those predictions into a cost-effective decision about *who to actually contact*, rather than contacting everyone.

## Data

- **~10,000 customers**
- **Overall churn rate: 20.4%** — a class-imbalanced problem, which shaped the modeling and evaluation choices
- Customer demographic, account, and usage features *(customize this line to your actual feature set)*

## Approach

1. **Exploratory analysis & cleaning** — examined distributions, handled missing values, and prepared features for modeling.
2. **Handled class imbalance** — with only 20.4% churners, raw accuracy is misleading (a model predicting "no one churns" would be ~80% accurate and useless). Evaluation therefore centered on **ROC-AUC** and **PR-AUC**.
3. **Modeling** — built a **logistic regression baseline**, then a **gradient boosting** model for stronger performance.
4. **Business framing** — converted predicted churn probabilities into a ranked target list and modeled the cost/benefit of **targeted vs. blanket** outreach.

## Results

| Metric | Score |
| --- | --- |
| ROC-AUC | **0.87** |
| PR-AUC | **0.72** |

- The gradient boosting model meaningfully outperformed the logistic regression baseline.
- Ranking customers by predicted churn probability produced a **728-customer target list that captures 76% of all churners.**

## Business Impact

Translating the model into dollars is where it matters most:

| Strategy | Estimated value |
| --- | --- |
| **Targeted outreach** (the 728-customer list) | **~ +$100K** |
| **Blanket outreach** (contacting everyone) | **~ –$21K** |
| **Net value of the targeted strategy** | **~ $122K** |

**Recommendation: avoid blanket calls.** A focused, model-driven target list captures the large majority of at-risk revenue at a fraction of the cost — the gains from contacting non-churners don't justify the expense.

## Tech Stack

**Python** · pandas · scikit-learn (logistic regression, gradient boosting) · matplotlib / seaborn

## Repository

- `churn_analysis.ipynb` — full analysis, modeling, and evaluation *(rename to match your file)*
- `data/` — dataset or data description
- `README.md` — this file

---

*Built by Shequita Stevenson — M.S. Data Science, Boston University (Dec 2026).*
