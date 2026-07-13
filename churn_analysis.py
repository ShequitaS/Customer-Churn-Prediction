"""
Retail Banking Customer Churn — Prediction & Retention Targeting
Shequita Stevenson

Dataset: public 10,000-customer retail bank churn dataset.
Goal is NOT "get the highest accuracy". Goal is: decide WHO the retention team
should call, and prove that calling them makes money.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (roc_auc_score, average_precision_score, roc_curve,
                             precision_recall_curve, classification_report,
                             confusion_matrix)

RNG = 42
plt.rcParams.update({"figure.dpi": 130, "font.size": 9})

# ----------------------------------------------------------------------
# 1. LOAD
# ----------------------------------------------------------------------
df = pd.read_csv("bank_churn.csv")
df = df.rename(columns={
    "Num Of Products": "NumProducts",
    "Has Credit Card": "HasCreditCard",
    "Is Active Member": "IsActiveMember",
    "Estimated Salary": "EstimatedSalary",
})
df = df.drop(columns=["CustomerId", "Surname"])  # identifiers, not features

print("=" * 70)
print("SECTION 1 — DATA")
print("=" * 70)
print(f"Rows: {len(df):,}   Columns: {df.shape[1]}   Nulls: {df.isna().sum().sum()}")
base_rate = df["Churn"].mean()
print(f"Churn rate: {base_rate:.1%}")
print(f"\nNOTE: a model that predicts 'nobody churns' is {1-base_rate:.1%} accurate")
print("and worth exactly zero dollars. Accuracy is the wrong metric here.")

# ----------------------------------------------------------------------
# 2. WHO CHURNS — segment analysis
# ----------------------------------------------------------------------
print("\n" + "=" * 70)
print("SECTION 2 — WHO CHURNS")
print("=" * 70)

def seg(col, label=None):
    t = df.groupby(col, observed=True)["Churn"].agg(["mean", "size"])
    t.columns = ["churn_rate", "customers"]
    t["churn_rate"] = (t["churn_rate"] * 100).round(1)
    t["lift_vs_base"] = (t["churn_rate"] / (base_rate * 100)).round(2)
    print(f"\n--- {label or col}")
    print(t.to_string())
    return t

df["AgeBand"] = pd.cut(df["Age"], [17, 30, 40, 50, 60, 100],
                       labels=["18-30", "31-40", "41-50", "51-60", "60+"])
df["BalanceBand"] = pd.cut(df["Balance"], [-1, 0.01, 50000, 100000, 150000, 1e9],
                           labels=["Zero", "<50k", "50-100k", "100-150k", "150k+"])

seg("IsActiveMember", "Active member (digital/product engagement)")
seg("NumProducts", "Number of products held")
seg("AgeBand", "Age band")
seg("Geography", "Geography")
seg("Gender", "Gender")
seg("BalanceBand", "Balance band")

# ----------------------------------------------------------------------
# 3. FEATURES + MODELS
# ----------------------------------------------------------------------
print("\n" + "=" * 70)
print("SECTION 3 — MODELING")
print("=" * 70)

X = df.drop(columns=["Churn", "AgeBand", "BalanceBand"])
y = df["Churn"]

cat = ["Geography", "Gender"]
num = [c for c in X.columns if c not in cat]

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=RNG)
print(f"Train {len(X_tr):,} / Test {len(X_te):,} (stratified)")

pre = ColumnTransformer([
    ("num", StandardScaler(), num),
    ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), cat),
])

models = {
    "Logistic Regression": Pipeline([
        ("pre", pre),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced",
                                   random_state=RNG)),
    ]),
    "Gradient Boosting": Pipeline([
        ("pre", pre),
        ("clf", HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.06, max_leaf_nodes=15,
            class_weight="balanced", random_state=RNG)),
    ]),
}

results = {}
cv = StratifiedKFold(5, shuffle=True, random_state=RNG)
for name, pipe in models.items():
    cv_auc = cross_val_score(pipe, X_tr, y_tr, cv=cv, scoring="roc_auc")
    pipe.fit(X_tr, y_tr)
    p = pipe.predict_proba(X_te)[:, 1]
    results[name] = {
        "pipe": pipe, "proba": p,
        "auc": roc_auc_score(y_te, p),
        "ap": average_precision_score(y_te, p),
        "cv": cv_auc,
    }
    print(f"\n{name}")
    print(f"  CV ROC-AUC (5-fold) : {cv_auc.mean():.3f} (+/- {cv_auc.std():.3f})")
    print(f"  Test ROC-AUC        : {results[name]['auc']:.3f}")
    print(f"  Test PR-AUC         : {results[name]['ap']:.3f}  (baseline = {base_rate:.3f})")

best_name = max(results, key=lambda k: results[k]["auc"])
best = results[best_name]
print(f"\n>> Selected: {best_name}")

# Coefficients from the logistic model — this is the "why", and it is what you
# explain to a retention manager who does not care about gradient boosting.
lr = results["Logistic Regression"]["pipe"]
feat_names = (num + list(lr.named_steps["pre"]
              .named_transformers_["cat"].get_feature_names_out(cat)))
coefs = pd.Series(lr.named_steps["clf"].coef_[0], index=feat_names)
odds = np.exp(coefs).round(2)
drivers = pd.DataFrame({"coef": coefs.round(3), "odds_ratio": odds}) \
            .sort_values("coef", ascending=False)
print("\nChurn drivers (logistic regression, standardized):")
print("odds_ratio > 1 = raises churn odds;  < 1 = protective")
print(drivers.to_string())

# Permutation importance on the selected model
from sklearn.inspection import permutation_importance
perm = permutation_importance(best["pipe"], X_te, y_te, n_repeats=10,
                              random_state=RNG, scoring="roc_auc")
imp = pd.Series(perm.importances_mean, index=X.columns).sort_values(ascending=False)
print(f"\nPermutation importance ({best_name}, drop in test ROC-AUC):")
print(imp.round(4).to_string())

# ----------------------------------------------------------------------
# 4. THE PART THAT MATTERS — what threshold, and does it pay?
# ----------------------------------------------------------------------
print("\n" + "=" * 70)
print("SECTION 4 — COST-BENEFIT: WHERE DO WE SET THE THRESHOLD?")
print("=" * 70)

# Assumptions. These are stated, not hidden, and every one of them is arguable.
OFFER_COST   = 100    # $ cost of the retention intervention per customer contacted
CLV          = 1500   # $ value of retaining a customer for the next period
SAVE_RATE    = 0.30   # % of genuinely-at-risk customers the offer actually saves

print(f"""Assumptions (change these and the answer changes — that is the point):
  Retention offer cost      : ${OFFER_COST} per customer contacted
  Customer lifetime value   : ${CLV}
  Offer success rate        : {SAVE_RATE:.0%} of true churners contacted are saved

Economics per contacted customer:
  True churner  -> spend ${OFFER_COST}, save them {SAVE_RATE:.0%} of the time
                   => expected value = {SAVE_RATE} x ${CLV} - ${OFFER_COST} = ${SAVE_RATE*CLV - OFFER_COST:,.0f}
  Non-churner   -> spend ${OFFER_COST} on someone who was never leaving
                   => expected value = -${OFFER_COST}
""")

proba = best["proba"]
rows = []
for t in np.arange(0.05, 0.96, 0.01):
    pred = (proba >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_te, pred).ravel()
    contacted = tp + fp
    net = tp * (SAVE_RATE * CLV - OFFER_COST) - fp * OFFER_COST
    rows.append({
        "threshold": round(t, 2),
        "contacted": contacted,
        "true_churners_caught": tp,
        "churners_missed": fn,
        "wasted_contacts": fp,
        "precision": tp / contacted if contacted else 0,
        "recall": tp / (tp + fn),
        "net_value": net,
    })
econ = pd.DataFrame(rows)

best_row = econ.loc[econ["net_value"].idxmax()]
default_row = econ.loc[(econ["threshold"] - 0.50).abs().idxmin()]
n_test = len(y_te)

print("Net value by threshold (test set, n = {:,}):".format(n_test))
print(econ[econ["threshold"].isin([0.10,0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.90])]
      .to_string(index=False, float_format=lambda v: f"{v:,.2f}"))

print(f"""
>> OPTIMAL THRESHOLD: {best_row.threshold:.2f}
   Contact {int(best_row.contacted):,} of {n_test:,} customers ({best_row.contacted/n_test:.0%} of the book)
   Catch {int(best_row.true_churners_caught):,} of {int(best_row.true_churners_caught + best_row.churners_missed):,} churners (recall {best_row.recall:.0%})
   Precision {best_row.precision:.0%} — {int(best_row.wasted_contacts):,} contacts wasted on customers who would have stayed
   NET VALUE: ${best_row.net_value:,.0f}

>> The naive 0.50 cutoff most people ship:
   NET VALUE: ${default_row.net_value:,.0f}
   Difference: ${best_row.net_value - default_row.net_value:,.0f} on a {n_test:,}-customer test set.

>> Contacting EVERY customer indiscriminately:
   NET VALUE: ${(y_te.sum() * (SAVE_RATE*CLV - OFFER_COST) - (n_test - y_te.sum()) * OFFER_COST):,.0f}
""")

print("Classification report at the optimal threshold:")
print(classification_report(y_te, (proba >= best_row.threshold).astype(int),
                            target_names=["Stayed", "Churned"], digits=3))

# ----------------------------------------------------------------------
# 5. WHO TO CALL FIRST — decile table for the retention team
# ----------------------------------------------------------------------
print("=" * 70)
print("SECTION 5 — TARGETING LIST (what the retention team actually receives)")
print("=" * 70)

scored = X_te.copy()
scored["churn_prob"] = proba
scored["actual_churn"] = y_te.values
# qcut labels ascend with the value, so D1 = lowest risk, D10 = highest risk.
scored["decile"] = pd.qcut(scored["churn_prob"], 10,
                           labels=[f"D{i}" for i in range(1, 11)])

dec = scored.groupby("decile", observed=True).agg(
    customers=("actual_churn", "size"),
    actual_churn_rate=("actual_churn", "mean"),
    avg_predicted_prob=("churn_prob", "mean"),
    avg_balance=("Balance", "mean"),
    pct_active=("IsActiveMember", "mean"),
    avg_age=("Age", "mean"),
)
dec = dec.reindex([f"D{i}" for i in range(10, 0, -1)])  # highest risk first
dec["lift"] = (dec["actual_churn_rate"] / base_rate).round(2)
dec["actual_churn_rate"] = (dec["actual_churn_rate"] * 100).round(1)
dec["avg_predicted_prob"] = (dec["avg_predicted_prob"] * 100).round(1)
dec["pct_active"] = (dec["pct_active"] * 100).round(1)
dec["avg_balance"] = dec["avg_balance"].round(0)
dec["avg_age"] = dec["avg_age"].round(1)
print("\nD10 = HIGHEST predicted risk. 'lift' = multiple of the 20.4% base churn rate.")
print(dec.to_string())

captured = scored[scored["decile"].isin(["D10", "D9"])]["actual_churn"].sum()
print(f"\n>> The top 2 deciles are 20% of the book and contain "
      f"{captured / y_te.sum():.0%} of all churners.")

scored.sort_values("churn_prob", ascending=False).head(200) \
      .to_csv("retention_call_list_top200.csv", index=False)

# ----------------------------------------------------------------------
# 6. FIGURES
# ----------------------------------------------------------------------
fig, ax = plt.subplots(2, 2, figsize=(11, 8))

a = ax[0, 0]
for n, r in results.items():
    fpr, tpr, _ = roc_curve(y_te, r["proba"])
    a.plot(fpr, tpr, label=f"{n} (AUC {r['auc']:.3f})")
a.plot([0, 1], [0, 1], "k--", lw=0.8, label="Random")
a.set_xlabel("False positive rate"); a.set_ylabel("True positive rate")
a.set_title("ROC curve"); a.legend(fontsize=7)

a = ax[0, 1]
a.plot(econ["threshold"], econ["net_value"], color="#2c7fb8")
a.axvline(best_row.threshold, color="green", ls="--", lw=1,
          label=f"Optimal {best_row.threshold:.2f} (${best_row.net_value:,.0f})")
a.axvline(0.50, color="red", ls="--", lw=1,
          label=f"Default 0.50 (${default_row.net_value:,.0f})")
a.axhline(0, color="k", lw=0.6)
a.set_xlabel("Classification threshold"); a.set_ylabel("Net value ($)")
a.set_title("Net retention value by threshold"); a.legend(fontsize=7)

a = ax[1, 0]
d = drivers.drop(index=[i for i in drivers.index if i == "EstimatedSalary"], errors="ignore")
colors = ["#d7301f" if v > 0 else "#2b8cbe" for v in d["coef"]]
a.barh(d.index, d["coef"], color=colors)
a.axvline(0, color="k", lw=0.8)
a.set_title("Churn drivers (logistic coefficients)")
a.set_xlabel("<- protective        raises churn risk ->")
a.tick_params(labelsize=7)

a = ax[1, 1]
a.bar(dec.index.astype(str), dec["actual_churn_rate"], color="#41b6c4")
a.axhline(base_rate * 100, color="red", ls="--", lw=1,
          label=f"Base rate {base_rate*100:.1f}%")
a.set_title("Actual churn rate by predicted-risk decile")
a.set_ylabel("Churn rate (%)"); a.legend(fontsize=7)

plt.tight_layout()
plt.savefig("churn_analysis.png", bbox_inches="tight")
print("\nSaved: churn_analysis.png")
print("Saved: retention_call_list_top200.csv")

econ.to_csv("threshold_economics.csv", index=False)
dec.to_csv("risk_deciles.csv")
print("Saved: threshold_economics.csv, risk_deciles.csv")
