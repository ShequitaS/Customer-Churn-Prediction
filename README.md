# Retail Banking Churn: Who Should the Retention Team Call?

**Shequita Stevenson** | Python (pandas, scikit-learn) · SQL · Tableau/Power BI

---

## The question I actually asked

Most churn projects answer *"can I predict who leaves?"* That question is not
worth money. A retention team has a finite number of analysts, a finite offer
budget, and a manager who has to justify both.

So the question I answered is:

> **Given a fixed retention budget, which customers do we call, and does calling
> them make more money than not calling them?**

That reframing changes the whole analysis — the metric, the threshold, and the
deliverable.

---

## Data

Public retail banking dataset: **10,000 customers**, 10 features, no missing
values. **20.4% churned.**

The first thing worth saying out loud: a model that predicts *"nobody churns"*
is **79.6% accurate** and worth exactly zero dollars. Accuracy is the wrong
metric for this problem, and any model evaluated on it is being graded on a
curve it didn't earn.

---

## What actually drives churn

| Segment | Churn rate | vs. 20.4% base |
|---|---|---|
| Holds **4 products** | **100.0%** | 4.9× |
| Holds **3 products** | **82.7%** | 4.1× |
| Age **51–60** | **56.2%** | 2.8× |
| **Germany** | 32.4% | 1.6× |
| **Inactive** member | 26.9% | 1.3× |
| Holds **2 products** | **7.6%** | 0.4× |
| Age 18–30 | 7.5% | 0.4× |

Three findings, in order of how much they'd change what a bank does:

**1. The product-count relationship is not linear, and it is the headline.**
Two products is the sweet spot (7.6% churn). One product churns at 27.7% —
these are single-product customers with nothing anchoring them. But **three and
four products churn at 82.7% and 100%.**

That is not a "more products = more loyal" story. Something is wrong at the top
of the product ladder.

**2. Engagement beats demographics.** Inactive members churn at nearly double
the rate of active ones. This is the most *actionable* driver on the list,
because unlike age or geography, **the bank can do something about it.** You
cannot make a customer younger. You can get them to log in.

**3. Age 51–60 is the risk band, and it is not the oldest band.** Churn peaks at
56.2% in the 51–60 range and then *drops* to 24.8% for 60+. This is a
life-stage effect, not a "seniors leave" effect.

---

## The model

| Model | CV ROC-AUC (5-fold) | Test ROC-AUC | Test PR-AUC |
|---|---|---|---|
| Logistic Regression | 0.761 ± 0.007 | 0.787 | 0.483 |
| **Gradient Boosting** | **0.858 ± 0.007** | **0.872** | **0.724** |

*(PR-AUC baseline = 0.204, the churn rate. The gradient boosting model is 3.5×
better than random at the thing that's actually hard: finding the minority
class.)*

I kept the logistic regression even though it lost, because it explains *why* in
a language a retention manager can act on. Gradient boosting scores the list.
Logistic regression justifies it.

---

## The part most portfolios skip: what threshold, and does it pay?

A model outputs a probability. Someone has to turn that into a phone call. That
someone is choosing a **threshold**, and the default of 0.50 is an arbitrary
convention, not a business decision.

**Stated assumptions** (every one of these is arguable, which is the point —
they should be argued about with the retention team, not buried in a notebook):

- Retention offer costs **$100** per customer contacted
- A retained customer is worth **$1,500**
- The offer actually saves **30%** of genuinely at-risk customers contacted

That gives the economics of each decision:

- Contact a **true churner** → `0.30 × $1,500 − $100 =` **+$350 expected**
- Contact a **non-churner** → **−$100** (money spent on someone who was never leaving)

Which means the cost of a false positive and the cost of a false negative are
**not the same**, so the threshold should not be 0.50.

### Result (2,500-customer test set)

| Strategy | Net value |
|---|---|
| Contact **everyone** | **−$20,950** |
| Contact at default 0.50 threshold | $99,750 |
| **Contact at optimal 0.48 threshold** | **$100,900** |

At the optimal threshold the team contacts **29% of the book**, catches **76%
of all churners**, and accepts a **53% precision rate** — meaning roughly half
of every call is "wasted" on someone who was staying anyway.

**And that is correct.** Because a wasted call costs $100 and a caught churner
returns $350, the model *should* tolerate a lot of false positives. A team
optimizing for precision would feel more efficient and make less money.

The single most important line in this whole project: **blanket-contacting every
customer destroys $20,950 in value.** The model doesn't just find churners — it
turns a money-losing campaign into a money-making one.

---

## What the retention team actually receives

Not a model. A ranked call list.

| Risk decile | Actual churn rate | Lift | % active | Avg age |
|---|---|---|---|---|
| **D10 (highest)** | **85.6%** | **4.2×** | 22.8% | 49.5 |
| D9 | 42.8% | 2.1× | 38.0% | 44.6 |
| D8 | 27.2% | 1.3× | 48.0% | 41.3 |
| … | | | | |
| D1 (lowest) | 1.6% | 0.1× | 67.2% | 33.6 |

**The top decile churns at 85.6% — more than four times the base rate. Call
these people first.** They are, on average, 49 years old, and only 23% of them
are active users.

`retention_call_list_top200.csv` is the actual deliverable: the top 200
customers, scored and sorted, ready to hand to a team.

---

## Recommendations

1. **Investigate the 3–4 product cohort immediately — before building any
   campaign around it.** An 82–100% churn rate at the top of the product ladder
   is not normal customer behavior; it is the signature of a process problem.
   From working in a bank's online channel, the hypotheses I would test first
   are: (a) these products were opened during complaint or problem-resolution
   flows, so the product count is a *symptom* of a customer already in trouble;
   (b) aggressive cross-selling into unsuitable products; or (c) a data
   definition issue where a closing customer's products get counted at exit.
   **If (c) is true, this feature is leakage and must come out of the model.**
   A 60-customer cell at exactly 100% churn is exactly the kind of number that
   should make an analyst suspicious rather than excited.

2. **Target the inactive-member segment, because it's the one lever the bank can
   actually pull.** Age and geography are not addressable. Engagement is. A
   digital re-engagement campaign aimed at inactive members in the top three
   risk deciles is the highest-ROI intervention available here.

3. **Set the threshold at 0.48, not 0.50 — and revisit it whenever the offer
   cost or CLV assumption changes.** The threshold is not a modeling parameter.
   It is a business decision that happens to live in the code.

4. **Contact the top 3 deciles, not everyone.** 30% of the book, most of the
   churners, positive ROI.

---

## Honest limitations

- **Point-in-time snapshot, no event timestamps.** I cannot confirm that every
  feature was known *before* the customer churned. The product-count anomaly is
  the leading leakage suspect and I have flagged it rather than shipped it.
- **The $100 / $1,500 / 30% assumptions are illustrative.** In a real
  engagement these come from finance and from a holdout test of the offer, not
  from me. The framework survives changing them; the specific threshold does not.
- **No holdout campaign.** Predicted churn is not the same as *preventable*
  churn. Some of the highest-risk customers are leaving no matter who calls
  them. The only way to know is an A/B test: contact a random half of the top
  deciles, leave the other half alone, and measure the difference. **That is
  what I would build next.**

---

## Files

```
churn_analysis.py                  full pipeline — EDA, models, economics
bank_churn.csv                     10,000-customer public dataset
churn_analysis.png                 ROC, net-value curve, drivers, deciles
retention_call_list_top200.csv     the deliverable: ranked call list
threshold_economics.csv            net value at every threshold
risk_deciles.csv                   decile summary
full_output.txt                    complete console output
```

Run with: `python churn_analysis.py`
