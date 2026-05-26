# ================================================================
# E-Commerce Customer Behavior & Churn Analysis
# Dataset : Olist Brazilian E-Commerce (Kaggle)
# Author  : Achal Tidke
# Tools   : Python 3.10 | Pandas | NumPy | Scikit-learn | Matplotlib | Seaborn
# VS Code : Run section by section using # %% cells
# ================================================================

# %%
# ── IMPORTS ──────────────────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve)
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['figure.dpi'] = 130

print("All libraries imported successfully ")

# ── CHANGE THIS PATH TO WHERE YOU SAVED THE KAGGLE FILES ─────────
BASE = "data/"          # e.g. "C:/Users/Achal/Downloads/olist/" 
OUT  = "outputs/"       # charts will save here

import os
os.makedirs(OUT, exist_ok=True)

print("✅ Imports done")

# %%
# ── SECTION 1: LOAD DATA ─────────────────────────────────────────
customers   = pd.read_csv(BASE + "olist_customers_dataset.csv")
orders      = pd.read_csv(BASE + "olist_orders_dataset.csv",
                          parse_dates=["order_purchase_timestamp",
                                       "order_approved_at",
                                       "order_delivered_customer_date",
                                       "order_estimated_delivery_date"])
order_items = pd.read_csv(BASE + "olist_order_items_dataset.csv")
products    = pd.read_csv(BASE + "olist_products_dataset.csv")
sellers     = pd.read_csv(BASE + "olist_sellers_dataset.csv")
reviews     = pd.read_csv(BASE + "olist_order_reviews_dataset.csv")
payments    = pd.read_csv(BASE + "olist_order_payments_dataset.csv")

print("Dataset shapes:")
for name, df in [("customers", customers), ("orders", orders),
                 ("order_items", order_items), ("products", products),
                 ("reviews", reviews), ("payments", payments)]:
    print(f"  {name:<15} {len(df):>8,} rows  |  {df.shape[1]} cols")

# %%
# ── SECTION 2: DATA QUALITY CHECK ────────────────────────────────
print("\n── NULL CHECK ──")
print(orders.isnull().sum())

print(f"\nOrder statuses:\n{orders['order_status'].value_counts()}")
print(f"\nDate range: {orders['order_purchase_timestamp'].min().date()} "
      f"→ {orders['order_purchase_timestamp'].max().date()}")

# %%
# ── SECTION 3: BUILD MASTER TABLE ────────────────────────────────
# Join all 6 tables into one analysis-ready dataframe
master = (orders
    .merge(customers,                             on="customer_id",   how="left")
    .merge(order_items,                           on="order_id",      how="left")
    .merge(products,                              on="product_id",    how="left")
    .merge(payments,                              on="order_id",      how="left")
    .merge(reviews[["order_id","review_score"]], on="order_id",      how="left"))

master["order_month"]   = master["order_purchase_timestamp"].dt.to_period("M")
master["delivery_days"] = (master["order_delivered_customer_date"] -
                            master["order_purchase_timestamp"]).dt.days

delivered = master[master["order_status"] == "delivered"].copy()

print(f"Master table rows : {len(master):,}")
print(f"Delivered orders  : {len(delivered):,}")

# %%
# ── SECTION 4: EDA — MONTHLY REVENUE ─────────────────────────────
monthly = (delivered.groupby("order_month")["payment_value"]
           .sum().reset_index())
monthly["order_month"] = monthly["order_month"].astype(str)

fig, ax = plt.subplots(figsize=(13, 4))
ax.fill_between(range(len(monthly)), monthly["payment_value"] / 1000,
                alpha=0.12, color="#1a56db")
ax.plot(range(len(monthly)), monthly["payment_value"] / 1000,
        color="#1a56db", linewidth=2.5, marker="o", markersize=5)
ax.set_xticks(range(len(monthly)))
ax.set_xticklabels(monthly["order_month"], rotation=45, ha="right", fontsize=8)
ax.set_title("Monthly Revenue Trend (BRL '000)", fontsize=14, fontweight="bold", pad=12)
ax.set_ylabel("Revenue (BRL '000)")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"R${x:.0f}k"))
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(OUT + "01_monthly_revenue.png", bbox_inches="tight")
plt.show()
print("✅ Chart 1 saved")

# %%
# ── SECTION 5: EDA — TOP CATEGORIES ──────────────────────────────
cat_rev = (delivered.groupby("product_category_name")["payment_value"]
           .sum().sort_values(ascending=False).head(10).reset_index())
cat_rev.columns = ["category", "revenue"]

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.barh(cat_rev["category"][::-1], cat_rev["revenue"][::-1] / 1000,
               color="#1a56db", edgecolor="white")
for bar, val in zip(bars, cat_rev["revenue"][::-1] / 1000):
    ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
            f"R${val:.0f}k", va="center", fontsize=8)
ax.set_title("Top 10 Product Categories by Revenue", fontsize=14, fontweight="bold")
ax.set_xlabel("Revenue (BRL '000)")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(OUT + "02_top_categories.png", bbox_inches="tight")
plt.show()
print("✅ Chart 2 saved")

# %%
# ── SECTION 6: EDA — REVIEW SCORE DISTRIBUTION ───────────────────
score_counts = delivered["review_score"].value_counts().sort_index()
colors = ["#e02424", "#f97316", "#eab308", "#84cc16", "#057a55"]

fig, ax = plt.subplots(figsize=(7, 4))
bars = ax.bar(score_counts.index, score_counts.values,
              color=colors, edgecolor="white", width=0.6)
for bar, val in zip(bars, score_counts.values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 100,
            f"{val:,}", ha="center", fontsize=9, fontweight="bold")
ax.set_title("Customer Review Score Distribution", fontsize=14, fontweight="bold")
ax.set_xlabel("Review Score  (1 = Worst, 5 = Best)")
ax.set_ylabel("Number of Orders")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(OUT + "03_review_scores.png", bbox_inches="tight")
plt.show()
print("✅ Chart 3 saved")

# %%
# ── SECTION 7: RFM SEGMENTATION ──────────────────────────────────
snapshot = delivered["order_purchase_timestamp"].max() + pd.Timedelta(days=1)

rfm = (delivered.groupby("customer_unique_id").agg(
    last_purchase = ("order_purchase_timestamp", "max"),
    frequency     = ("order_id",                "nunique"),
    monetary      = ("payment_value",            "sum")
).reset_index())

rfm["recency"] = (snapshot - rfm["last_purchase"]).dt.days

# Quartile scoring  (4 = best)
rfm["R"] = pd.qcut(rfm["recency"],                        4, labels=[4,3,2,1]).astype(int)
rfm["F"] = pd.qcut(rfm["frequency"].rank(method="first"), 4, labels=[1,2,3,4]).astype(int)
rfm["M"] = pd.qcut(rfm["monetary"].rank(method="first"),  4, labels=[1,2,3,4]).astype(int)

def segment(row):
    r, f, m = row["R"], row["F"], row["M"]
    if   r >= 4 and f >= 3 and m >= 3: return "Champions"
    elif r >= 3 and f >= 3:            return "Loyal Customers"
    elif r >= 4 and f <= 1:            return "New Customers"
    elif r >= 3 and f <= 2:            return "Potential Loyalists"
    elif r == 2 and f >= 2:            return "At Risk"
    elif r <= 2 and f <= 2:            return "Churned"
    else:                              return "Needs Attention"

rfm["Segment"] = rfm.apply(segment, axis=1)

seg_summary = (rfm.groupby("Segment")
               .agg(Customers=("customer_unique_id","count"),
                    Avg_Recency=("recency","mean"),
                    Avg_Orders=("frequency","mean"),
                    Avg_Revenue=("monetary","mean"))
               .round(1).reset_index()
               .sort_values("Customers", ascending=False))

print("\n── RFM SEGMENTS ──")
print(seg_summary.to_string(index=False))

# Plot
palette = {"Champions":"#057a55","Loyal Customers":"#0ea5e9",
           "Potential Loyalists":"#8b5cf6","New Customers":"#06b6d4",
           "At Risk":"#f97316","Needs Attention":"#eab308","Churned":"#e02424"}
seg_plot = seg_summary.set_index("Segment")["Customers"]
bar_colors = [palette.get(s, "#1a56db") for s in seg_plot.index]

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(seg_plot.index, seg_plot.values, color=bar_colors, edgecolor="white")
for bar, val in zip(bars, seg_plot.values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 20,
            f"{val:,}", ha="center", fontsize=9, fontweight="bold")
ax.set_title("RFM Customer Segments", fontsize=14, fontweight="bold")
ax.set_ylabel("Number of Customers")
ax.set_xticklabels(seg_plot.index, rotation=25, ha="right")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(OUT + "04_rfm_segments.png", bbox_inches="tight")
plt.show()
print("✅ Chart 4 saved")

# %%
# ── SECTION 8: COHORT RETENTION ──────────────────────────────────
cohort_df = delivered[["customer_unique_id","order_purchase_timestamp"]].copy()
cohort_df["order_month"]  = cohort_df["order_purchase_timestamp"].dt.to_period("M")
cohort_df["cohort_month"] = (cohort_df.groupby("customer_unique_id")["order_purchase_timestamp"]
                             .transform("min").dt.to_period("M"))
cohort_df["period"] = (cohort_df["order_month"] - cohort_df["cohort_month"]).apply(lambda x: x.n)

cohort_pivot = (cohort_df.groupby(["cohort_month","period"])["customer_unique_id"]
                .nunique().reset_index()
                .pivot(index="cohort_month", columns="period", values="customer_unique_id"))

retention = cohort_pivot.divide(cohort_pivot[0], axis=0).round(3).iloc[:, :7]

fig, ax = plt.subplots(figsize=(11, 7))
sns.heatmap(retention, annot=True, fmt=".0%", cmap="YlOrRd_r",
            mask=retention.isnull(), ax=ax, linewidths=0.5,
            linecolor="white", vmin=0, vmax=1,
            annot_kws={"size": 8})
ax.set_title("Monthly Cohort Retention Rate", fontsize=14, fontweight="bold")
ax.set_xlabel("Months After First Purchase")
ax.set_ylabel("Cohort (First Purchase Month)")
ax.set_xticklabels([f"M+{i}" for i in range(7)], fontsize=9)
plt.tight_layout()
plt.savefig(OUT + "05_cohort_retention.png", bbox_inches="tight")
plt.show()
print("✅ Chart 5 saved")

# %%
# ── SECTION 9: CHURN PREDICTION MODEL ────────────────────────────
# Define churn: no purchase in last 90 days of dataset window
churn_cutoff = snapshot - pd.Timedelta(days=90)
rfm["churned"] = (rfm["last_purchase"] < churn_cutoff).astype(int)

print(f"Churn rate : {rfm['churned'].mean():.1%}")
print(f"Churned    : {rfm['churned'].sum():,}")
print(f"Active     : {(rfm['churned']==0).sum():,}")

features = rfm[["recency","frequency","monetary","R","F","M"]]
target   = rfm["churned"]

X_train, X_test, y_train, y_test = train_test_split(
    features, target, test_size=0.2, random_state=42, stratify=target)

# Scale for Logistic Regression
scaler  = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# ── Logistic Regression
lr = LogisticRegression(max_iter=500, random_state=42)
lr.fit(X_train_sc, y_train)
lr_pred = lr.predict(X_test_sc)
lr_prob = lr.predict_proba(X_test_sc)[:, 1]

# ── Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_prob = rf.predict_proba(X_test)[:, 1]

lr_auc = roc_auc_score(y_test, lr_prob)
rf_auc = roc_auc_score(y_test, rf_prob)

print("\n── LOGISTIC REGRESSION ──")
print(classification_report(y_test, lr_pred, target_names=["Active","Churned"]))
print("── RANDOM FOREST ──")
print(classification_report(y_test, rf_pred, target_names=["Active","Churned"]))
print(f"ROC-AUC → LR: {lr_auc:.4f}  |  RF: {rf_auc:.4f}")

# %%
# ── SECTION 10: ROC CURVE ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 6))
for prob, auc, label, color in [
        (lr_prob, lr_auc, "Logistic Regression", "#1a56db"),
        (rf_prob, rf_auc, "Random Forest",       "#057a55")]:
    fpr, tpr, _ = roc_curve(y_test, prob)
    ax.plot(fpr, tpr, label=f"{label}  (AUC = {auc:.3f})",
            linewidth=2.2, color=color)
ax.plot([0,1],[0,1], "--", color="gray", linewidth=1, label="Random Baseline")
ax.set_title("ROC Curve — Churn Prediction", fontsize=14, fontweight="bold")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.legend(loc="lower right", fontsize=10)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(OUT + "06_roc_curve.png", bbox_inches="tight")
plt.show()
print("✅ Chart 6 saved")

# %%
# ── SECTION 11: FEATURE IMPORTANCE ───────────────────────────────
fi = (pd.DataFrame({"feature": features.columns,
                    "importance": rf.feature_importances_})
      .sort_values("importance", ascending=True))

fig, ax = plt.subplots(figsize=(8, 4))
bars = ax.barh(fi["feature"], fi["importance"],
               color="#1a56db", edgecolor="white")
for bar, val in zip(bars, fi["importance"]):
    ax.text(bar.get_width() + 0.003, bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}", va="center", fontsize=9)
ax.set_title("Feature Importance — Random Forest Churn Model",
             fontsize=13, fontweight="bold")
ax.set_xlabel("Importance Score")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(OUT + "07_feature_importance.png", bbox_inches="tight")
plt.show()
print("✅ Chart 7 saved")

# %%
# ── SECTION 12: CONFUSION MATRIX ─────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax, pred, title in [(axes[0], lr_pred, "Logistic Regression"),
                         (axes[1], rf_pred, "Random Forest")]:
    cm = confusion_matrix(y_test, pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Active","Churned"],
                yticklabels=["Active","Churned"],
                linewidths=0.5, cbar=False,
                annot_kws={"size": 13, "fontweight":"bold"})
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
plt.suptitle("Confusion Matrix — Churn Prediction",
             fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(OUT + "08_confusion_matrix.png", bbox_inches="tight")
plt.show()
print("✅ Chart 8 saved")

# %%
# ── SECTION 13: DELIVERY DAYS vs REVIEW SCORE ────────────────────
delivery_review = delivered[delivered["delivery_days"].between(0, 60)].copy()

fig, ax = plt.subplots(figsize=(9, 5))
colors_map = {1:"#e02424", 2:"#f97316", 3:"#eab308", 4:"#84cc16", 5:"#057a55"}
for score in [1,2,3,4,5]:
    sub = delivery_review[delivery_review["review_score"] == score]["delivery_days"]
    ax.hist(sub, bins=20, alpha=0.55, label=f"Score {score}",
            color=colors_map[score], density=True)
ax.set_title("Delivery Time vs Review Score", fontsize=14, fontweight="bold")
ax.set_xlabel("Delivery Days")
ax.set_ylabel("Density")
ax.legend(title="Review Score", fontsize=9)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(OUT + "09_delivery_vs_review.png", bbox_inches="tight")
plt.show()
print("✅ Chart 9 saved")

# %%
# ── SECTION 14: FINAL SUMMARY ────────────────────────────────────
print("\n" + "="*55)
print("BUSINESS INSIGHTS SUMMARY")
print("="*55)
print(f"  Total Revenue (delivered) : R${delivered['payment_value'].sum():>12,.0f}")
print(f"  Avg Order Value           : R${delivered.groupby('order_id')['payment_value'].first().mean():>10.2f}")
print(f"  Avg Delivery Days         : {delivery_review['delivery_days'].mean():>10.1f} days")
print(f"  Avg Review Score          : {delivered['review_score'].mean():>10.2f} / 5.00")
print(f"  Overall Churn Rate (90d)  : {rfm['churned'].mean():>10.1%}")
print(f"  Champions Customers       : {(rfm['Segment']=='Champions').sum():>10,}")
print(f"  Random Forest AUC         : {rf_auc:>10.4f}")
print("\n✅ All 9 charts saved to outputs/")
