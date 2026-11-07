"""
ApexLark AI & ML Internship - Day 4
Feature Engineering & Data Transformation  |  Telco Customer Churn

Dataset : Telco Customer Churn (IBM sample data, Kaggle: blastchar/telco-customer-churn)
          7,043 rows x 21 columns  (file: WA_Fn-UseC_-Telco-Customer-Churn.csv)
Usage   : python day4_feature_engineering.py [path_to_csv]
Outputs : figures/*.png, Day4_Feature_Engineering_Report.pdf, churn_transformed.csv
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # works without a display; in Jupyter just remove this line
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
sns.set_theme(style="whitegrid")
os.makedirs("figures", exist_ok=True)

CSV = sys.argv[1] if len(sys.argv) > 1 else "WA_Fn-UseC_-Telco-Customer-Churn.csv"
LOG = {}   # numbers collected here are re-used by the PDF report


def section(title):
    print("\n" + "=" * 80 + f"\n{title}\n" + "=" * 80)


# --------------------------------------------------------------------------- #
# 1. DATASET COLLECTION
# --------------------------------------------------------------------------- #
section("1. DATASET COLLECTION")
df = pd.read_csv(CSV)
print(f"Loaded '{CSV}' -> {df.shape[0]} records, {df.shape[1]} columns")
assert df.shape[0] >= 1000 and df.shape[1] >= 8, "Dataset must have >=1000 rows and >=8 columns"
LOG["rows"], LOG["cols"] = df.shape

# --------------------------------------------------------------------------- #
# 2. DATASET ANALYSIS
# --------------------------------------------------------------------------- #
section("2. DATASET ANALYSIS")
print("First 10 records:\n", df.head(10))
print("\nDataset info:")
df.info()

# TotalCharges is stored as text because new customers (tenure = 0) have a blank value.
# Convert to numeric first so that it is analysed as a numerical feature.
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

print("\nSummary statistics:\n", df.describe())
num_cols = df.select_dtypes(include="number").columns.tolist()
cat_cols = df.select_dtypes(exclude="number").columns.tolist()
print("\nNumerical features :", num_cols)
print("Categorical features:", cat_cols)
missing = df.isnull().sum()
print("\nMissing values:\n", missing[missing > 0] if missing.any() else "None")
LOG["missing_total"] = int(df["TotalCharges"].isnull().sum())
LOG["churn_rate"] = (df["Churn"] == "Yes").mean() * 100

# Handle missing TotalCharges: these customers have tenure 0 -> they have not been billed yet.
df["TotalCharges"] = df["TotalCharges"].fillna(0)

# --------------------------------------------------------------------------- #
# 3. FEATURE ENGINEERING
# --------------------------------------------------------------------------- #
section("3. FEATURE ENGINEERING")

# 3a. Tidy inconsistent categories: 'No internet service' / 'No phone service' just mean 'No'.
#     Keeping them separate creates redundant dummy columns.
for c in ["MultipleLines", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
          "TechSupport", "StreamingTV", "StreamingMovies"]:
    df[c] = df[c].replace({"No internet service": "No", "No phone service": "No"})

# 3b. NEW FEATURES (6 created; 3 required)
service_cols = ["PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
                "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
# (1) NumServices  - how many add-on services the customer subscribes to (stickiness)
df["NumServices"] = (df[service_cols] == "Yes").sum(axis=1)
# (2) AvgMonthlyCharge - real average spend per month of tenure (captures price changes)
df["AvgMonthlyCharge"] = np.where(df["tenure"] > 0, df["TotalCharges"] / df["tenure"], df["MonthlyCharges"])
# (3) ChargeGap - current price minus historical average (positive = recent price increase)
df["ChargeGap"] = df["MonthlyCharges"] - df["AvgMonthlyCharge"]
# (4) HasSecuritySupport - protected customers churn less (boolean flag)
df["HasSecuritySupport"] = (df["OnlineSecurity"].eq("Yes") | df["TechSupport"].eq("Yes"))
# (5) IsLongContract - boolean flag for 1-year / 2-year contracts
df["IsLongContract"] = df["Contract"].ne("Month-to-month")
# (6) AutoPay - boolean flag for automatic payment methods
df["AutoPay"] = df["PaymentMethod"].str.contains("automatic")
new_features = ["NumServices", "AvgMonthlyCharge", "ChargeGap", "HasSecuritySupport", "IsLongContract", "AutoPay"]
print("New features:\n", df[new_features].head())

# 3c. BINNING a numerical column -> tenure groups
df["TenureGroup"] = pd.cut(df["tenure"], bins=[-1, 12, 24, 48, 72],
                           labels=["0-1yr", "1-2yr", "2-4yr", "4-6yr"])
print("\nTenureGroup counts:\n", df["TenureGroup"].value_counts().sort_index())

# 3d. DATE COLUMNS - Telco has none; generic extraction kept so the script works on other data
date_cols = [c for c in df.columns if "date" in c.lower()]
for c in date_cols:
    d = pd.to_datetime(df[c], errors="coerce")
    df[c + "_year"], df[c + "_month"], df[c + "_dow"] = d.dt.year, d.dt.month, d.dt.dayofweek
print("\nDate columns found:", date_cols if date_cols else "none (dataset has no date column)")

# 3e. REMOVE irrelevant / redundant features
#   customerID : unique identifier, no predictive information
#   (redundant columns are removed after the correlation analysis in step 5)
df = df.drop(columns=["customerID"])
print("\nDropped irrelevant column: customerID")

# 3f. ENCODING
df["Churn"] = LabelEncoder().fit_transform(df["Churn"])          # No=0, Yes=1 (target)
text_cols = df.select_dtypes(exclude=["number", "bool"]).columns   # object / string / category columns
binary_cols = [c for c in text_cols if df[c].nunique() == 2 and c != "TenureGroup"]
le = LabelEncoder()
for c in binary_cols:                                            # Label encoding for 2-class columns
    df[c] = le.fit_transform(df[c])
multi_cols = [c for c in text_cols if c not in binary_cols]
print("Label-encoded (binary):", binary_cols)
print("One-hot encoded (multi-class):", multi_cols)
df = pd.get_dummies(df, columns=multi_cols, drop_first=True)     # One-hot encoding

# --------------------------------------------------------------------------- #
# 4. DATA TRANSFORMATION
# --------------------------------------------------------------------------- #
section("4. DATA TRANSFORMATION")

# 4a. Boolean -> numerical (get_dummies and the flags above are bool dtype)
bool_cols = df.select_dtypes(include="bool").columns
df[bool_cols] = df[bool_cols].astype(int)
print(f"Converted {len(bool_cols)} boolean columns to int (0/1)")

# 4b. Log transformation for skewed data
skew_before = df[["tenure", "MonthlyCharges", "TotalCharges", "AvgMonthlyCharge"]].skew()
print("\nSkewness before transformation:\n", skew_before.round(3))
df["TotalCharges_log"] = np.log1p(df["TotalCharges"])            # log1p handles the 0 values safely
LOG["skew_total_before"] = skew_before["TotalCharges"]
LOG["skew_total_after"] = df["TotalCharges_log"].skew()
print(f"TotalCharges skew {LOG['skew_total_before']:.2f} -> {LOG['skew_total_after']:.2f} after log1p")
hist_before = df["TotalCharges"].copy()

# 4c. Normalisation (Min-Max -> range [0, 1]) of one numerical feature
df["MonthlyCharges_norm"] = MinMaxScaler().fit_transform(df[["MonthlyCharges"]])
print(f"MonthlyCharges_norm range: {df['MonthlyCharges_norm'].min():.2f} - {df['MonthlyCharges_norm'].max():.2f}")

# 4d. Standard scaling (mean 0, std 1)
to_scale = ["tenure", "MonthlyCharges", "TotalCharges_log", "AvgMonthlyCharge", "ChargeGap", "NumServices"]
LOG["std_before"] = df[to_scale].std().round(2).to_dict()
scaled = pd.DataFrame(StandardScaler().fit_transform(df[to_scale]),
                      columns=[c + "_scaled" for c in to_scale], index=df.index)
df = pd.concat([df, scaled], axis=1)
print("\nAfter StandardScaler (mean ~0, std ~1):\n", scaled.agg(["mean", "std"]).round(3))

# 4e. Verify transformed data set
assert df.isnull().sum().sum() == 0, "Missing values remain"
assert all(pd.api.types.is_numeric_dtype(t) for t in df.dtypes), "Non-numeric columns remain"
print(f"\nVerification OK -> shape {df.shape}, no NaN, all columns numeric")
print(df.dtypes.value_counts())

# --------------------------------------------------------------------------- #
# 5. FEATURE SELECTION
# --------------------------------------------------------------------------- #
section("5. FEATURE SELECTION")
# Use the un-scaled engineered features for correlation (scaling does not change correlation)
raw_features = [c for c in df.columns if not c.endswith("_scaled") and c not in ("MonthlyCharges_norm",)]
corr = df[raw_features].corr()
target_corr = corr["Churn"].drop("Churn")

# Highly correlated pairs (|r| > 0.8)
pairs = [(a, b, corr.loc[a, b]) for i, a in enumerate(corr.columns) for b in corr.columns[i + 1:]
         if abs(corr.loc[a, b]) > 0.8 and "Churn" not in (a, b)]
print("Highly correlated pairs (|r|>0.8):")
for a, b, r in sorted(pairs, key=lambda x: -abs(x[2])):
    print(f"  {a:35s} {b:35s} r={r:.2f}")
LOG["pairs"] = [(a, b, round(r, 2)) for a, b, r in sorted(pairs, key=lambda x: -abs(x[2]))][:6]

# Drop the member of each pair that is less correlated with the target (redundant feature removal)
to_drop = set()
for a, b, _ in pairs:
    if a in to_drop or b in to_drop:
        continue
    to_drop.add(a if abs(target_corr[a]) < abs(target_corr[b]) else b)
print("\nRedundant features removed:", sorted(to_drop))
LOG["dropped"] = sorted(to_drop)

candidates = target_corr.drop(labels=list(to_drop), errors="ignore")
selected = candidates[candidates.abs() >= 0.10].abs().sort_values(ascending=False).index.tolist()
print(f"\nSelected {len(selected)} features (|corr with Churn| >= 0.10):")
for f in selected:
    print(f"  {f:40s} r = {target_corr[f]:+.3f}")
LOG["selected"] = [(f, round(target_corr[f], 3)) for f in selected]

# Why they matter - plain-English explanation
print("\nWhy these features matter:")
for f in selected[:8]:
    direction = "raise" if target_corr[f] > 0 else "lower"
    print(f"  - {f}: higher values {direction} the churn probability (r={target_corr[f]:+.2f}).")

# --------------------------------------------------------------------------- #
# 6. DATA VISUALIZATION
# --------------------------------------------------------------------------- #
section("6. DATA VISUALIZATION")
churn_label = df["Churn"].map({0: "No", 1: "Yes"})

# 6a. Correlation heatmap (selected features + target)
top = ["Churn"] + selected[:14]
plt.figure(figsize=(11, 9))
sns.heatmap(df[top].corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0, linewidths=.4)
plt.title("Correlation Heatmap - Churn & Top Selected Features")
plt.tight_layout(); plt.savefig("figures/01_correlation_heatmap.png", dpi=150); plt.close()

# 6b. Histograms of transformed features (+ before/after log)
fig, ax = plt.subplots(2, 3, figsize=(15, 8))
ax = ax.ravel()
ax[0].hist(hist_before, bins=40, color="#c0392b"); ax[0].set_title(f"TotalCharges (raw) skew={LOG['skew_total_before']:.2f}")
ax[1].hist(df["TotalCharges_log"], bins=40, color="#27ae60"); ax[1].set_title(f"TotalCharges (log1p) skew={LOG['skew_total_after']:.2f}")
ax[2].hist(df["MonthlyCharges_norm"], bins=40, color="#2980b9"); ax[2].set_title("MonthlyCharges (Min-Max normalised)")
ax[3].hist(df["tenure_scaled"], bins=40, color="#8e44ad"); ax[3].set_title("tenure (standard scaled)")
ax[4].hist(df["AvgMonthlyCharge_scaled"], bins=40, color="#d35400"); ax[4].set_title("AvgMonthlyCharge (standard scaled)")
ax[5].hist(df["NumServices_scaled"], bins=20, color="#16a085"); ax[5].set_title("NumServices (standard scaled)")
for a in ax:
    a.set_ylabel("Frequency")
plt.suptitle("Histograms of Transformed Features", fontsize=15)
plt.tight_layout(); plt.savefig("figures/02_histograms_transformed.png", dpi=150); plt.close()

# 6c. Box plot
plt.figure(figsize=(8, 6))
sns.boxplot(x=churn_label, y=df["MonthlyCharges"], hue=churn_label, palette="Set2", legend=False)
plt.title("Monthly Charges by Churn Status"); plt.xlabel("Churn"); plt.ylabel("Monthly Charges ($)")
plt.tight_layout(); plt.savefig("figures/03_boxplot_monthly_charges.png", dpi=150); plt.close()

# 6d. Count plot (churn by tenure group)
tg_cols = [c for c in df.columns if c.startswith("TenureGroup_")]
tenure_group = df[tg_cols].idxmax(axis=1).str.replace("TenureGroup_", "")
tenure_group[df[tg_cols].sum(axis=1) == 0] = "0-1yr"     # baseline category dropped by drop_first
order = ["0-1yr", "1-2yr", "2-4yr", "4-6yr"]
plt.figure(figsize=(8, 6))
sns.countplot(x=tenure_group, hue=churn_label, order=order, palette="Set1")
plt.title("Customer Count by Tenure Group and Churn"); plt.xlabel("Tenure Group"); plt.ylabel("Customers")
plt.legend(title="Churn"); plt.tight_layout(); plt.savefig("figures/04_countplot_tenure_group.png", dpi=150); plt.close()
LOG["churn_by_group"] = {g: round(df.loc[tenure_group == g, "Churn"].mean() * 100, 1) for g in order}

# 6e. Scatter plot of two important features
plt.figure(figsize=(9, 6))
sns.scatterplot(x=df["tenure"], y=df["MonthlyCharges"], hue=churn_label, alpha=.5, palette={"No": "#2980b9", "Yes": "#c0392b"})
plt.title("Tenure vs Monthly Charges (coloured by Churn)"); plt.xlabel("Tenure (months)"); plt.ylabel("Monthly Charges ($)")
plt.grid(True, alpha=.3); plt.tight_layout(); plt.savefig("figures/05_scatter_tenure_monthly.png", dpi=150); plt.close()
print("Saved 5 figures to ./figures")

# Save the final dataset
df.to_csv("churn_transformed.csv", index=False)
print("Saved churn_transformed.csv", df.shape)

# --------------------------------------------------------------------------- #
# 7. BUSINESS INSIGHTS REPORT (PDF) - numbers below are computed from the data
# --------------------------------------------------------------------------- #
section("7. BUSINESS INSIGHTS REPORT")
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak

mtm = df.loc[df["IsLongContract"] == 0, "Churn"].mean() * 100
lc = df.loc[df["IsLongContract"] == 1, "Churn"].mean() * 100
fib_col = "InternetService_Fiber optic"
fib = df.loc[df[fib_col] == 1, "Churn"].mean() * 100 if fib_col in df else float("nan")
ns_hi = df.loc[df["NumServices"] >= 5, "Churn"].mean() * 100
ns_lo = df.loc[df["NumServices"] <= 2, "Churn"].mean() * 100
sec = df.loc[df["HasSecuritySupport"] == 1, "Churn"].mean() * 100
nosec = df.loc[df["HasSecuritySupport"] == 0, "Churn"].mean() * 100
ap = df.loc[df["AutoPay"] == 1, "Churn"].mean() * 100
nap = df.loc[df["AutoPay"] == 0, "Churn"].mean() * 100
top_f, top_r = LOG["selected"][0]
g = LOG["churn_by_group"]

obs = [
    f"Overall churn rate is {LOG['churn_rate']:.1f}% across {LOG['rows']:,} customers, so the target is imbalanced (stratified splits / class weights are needed).",
    f"Month-to-month customers churn at {mtm:.1f}% versus {lc:.1f}% for 1-2 year contracts.",
    f"Churn falls with tenure: {g['0-1yr']}% in the first year versus {g['4-6yr']}% after four years.",
    f"Customers with security or tech-support add-ons churn at {sec:.1f}% versus {nosec:.1f}% without them; automatic-payment customers churn at {ap:.1f}% versus {nap:.1f}%.",
    f"The strongest single correlate of churn is {top_f} (r = {top_r:+.2f}); fibre-optic subscribers churn at {fib:.1f}%.",
]
feat_rows = [["Feature", "Formula", "Purpose"],
             ["NumServices", "count of 'Yes' across 8 service columns", "Measures how embedded the customer is; more services = higher switching cost."],
             ["AvgMonthlyCharge", "TotalCharges / tenure", "True average spend; reveals price changes hidden by MonthlyCharges."],
             ["ChargeGap", "MonthlyCharges - AvgMonthlyCharge", "Flags customers whose bill recently rose (price-shock churn)."],
             ["HasSecuritySupport / IsLongContract / AutoPay", "boolean flags", "Simple, strong binary signals of commitment and support."]]
trans_rows = [["Transformation", "Impact"],
              ["Log (log1p) on TotalCharges", f"Skewness moved from {LOG['skew_total_before']:.2f} to {LOG['skew_total_after']:.2f} (closer to a symmetric shape); compresses the long right tail, which helps linear / distance-based models."],
              ["Standard scaling (6 features)", "All features now have mean 0 and std 1 (std before: " + ", ".join(f"{k} {v}" for k, v in list(LOG['std_before'].items())[:3]) + " ...); prevents large-valued columns dominating."],
              ["Min-Max normalisation of MonthlyCharges", "Rescaled to [0, 1] for algorithms that need bounded inputs (e.g. neural networks, KNN)."],
              ["Encoding + boolean -> int", f"Label encoding for binary columns, one-hot for the rest; dataset grew from {LOG['cols']} to {df.shape[1]} fully numeric columns."]]
recs = [
    "Add interaction features (e.g. Contract x InternetService, tenure x MonthlyCharges) to capture combined effects.",
    "Use target / frequency encoding or grouped categories if high-cardinality columns are added, to avoid dimensionality explosion.",
    "Apply recursive feature elimination or model-based importance (Random Forest / XGBoost) in addition to correlation to catch non-linear signals.",
    "Handle class imbalance (SMOTE, class weights) and fit all scalers inside a Pipeline on the training split only, to prevent data leakage.",
    "Collect time-based features (contract start date, last complaint date, recent usage trend) - recency features are usually strong churn predictors.",
]

ss = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=ss["Heading1"], textColor=colors.HexColor("#1f3a5f"))
H2 = ParagraphStyle("H2", parent=ss["Heading2"], textColor=colors.HexColor("#2a6ebb"))
B = ParagraphStyle("B", parent=ss["BodyText"], fontSize=10, leading=14)
C = ParagraphStyle("C", parent=B, fontSize=8.5, leading=11)


def tbl(rows, widths):
    t = Table([[Paragraph(str(c), C) for c in r] for r in rows], colWidths=widths, repeatRows=1)
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#d9e6f5")),
                           ("GRID", (0, 0), (-1, -1), .4, colors.grey), ("VALIGN", (0, 0), (-1, -1), "TOP")]))
    return t


def fig(path, caption, w=15):
    img = Image(path); r = img.imageHeight / img.imageWidth
    img.drawWidth, img.drawHeight = w * cm, w * cm * r
    return [img, Paragraph(f"<i>{caption}</i>", C), Spacer(1, 8)]


S = [Paragraph("Feature Engineering Report", H1),
     Paragraph("ApexLark AI &amp; Machine Learning Internship - Day 4 | Telco Customer Churn", B), Spacer(1, 10),
     Paragraph("1. Dataset &amp; Approach", H2),
     Paragraph(f"The Telco Customer Churn dataset has {LOG['rows']:,} records and {LOG['cols']} columns (numerical: tenure, MonthlyCharges, "
               f"TotalCharges, SeniorCitizen; the rest categorical). {LOG['missing_total']} blank TotalCharges values (customers with tenure 0) were "
               "converted to numeric and set to 0. customerID was dropped as irrelevant. Six new features and one binned feature were created, "
               "categories encoded, skewed data log-transformed, values scaled, and features selected by correlation.", B),
     Paragraph("2. Five Observations", H2)]
S += [Paragraph(f"{i}. {o}", B) for i, o in enumerate(obs, 1)]
S += [Paragraph("3. Three Newly Created Features and Their Purpose", H2), tbl(feat_rows, [4.5 * cm, 5 * cm, 7.5 * cm]),
      Paragraph("4. Three Transformations and Their Impact", H2), tbl(trans_rows, [5 * cm, 12 * cm]),
      Paragraph("5. Feature Selection", H2),
      Paragraph("Highly correlated pairs (|r| &gt; 0.8): " + ("; ".join(f"{a} / {b} (r={r})" for a, b, r in LOG["pairs"]) or "none") +
                ". Redundant columns removed (kept the one more related to churn): " + (", ".join(LOG["dropped"]) or "none") + ".", B),
      Paragraph("Selected features (|r with Churn| &ge; 0.10): " + ", ".join(f"{f} ({r:+.2f})" for f, r in LOG["selected"][:12]) +
                ". They are important because each shows a clear, measurable link with churn - contract commitment, tenure, price level, "
                "service bundle and payment behaviour are all direct levers the business can act on.", B),
      PageBreak(), Paragraph("6. Visualisations", H2)]
S += fig("figures/01_correlation_heatmap.png", "Figure 1: Correlation heatmap", 13)
S += fig("figures/02_histograms_transformed.png", "Figure 2: Histograms of transformed features", 16)
S += [PageBreak()]
S += fig("figures/03_boxplot_monthly_charges.png", "Figure 3: Box plot - MonthlyCharges by churn", 10)
S += fig("figures/04_countplot_tenure_group.png", "Figure 4: Count plot - tenure group vs churn", 10)
S += fig("figures/05_scatter_tenure_monthly.png", "Figure 5: Scatter plot - tenure vs MonthlyCharges", 11)
S += [Paragraph("7. Five Recommendations for Improving Model Performance", H2)]
S += [Paragraph(f"{i}. {r}", B) for i, r in enumerate(recs, 1)]
SimpleDocTemplate("Day4_Feature_Engineering_Report.pdf", pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm,
                  topMargin=2 * cm, bottomMargin=2 * cm).build(S)
print("Saved Day4_Feature_Engineering_Report.pdf")
