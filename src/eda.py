"""
eda.py
------
Performs Exploratory Data Analysis on the cleaned Telco Churn dataset.
Creates and saves 7+ meaningful visualizations to the outputs/eda/ folder.

Visualizations produced:
  1. Churn distribution (pie + bar)
  2. Churn by Contract type
  3. Churn by Internet Service type
  4. Churn by tenure (histogram with KDE)
  5. MonthlyCharges vs Churn (violin plot)
  6. Correlation heatmap (numeric features)
  7. Senior Citizen vs Churn
  8. Payment Method vs Churn
  9. Stacked bar: multiple service features vs Churn
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")          # Non-interactive backend for saving files
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(ROOT, "data", "cleaned_churn.csv")
OUT_DIR   = os.path.join(ROOT, "outputs", "eda")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Palette ──────────────────────────────────────────────────────────────────
COLORS = {"No": "#4A90D9", "Yes": "#E74C3C"}
PALETTE = [COLORS["No"], COLORS["Yes"]]

sns.set_theme(style="whitegrid", context="notebook", font_scale=1.1)
plt.rcParams["figure.dpi"] = 120


def load() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    # Restore string labels for plots
    df["Churn_label"] = df["Churn"].map({0: "No", 1: "Yes"})
    print(f"[EDA] Loaded cleaned data: {df.shape}")
    return df


# ── Plot helpers ─────────────────────────────────────────────────────────────

def save(fig: plt.Figure, name: str) -> None:
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [SAVED] {path}")


def plot_churn_distribution(df: pd.DataFrame) -> None:
    """Plot 1 – Churn class distribution (bar + pie)."""
    counts = df["Churn_label"].value_counts()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Customer Churn Distribution", fontsize=15, fontweight="bold")

    # Bar chart
    axes[0].bar(counts.index, counts.values,
                color=[COLORS[k] for k in counts.index], edgecolor="white", width=0.5)
    for i, v in enumerate(counts.values):
        axes[0].text(i, v + 50, f"{v}\n({v/len(df)*100:.1f}%)",
                     ha="center", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Churn", fontsize=12)
    axes[0].set_ylabel("Number of Customers", fontsize=12)
    axes[0].set_title("Count of Churned vs Retained", fontsize=12)
    axes[0].set_ylim(0, counts.max() * 1.2)

    # Pie chart
    axes[1].pie(counts.values, labels=counts.index, autopct="%1.1f%%",
                colors=PALETTE, startangle=140,
                wedgeprops={"edgecolor": "white", "linewidth": 2})
    axes[1].set_title("Churn Proportion", fontsize=12)

    save(fig, "01_churn_distribution.png")


def plot_churn_by_contract(df: pd.DataFrame) -> None:
    """Plot 2 – Churn rate by Contract type."""
    ct = df.groupby("Contract")["Churn"].mean().reset_index()
    ct["Churn %"] = ct["Churn"] * 100

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(ct["Contract"], ct["Churn %"],
                   color=["#E74C3C", "#F39C12", "#2ECC71"], edgecolor="white")
    for bar in bars:
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{bar.get_width():.1f}%", va="center", fontsize=11, fontweight="bold")
    ax.set_xlabel("Churn Rate (%)", fontsize=12)
    ax.set_title("Churn Rate by Contract Type", fontsize=13, fontweight="bold")
    ax.set_xlim(0, ct["Churn %"].max() * 1.2)

    save(fig, "02_churn_by_contract.png")


def plot_churn_by_internet(df: pd.DataFrame) -> None:
    """Plot 3 – Churn by Internet Service."""
    ct = df.groupby(["InternetService", "Churn_label"]).size().reset_index(name="Count")
    fig, ax = plt.subplots(figsize=(9, 5))
    services = df["InternetService"].unique()
    width = 0.35
    x = np.arange(len(services))
    no_vals  = [ct[(ct["InternetService"] == s) & (ct["Churn_label"] == "No")]["Count"].values[0]
                if len(ct[(ct["InternetService"] == s) & (ct["Churn_label"] == "No")]) > 0 else 0
                for s in services]
    yes_vals = [ct[(ct["InternetService"] == s) & (ct["Churn_label"] == "Yes")]["Count"].values[0]
                if len(ct[(ct["InternetService"] == s) & (ct["Churn_label"] == "Yes")]) > 0 else 0
                for s in services]

    ax.bar(x - width/2, no_vals,  width, label="No Churn",  color=COLORS["No"],  edgecolor="white")
    ax.bar(x + width/2, yes_vals, width, label="Churned",   color=COLORS["Yes"], edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels(services, fontsize=11)
    ax.set_ylabel("Number of Customers", fontsize=12)
    ax.set_title("Churn by Internet Service Type", fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)

    save(fig, "03_churn_by_internet.png")


def plot_tenure_distribution(df: pd.DataFrame) -> None:
    """Plot 4 – Tenure distribution split by churn."""
    fig, ax = plt.subplots(figsize=(10, 5))
    for label, color in COLORS.items():
        subset = df[df["Churn_label"] == label]["tenure"]
        ax.hist(subset, bins=30, alpha=0.7, color=color, label=f"Churn={label}", edgecolor="white")
    ax.set_xlabel("Tenure (months)", fontsize=12)
    ax.set_ylabel("Number of Customers", fontsize=12)
    ax.set_title("Tenure Distribution by Churn Status", fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)

    save(fig, "04_tenure_distribution.png")


def plot_monthly_charges_violin(df: pd.DataFrame) -> None:
    """Plot 5 – Monthly charges vs Churn (violin)."""
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.violinplot(data=df, x="Churn_label", y="MonthlyCharges",
                   hue="Churn_label", palette=PALETTE, ax=ax, inner="quartile", cut=0, legend=False)
    ax.set_xlabel("Churn", fontsize=12)
    ax.set_ylabel("Monthly Charges ($)", fontsize=12)
    ax.set_title("Monthly Charges Distribution by Churn Status", fontsize=13, fontweight="bold")

    save(fig, "05_monthly_charges_violin.png")


def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    """Plot 6 – Correlation heatmap of numeric features."""
    num_cols = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges", "Churn"]
    corr = df[num_cols].corr()

    fig, ax = plt.subplots(figsize=(8, 6))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", mask=mask,
                linewidths=0.5, square=True, ax=ax, cbar_kws={"shrink": 0.8})
    ax.set_title("Correlation Heatmap (Numeric Features)", fontsize=13, fontweight="bold")

    save(fig, "06_correlation_heatmap.png")


def plot_senior_citizen_churn(df: pd.DataFrame) -> None:
    """Plot 7 – Senior Citizen vs Churn."""
    df2 = df.copy()
    df2["SeniorCitizen_label"] = df2["SeniorCitizen"].map({0: "Non-Senior", 1: "Senior"})
    ct = df2.groupby(["SeniorCitizen_label", "Churn_label"]).size().reset_index(name="Count")

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=ct, x="SeniorCitizen_label", y="Count", hue="Churn_label",
                palette=PALETTE, ax=ax, edgecolor="white")
    ax.set_xlabel("Customer Segment", fontsize=12)
    ax.set_ylabel("Number of Customers", fontsize=12)
    ax.set_title("Senior Citizen vs Churn", fontsize=13, fontweight="bold")
    ax.legend(title="Churn", fontsize=11)

    save(fig, "07_senior_citizen_churn.png")


def plot_payment_method_churn(df: pd.DataFrame) -> None:
    """Plot 8 – Payment Method vs Churn rate."""
    ct = df.groupby("PaymentMethod")["Churn"].mean().sort_values(ascending=False) * 100
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(ct.index, ct.values,
                  color=["#E74C3C", "#F39C12", "#4A90D9", "#2ECC71"], edgecolor="white")
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{bar.get_height():.1f}%", ha="center", fontsize=10, fontweight="bold")
    ax.set_ylabel("Churn Rate (%)", fontsize=12)
    ax.set_title("Churn Rate by Payment Method", fontsize=13, fontweight="bold")
    ax.set_ylim(0, ct.max() * 1.25)
    plt.xticks(rotation=20, ha="right")

    save(fig, "08_payment_method_churn.png")


def plot_service_features_churn(df: pd.DataFrame) -> None:
    """Plot 9 – Multiple service features vs Churn (stacked bar)."""
    service_cols = [
        "PhoneService", "MultipleLines", "OnlineSecurity",
        "OnlineBackup", "DeviceProtection", "TechSupport",
        "StreamingTV", "StreamingMovies"
    ]
    churn_rates = {}
    for col in service_cols:
        # Only compute for values that are "Yes" (ignoring "No phone service" etc.)
        mask = df[col] == "Yes"
        churn_rates[col] = df.loc[mask, "Churn"].mean() * 100

    names = list(churn_rates.keys())
    rates = list(churn_rates.values())

    fig, ax = plt.subplots(figsize=(11, 6))
    bar_colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(names)))
    bars = ax.barh(names, rates, color=bar_colors, edgecolor="white")
    for bar in bars:
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                f"{bar.get_width():.1f}%", va="center", fontsize=10, fontweight="bold")
    ax.set_xlabel("Churn Rate among subscribers (%)", fontsize=12)
    ax.set_title("Churn Rate by Service Subscription", fontsize=13, fontweight="bold")
    ax.set_xlim(0, max(rates) * 1.2)

    save(fig, "09_service_features_churn.png")


def run_eda() -> None:
    df = load()
    print("\n[EDA] Generating visualizations...")
    plot_churn_distribution(df)
    plot_churn_by_contract(df)
    plot_churn_by_internet(df)
    plot_tenure_distribution(df)
    plot_monthly_charges_violin(df)
    plot_correlation_heatmap(df)
    plot_senior_citizen_churn(df)
    plot_payment_method_churn(df)
    plot_service_features_churn(df)
    print("\n[EDA] All 9 charts saved to:", OUT_DIR)


if __name__ == "__main__":
    run_eda()
