 # =================================================== 
#   Real-World Data Analysis: Titanic Dataset
#   Tools: NumPy, Pandas, Matplotlib
# =================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── 1. LOAD DATA ───────────────────────────────────────

url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

# ── 2. BASIC EXPLORATION ──────────────────────────────────
print("=" * 50)
print("DATASET OVERVIEW")
print("=" * 50)
print(f"Shape       : {df.shape[0]} rows × {df.shape[1]} columns")
print(f"\nColumn Names:\n{df.columns.tolist()}")
print(f"\nFirst 5 Rows:\n{df.head()}")
print(f"\nData Types:\n{df.dtypes}")
print(f"\nMissing Values:\n{df.isnull().sum()}")

# ── 3. DATA CLEANING ─────────────────────────────────────────
df["Age"] = df["Age"].fillna(df["Age"].median())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
df.drop(columns=["Cabin"], inplace=True)

print("\n✅ Missing values after cleaning:")
print(df.isnull().sum())

# ── 4. FEATURE ENGINEERING ─────────────────────────────────
df["AgeGroup"] = pd.cut(
    df["Age"],
    bins=[0, 12, 18, 35, 60, 100],
    labels=["Child", "Teen", "Young Adult", "Adult", "Senior"]
)

# ── 5. NUMPY STATS ──────────────────────────────────────────
ages = df["Age"].values

print("\n" + "=" * 50)
print("NUMPY STATISTICS — AGE")
print("=" * 50)
print(f"  Mean Age   : {np.mean(ages):.1f}")
print(f"  Median Age : {np.median(ages):.1f}")
print(f"  Std Dev    : {np.std(ages):.1f}")
print(f"  Min Age    : {np.min(ages):.0f}")
print(f"  Max Age    : {np.max(ages):.0f}")

# ── 6. PANDAS ANALYSIS ──────────────────────────────────────
print("\n" + "=" * 50)
print("PANDAS ANALYSIS")
print("=" * 50)

survival_rate = df["Survived"].mean() * 100
print(f"\nOverall Survival Rate: {survival_rate:.1f}%")

print("\nSurvival by Gender:")
print(df.groupby("Sex")["Survived"].mean().apply(lambda x: f"{x*100:.1f}%"))

print("\nSurvival by Passenger Class:")
print(df.groupby("Pclass")["Survived"].mean().apply(lambda x: f"{x*100:.1f}%"))

print("\nAverage Fare by Class:")
print(df.groupby("Pclass")["Fare"].mean().round(2))

print("\nSurvival by Age Group:")
print(df.groupby("AgeGroup", observed=True)["Survived"].mean().apply(lambda x: f"{x*100:.1f}%"))

# ── 7. VISUALISATION ────────────────────────────────────────
fig = plt.figure(figsize=(14, 10))
fig.suptitle("Titanic Dataset — Exploratory Data Analysis", fontsize=16, fontweight="bold", y=0.98)
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]

ax1 = fig.add_subplot(gs[0, 0])
counts = df["Survived"].value_counts()
ax1.bar(["Did Not Survive", "Survived"], counts.values, color=[COLORS[3], COLORS[2]], edgecolor="white")
ax1.set_title("Overall Survival Count", fontweight="bold")
ax1.set_ylabel("Count")
for i, v in enumerate(counts.values):
    ax1.text(i, v + 5, str(v), ha="center", fontweight="bold")

ax2 = fig.add_subplot(gs[0, 1])
gender_survival = df.groupby("Sex")["Survived"].mean() * 100
bars = ax2.bar(gender_survival.index, gender_survival.values, color=[COLORS[0], COLORS[1]], edgecolor="white")
ax2.set_title("Survival Rate by Gender", fontweight="bold")
ax2.set_ylabel("Survival Rate (%)")
ax2.set_ylim(0, 100)
for bar in bars:
    ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
             f"{bar.get_height():.1f}%", ha="center", fontweight="bold")

ax3 = fig.add_subplot(gs[0, 2])
class_survival = df.groupby("Pclass")["Survived"].mean() * 100
ax3.bar(["1st Class", "2nd Class", "3rd Class"], class_survival.values,
        color=COLORS[:3], edgecolor="white")
ax3.set_title("Survival Rate by Passenger Class", fontweight="bold")
ax3.set_ylabel("Survival Rate (%)")
ax3.set_ylim(0, 100)
for i, v in enumerate(class_survival.values):
    ax3.text(i, v + 1, f"{v:.1f}%", ha="center", fontweight="bold")

ax4 = fig.add_subplot(gs[1, 0])
survived = df[df["Survived"] == 1]["Age"]
not_survived = df[df["Survived"] == 0]["Age"]
ax4.hist(not_survived, bins=20, alpha=0.6, color=COLORS[3], label="Did Not Survive")
ax4.hist(survived, bins=20, alpha=0.6, color=COLORS[2], label="Survived")
ax4.set_title("Age Distribution by Survival", fontweight="bold")
ax4.set_xlabel("Age")
ax4.set_ylabel("Count")
ax4.legend()

ax5 = fig.add_subplot(gs[1, 1])
data_by_class = [df[df["Pclass"] == c]["Fare"].values for c in [1, 2, 3]]
bp = ax5.boxplot(data_by_class, patch_artist=True, labels=["1st", "2nd", "3rd"])
for patch, color in zip(bp["boxes"], COLORS[:3]):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax5.set_title("Fare Distribution by Class", fontweight="bold")
ax5.set_xlabel("Passenger Class")
ax5.set_ylabel("Fare (£)")

ax6 = fig.add_subplot(gs[1, 2])
embark_labels = {"C": "Cherbourg", "Q": "Queenstown", "S": "Southampton"}
embark_survival = df.groupby("Embarked")["Survived"].mean() * 100
labels = [embark_labels.get(k, k) for k in embark_survival.index]
ax6.pie(embark_survival.values, labels=labels, autopct="%1.1f%%",
        colors=COLORS[:3], startangle=90)
ax6.set_title("Avg Survival Rate\nby Embarkation Port", fontweight="bold")

plt.savefig("titanic_analysis.png", dpi=150, bbox_inches="tight")
print("\n✅ Chart saved to titanic_analysis.png")
plt.show()
print("\n🎉 Analysis complete!")
