import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import fisher_exact
import numpy as np

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv(r"C:\Users\nahid_vv0xche\Downloads\Field Trial.csv")

df["Farm_num"] = df["FarmID"].str.extract(r'(\d+)').astype(int)
df = df.sort_values("Farm_num")

# -----------------------------
# CALCULATE RATES (%)
# -----------------------------
df["Pre_rate"] = (df["Abortions_prevaccination"] / df["Number_of_Cattle"]) * 100
df["Post_rate"] = (df["Abortions_post_vaccination"] / df["Number_of_Cattle"]) * 100
df["Vacc_rate"] = (df["Vaccinated_animal_aborted"] / df["Number_vaccinated"]) * 100
df = df.fillna(0)

# -----------------------------
# SIGNIFICANCE FUNCTION
# -----------------------------
def stars(p):
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return ""

# -----------------------------
# PER-FARM FISHER TEST
# -----------------------------
df["p_value"] = np.nan
df["significance"] = ""

for i, row in df.iterrows():
    table = [
        [row["Abortions_prevaccination"],
         row["Number_of_Cattle"] - row["Abortions_prevaccination"]],
        [row["Abortions_post_vaccination"],
         row["Number_of_Cattle"] - row["Abortions_post_vaccination"]]
    ]

    # Skip farms with no variation
    if sum(map(sum, table)) == 0:
        continue

    _, p = fisher_exact(table)
    df.loc[i, "p_value"] = p
    df.loc[i, "significance"] = stars(p)

# -----------------------------
# PLOT
# -----------------------------
fig, ax = plt.subplots(figsize=(16, 6))

ax.plot(df["FarmID"], df["Pre_rate"], marker="o", label="Pre-vaccination")
ax.plot(df["FarmID"], df["Post_rate"], marker="^", linestyle="--", label="Post-vaccination")
ax.plot(df["FarmID"], df["Vacc_rate"], marker="s", linestyle=":", label="Vaccinated Abortion")

ax.set_xlabel("Farm ID", fontsize=12, fontweight="bold")
ax.set_ylabel("Abortion Rate (%)", fontsize=12, fontweight="bold")

ax.tick_params(axis="x", rotation=90)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", linestyle="--", alpha=0.3)
ax.legend(frameon=False)

# -----------------------------
# ADD STARS PER FARM
# -----------------------------
for i, row in df.iterrows():
    if row["significance"] != "":
        y = max(row["Pre_rate"], row["Post_rate"]) + 0.8
        ax.text(i, y, row["significance"],
                ha="center", va="bottom",
                fontsize=10, fontweight="bold")

plt.tight_layout()

# -----------------------------
# SAVE FIGURE
# -----------------------------
plt.savefig(
    r"C:\Users\nahid_vv0xche\Downloads\New folder (11)\Abortion_Rate_PerFarm_Significance.tiff",
    dpi=600,
    format="tiff",
    bbox_inches="tight"
)

plt.show()
