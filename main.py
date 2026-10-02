import pandas as pd
import matplotlib.pyplot as plt

# WEEK 1

EXPECTED_ROWS = 2610 # research project explanatory PDF
EXPECTED_NILOTINIB = 85 # research project explanatory PDF

FREE_CMAX = 60.4 # blood concentration of the drug (in nM). p.2 Table 1

df = pd.read_csv("data/raw/mergedpatchclampdata-20160514.csv")

nilotinib = df[df["Drug"] == "Nilotinib"]

print(len(df) == EXPECTED_ROWS)
print(len(nilotinib) == EXPECTED_NILOTINIB)

print(nilotinib["Conc"].unique()) # four different concentrations of the drug that were tested in lab
print(nilotinib["channel"].unique()) # seven ion channels, of course hERG should be here
print(nilotinib["channel"].value_counts()) # make sure the counts across all channels add to EXPECTED_NILOTINIB
print(nilotinib["Units"].unique()) # verify units

print(nilotinib.isna().sum()) # check for empty values within data

print(nilotinib["block"].describe()) # all measurements for channel block
print(nilotinib.sort_values("block").head()) # smallest block values
print(nilotinib.sort_values("block").tail()) # largest block values

print(nilotinib.groupby(["channel", "Conc"]).size()) # groups data by channe and concentration

# WEEKS 2 & 3

channels = nilotinib["channel"].unique()
result = plt.subplots(2, 4, figsize=(14, 7)) # all 7 plots on one screen
fig = result[0]
axes = result[1]
axes = axes.flatten()

for i, channel in enumerate(channels):
    channel_data = nilotinib[nilotinib["channel"] == channel]

    summary = channel_data.groupby("Conc")["block"].agg(["mean", "std"]).reset_index()

    axes[i].scatter(channel_data["Conc"], channel_data["block"])
    axes[i].errorbar(summary["Conc"], summary["mean"], yerr=summary["std"], marker="o", capsize=4)
    axes[i].set_xscale("log")
    axes[i].set_title(channel)
    axes[i].set_xlabel("Nilotinib concentration (nM)")
    axes[i].set_ylabel("Block (%)")

axes[-1].axis("off")

plt.tight_layout()
plt.show()

### IK1 is basically flat until the greatest concentration. Thus, the concentration range did not get high enough to its IC50 value.
### Concentrations Calcium, IKs, Peak sodium, Kv4.3 yield higher block percentages than IK1, though not enough to see their IC50 values.
### Late sodium and especially hERG concentrations were high enough to identify their IC50 values.