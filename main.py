import pandas as pd

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