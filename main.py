import pandas as pd

EXPECTED_ROWS = 2610 # research project explanatory PDF
EXPECTED_NILOTINIB = 85 # research project explanatory PDF

FREE_CMAX = 60.4 # in nM. p.2 Table 1

df = pd.read_csv("data/raw/mergedpatchclampdata-20160514.csv")

nilotinib = df[df["Drug"] == "Nilotinib"]

print(len(df) == EXPECTED_ROWS)
print(len(nilotinib) == EXPECTED_NILOTINIB)