import pandas as pd

EXPECTED_ROWS = 2610
EXPECTED_NILOTINIB = 85

df = pd.read_csv("data/raw/mergedpatchclampdata-20160514.csv")

nilotinib = df[df["Drug"] == "Nilotinib"]

print(len(df) == EXPECTED_ROWS)
print(len(nilotinib) == EXPECTED_NILOTINIB)