import pandas as pd
import json

df =pd.read_csv("data/sample_finance_data.csv")

#z.score is used for only discounts and then payment delays

# Calculate z-scores for discounts
#z=x-mean/standard deviation
df["discounts_zscore"]=(df["discounts"]-df["discounts"].mean())/df["discounts"].std()
threshold=3
if(abs(df["discounts_zscore"])>threshold):
    print("Anomaly detected in discounts!")
    