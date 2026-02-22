import pandas as pd, json

df = pd.read_csv(r"data\creditcard.csv")
row = df.iloc[0].to_dict()

features = {f"V{i}": float(row[f"V{i}"]) for i in range(1, 29)}
features["Amount"] = float(row["Amount"])

with open("payload.json", "w", encoding="utf-8") as f:
    json.dump({"features": features}, f)

print("Wrote payload.json with", len(features), "features")