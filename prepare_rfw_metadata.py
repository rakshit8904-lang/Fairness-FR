from pathlib import Path
import pandas as pd

INPUT_FILE = Path("data/metadata/rfw_metadata.csv")
OUTPUT_FILE = Path("metadata/rfw_metadata.csv")

print("=" * 70)
print("FAIRNESS-FR — RFW METADATA PREPARATION")
print("=" * 70)

df = pd.read_csv(INPUT_FILE)

print(f"\nInput rows: {len(df)}")
print("\nInput columns:")
print(df.columns.tolist())

# Clean column names
df.columns = [str(c).strip() for c in df.columns]

# Detect path column
path_col = "Path" if "Path" in df.columns else "image_path"

# Create FAIRNESS-FR compatible metadata
out = pd.DataFrame()

out["image_path"] = df[path_col].astype(str).str.strip()

# Identity = folder containing the image
out["identity"] = out["image_path"].apply(
    lambda x: "/".join(x.replace("\\", "/").split("/")[:-1])
)

# Group = first folder: African / Asian / Caucasian / Indian
out["group"] = out["image_path"].apply(
    lambda x: x.replace("\\", "/").split("/")[0]
)

# Gender
if "Gender" in df.columns:
    out["gender"] = df["Gender"].astype(str).str.strip()
else:
    out["gender"] = "unknown"

# Age group
if "Age_Category" in df.columns:
    out["age_group"] = df["Age_Category"].astype(str).str.strip()
elif "Age_id" in df.columns:
    out["age_group"] = df["Age_id"].astype(str).str.strip()
else:
    out["age_group"] = "unknown"

# Remove duplicates
before = len(out)
out = out.drop_duplicates(subset=["image_path"]).reset_index(drop=True)

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
out.to_csv(OUTPUT_FILE, index=False)

print("\n" + "=" * 70)
print("METADATA PREPARATION COMPLETE")
print("=" * 70)

print(f"Rows before duplicate removal: {before}")
print(f"Rows written:                  {len(out)}")

print("\nGroups:")
print(out["group"].value_counts().to_string())

print("\nGender:")
print(out["gender"].value_counts().to_string())

print("\nAge groups:")
print(out["age_group"].value_counts().head(20).to_string())

print("\nFirst 5 rows:")
print(out.head().to_string(index=False))

print(f"\nOutput saved to:\n{OUTPUT_FILE.resolve()}")
print("=" * 70)