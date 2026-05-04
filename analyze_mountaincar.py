'''
This file loops through each MountainCar CSV and runs an 80/20 train-test split using MLP Classifier. It
reports statistics such as accuracy, precision, recall, and f1-score, as well as providing confusion matrices
for both the training and validation sets.
'''
import os
import glob
import json
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

DATA_DIR = "/var/data/human_data"
PATTERN = os.path.join(DATA_DIR, "mountaincar*.csv")

files = glob.glob(PATTERN)

print("\nFound files:")
for f in files:
    print(" -", f)

dfs = []

for f in files:
    try:
        df = pd.read_csv(f)
        df["source_file"] = os.path.basename(f)
        dfs.append(df)
        print(f"Loaded {f} with {len(df)} rows")
    except Exception as e:
        print(f"Failed to load {f}: {e}")

if not dfs:
    raise RuntimeError("No data loaded.")

df = pd.concat(dfs, ignore_index=True)

print("\nTotal rows:", len(df))

def parse_state(s):
    try:
        if not isinstance(s, str):
            return None
        return np.array(json.loads(s), dtype=float)
    except:
        return None

df["state_parsed"] = df["state"].apply(parse_state)

before = len(df)
df = df.dropna(subset=["state_parsed"])
after = len(df)

print(f"Dropped {before - after} bad rows")

if "training" in df.columns:
    df = df[df["training"] == False]
    print("Filtered out training data. Rows left:", len(df))

# Keep only episodes that actually finished
finished_episodes = df[df["done"] == True]["episode"].unique()

df = df[df["episode"].isin(finished_episodes)]

print("After removing unfinished episodes:", len(df))

X = np.vstack(df["state_parsed"].values)
y = df["action"].astype(int).values

print("\nDataset shape:", X.shape)
print("Action distribution:", np.bincount(y))

if len(X) == 0:
    raise RuntimeError("No valid data.")

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = MLPClassifier(
    hidden_layer_sizes=(64, 64),
    max_iter=100,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\n=== RESULTS ===")
print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

random_preds = np.random.choice(np.unique(y), size=len(y_test))
majority_class = np.bincount(y_train).argmax()
majority_preds = np.full_like(y_test, majority_class)

print("\n=== BASELINES ===")
print("Random:", accuracy_score(y_test, random_preds))
print("Majority:", accuracy_score(y_test, majority_preds))

print("\n=== PERFORMANCE METRICS ===")

# Episode length
episode_lengths = df.groupby("episode")["step"].max()
print("Average episode length:", episode_lengths.mean())

# Time per episode
episode_time = df.groupby("episode")["t"].max()
print("Average completion time:", episode_time.mean())

# Success rate
if "success" in df.columns:
    success_rate = df.groupby("episode")["success"].max().mean()
    print("Success rate:", success_rate)
