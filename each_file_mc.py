'''
This file runs rollouts (using MLP) on each file in the MountainCar data section and reports data such as
accuracy, zero-one loss, log-loss, rollout lengths, rollout times, rollout success rates, and a measure of 
overlap between the states seen in training vs the states seen in the rollout. It stores these in a dictionary
and reports them.
'''

import os
import glob
import json
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, log_loss, zero_one_loss
from sklearn.neighbors import NearestNeighbors

import gymnasium as gym

DATA_DIR = "/var/data/human_data"
PATTERN = os.path.join(DATA_DIR, "mountaincar*.csv")

def parse_state(s):
    try:
        if not isinstance(s, str):
            return None
        return np.array(json.loads(s), dtype=float)
    except:
        return None


def rollout_model(model, scaler, episodes=25):
    env = gym.make("MountainCar-v0")

    rollout_lengths = []
    all_states = []
    successes = 0

    for _ in range(episodes):

        obs, _ = env.reset()
        done = False
        steps = 0
        episode_states = []

        while not done and steps < 200:

            state = np.array(obs, dtype=float).reshape(1, -1)
            state = scaler.transform(state)

            action = model.predict(state)[0]

            obs, reward, terminated, truncated, _ = env.step(int(action))
            done = terminated or truncated

            episode_states.append(obs)
            steps += 1

        rollout_lengths.append(steps)
        all_states.extend(episode_states)

        if reward >= 0 or steps < 200:
            successes += 1

    env.close()

    return rollout_lengths, np.array(all_states), successes / episodes

files = glob.glob(PATTERN)
results = {}

print("\nFound files:")
for f in files:
    print(" -", f)

for f in files:

    print("\n" + "=" * 70)
    print("FILE:", os.path.basename(f))
    print("=" * 70)

    try:
        df = pd.read_csv(f)

        print("Rows:", len(df))

        if "training" in df.columns:
            df = df[df["training"] != True]

        df["state_parsed"] = df["state"].apply(parse_state)

        before = len(df)
        df = df.dropna(subset=["state_parsed"])
        print("Dropped bad rows:", before - len(df))

        if "user_id" in df.columns and "episode" in df.columns:
            human_episode_lengths = (
                df.groupby(["user_id", "episode"])["step"]
                .max()
                .values
            )
        else:
            human_episode_lengths = df.groupby("episode")["step"].max().values

        avg_human_length = float(np.mean(human_episode_lengths))

        print("\nAverage human episode length:", avg_human_length)

        human_episode_times = (
            df.groupby(["user_id", "episode"])["t"]
            .agg(lambda x: x.max() - x.min())
            .values
        )
        
        avg_human_time = float(np.mean(human_episode_times))
        
        print("Average human episode time (s):", avg_human_time)

        X = np.vstack(df["state_parsed"].values)
        y = df["action"].astype(int).values

        print("Dataset shape:", X.shape)

        if len(X) < 50:
            print("Skipping: not enough data")
            continue

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        model = MLPClassifier(
            hidden_layer_sizes=(64, 64),
            max_iter=200,
            random_state=42
        )

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        accuracy = float(accuracy_score(y_test, y_pred))
        zero_one = float(zero_one_loss(y_test, y_pred))
        
        print("\nModel accuracy:", accuracy)
        print("0-1 loss:", zero_one)

        y_proba = model.predict_proba(X_test)

        mlp_val_loss = float(log_loss(y_test, y_proba))
        noisiness = mlp_val_loss 

        print("MLP validation loss (noisiness):", mlp_val_loss)

        rollout_lengths, X_rollout, rollout_success = rollout_model(model, scaler)

        print("\nRollout lengths:", rollout_lengths)
        print("Average rollout length:", np.mean(rollout_lengths))
        print("Rollout success rate:", rollout_success)

        dt = 0.02

        rollout_times = np.array(rollout_lengths) * dt
        avg_rollout_time = float(np.mean(rollout_times))
        
        print("Average rollout time (s):", avg_rollout_time)

        nn = NearestNeighbors(n_neighbors=1)
        nn.fit(X_train)

        distances, _ = nn.kneighbors(X_rollout)

        overlap = np.exp(-distances.mean())

        print("Overlap:", overlap)

        results[os.path.basename(f)] = {
            "human_episode_len": avg_human_length,
            "human_episode_time": avg_human_time,
        
            "accuracy": accuracy,
            "zero_one_loss": zero_one,
            "mlp_val_loss": mlp_val_loss,
            "noisiness": noisiness,
        
            "rollout_len": float(np.mean(rollout_lengths)),
            "rollout_time": avg_rollout_time,
        
            "overlap": float(overlap),
            "rollout_success": float(rollout_success),
        }

    except Exception as e:
        print("\nFAILED:")
        print(e)


print("\n\n================ FINAL RESULTS ================\n")
print(json.dumps(results, indent=4))
