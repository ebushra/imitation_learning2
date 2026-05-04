'''
This file runs rollouts (using MLP) on each file in the Acrobot data section and reports data such as
accuracy, zero-one loss, log-loss, rollout lengths, rollout times, rollout success rates, and a measure of 
overlap between the states seen in training vs the states seen in the rollout. It stores these in a dictionary
and reports them.
'''

import os
import glob
import json
import numpy as np
import pandas as pd
import time

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, log_loss,  zero_one_loss

import gymnasium as gym
from sklearn.neighbors import NearestNeighbors

DATA_DIR = "/var/data/human_data"
PATTERN = os.path.join(DATA_DIR, "acrobot*.csv")

def parse_state(s):
    try:
        if not isinstance(s, str):
            return None
        return np.array(json.loads(s), dtype=float)
    except:
        return None

def rollout_model(model, scaler, episodes=25):

    env = gym.make("Acrobot-v1")

    rollout_lengths = []
    all_states = []
    rollout_times = []
    successes = 0

    for ep in range(episodes):

        obs, _ = env.reset()
    
        start_time = time.time()
    
        done = False
        steps = 0
        episode_states = []

        while not done and steps < 500:

            state = np.array(obs, dtype=float).reshape(1, -1)
            state = scaler.transform(state)

            action = model.predict(state)[0]

            obs, reward, terminated, truncated, _ = env.step(int(action))
            done = terminated or truncated

            episode_states.append(obs)
            steps += 1

        episode_time = time.time() - start_time
        rollout_lengths.append(steps)
        rollout_times.append(episode_time)
        all_states.extend(episode_states)

        if steps < 500:
            successes += 1

    env.close()

    success_rate = successes / episodes

    return rollout_lengths, rollout_times, np.array(all_states), success_rate

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

        episode_lengths = (
            df.groupby(["user_id", "episode"])["step"]
            .max()
            .reset_index()
        )
        
        valid_episodes = episode_lengths[
            episode_lengths["step"] <= 500
        ][["user_id", "episode"]]
        
        before_rows = len(df)
        
        df = df.merge(
            valid_episodes,
            on=["user_id", "episode"],
            how="inner"
        )

        print("Dropped rows from long episodes:", before_rows - len(df))
        
        human_episode_lengths = (
            df.groupby(["user_id", "episode"])["step"]
            .max()
            .values
        )
        
        avg_human_length = np.mean(human_episode_lengths)
        
        print("\nAverage human episode length (steps):", avg_human_length)

        
        human_episode_times = (
            df.groupby(["user_id", "episode"])["t"]
            .agg(lambda x: x.max() - x.min())
            .values
        )
        
        avg_human_time = np.mean(human_episode_times)
        
        print("Average human episode duration (time):", avg_human_time)

        X = np.vstack(df["state_parsed"].values)
        y = df["action"].astype(int).values

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
        y_proba = model.predict_proba(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        
        print("\nModel accuracy:", accuracy)
        
        zero_one = zero_one_loss(y_test, y_pred)
        
        print("0-1 loss:", zero_one)
        
        cross_entropy_loss = log_loss(y_test, y_proba)
        
        print("Cross-entropy loss:", cross_entropy_loss)
        
        # noisiness definition
        noisiness = cross_entropy_loss

        rollout_lengths, rollout_times, X_rollout, rollout_success = rollout_model(model, scaler)

        avg_rollout_length = np.mean(rollout_lengths)
        avg_rollout_time = np.mean(rollout_lengths) * 0.2

        print("\nRollout lengths:", rollout_lengths)
        print("Average rollout length:", avg_rollout_length)
        print("Average rollout duration (time):", avg_rollout_time)
        print("Rollout success rate:", rollout_success)

        nn = NearestNeighbors(n_neighbors=1)
        nn.fit(X_train)

        distances, _ = nn.kneighbors(X_rollout)

        overlap = np.exp(-distances.mean())

        print("Overlap:", overlap)

        results[os.path.basename(f)] = {
            "human_episode_len": float(avg_human_length),
            "human_episode_time": float(avg_human_time),
            "accuracy": float(accuracy),
            "zero_one_loss": float(zero_one),
            "noisiness": float(noisiness),
            "rollout_len": float(avg_rollout_length),
            "rollout_time": float(avg_rollout_time),
            "overlap": float(overlap),
            "rollout_success": float(rollout_success),
        }

    except Exception as e:
        print("\nFAILED:")
        print(e)

print("\n\n================ FINAL RESULTS ================\n")
print(json.dumps(results, indent=4))
