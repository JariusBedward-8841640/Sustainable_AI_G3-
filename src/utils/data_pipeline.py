import pandas as pd
import numpy as np
import re
from feature_engineering import compute_feature
import nltk

from nltk.corpus import stopwords

nltk.download("stopwords", quiet=True)
stop_words = set(stopwords.words("english"))

def clean_text(text):
    if not isinstance(text, str):
        text = str(text)

        # lowercase
    text = text.lower()

    # remove punctuation and numbers
    text = re.sub(r"[^a-z\s]", " ", text)

    # normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # remove stopwords
    words = [w for w in text.split() if w not in stop_words]

    # reassemble
    cleaned = " ".join(words)

    # filter very short texts
    return cleaned if len(cleaned) > 5 else None


def load_clean_raw(csv_path="data/raw/raw_prompts.csv"):
    #load csv and clean prompts
    df = pd.read_csv(csv_path)
    df["prompt"] = df["prompt"].apply(clean_text)
    df = df.dropna(subset=["prompt"]).reset_index(drop=True)
    return df

def create_feature_pipeline(df, num_layers_list, training_hours_list, flops_per_hour_list):
    #Apply feature computation to entire dataset

    rows = []
    for i, row in df.iterrows():
        features = compute_feature(
            row["prompt"],
            num_layers_list[i],
            training_hours_list[i],
            flops_per_hour_list[i],
        )
        rows.append(features)
    return pd.DataFrame(rows)

if __name__ == "__main__":
    #load and clean raw prompts
    df = load_clean_raw()

    #Generate random model aprams for synthetic features
    n = len(df)
    layers = np.random.randint(4,48, size=n)
    hours = np.random.uniform(0.5, 20, size=n)
    flops = np.random.uniform(1e9, 1e12, size=n)

    #compute features
    features_df = create_feature_pipeline(df, layers, hours, flops)

    #Save processed dataset
    features_df.to_csv("data/processed/features_df.csv", index=False)
    print("Processed dataset saved to data/processed/features_df.csv")