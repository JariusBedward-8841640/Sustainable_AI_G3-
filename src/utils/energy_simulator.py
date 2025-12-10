import pandas as pd
import numpy as np

def generate_energy_data(df):
    #Generate synthetic energy labels for testing

    #Create synth energy consumption column for testing purposes

    df = df.copy()
    df["energy_kwh"] = (
        0.5 #baseline energy cost
        + df["token_count"] * 0.003 #increass energy proportionally with number of tokens in prompt
        + df["avg_word_length"] * 0.10 # slightly increases energy prompt with longer words
        + df ["token_count"] * df["avg_word_length"] * 0.001
        + (df["token_count"] ** 2) * 0.00005
        + np.random.normal(0,0.2, size=len(df)) #small noise to set to make dataset more realistic and avoid perfect linearity
    )
    df.to_csv("data/synthetic/energy_dataset.csv", index=False)
    return df

if __name__ == "__main__":
    df = pd.read_csv("data/processed/features.csv")
    generate_energy_data(df)
    print("Synthetic energy dataset saved to data/synthetic/energy_dataset.csv")