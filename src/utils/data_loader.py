from datasets import load_dataset
import pandas as pd

# Load raw text dataset from hugging face andf save it as csv
def load_dataset_raw():
    #Load yelp review (full train split)
    dataset = load_dataset('yelp_review_full', split="train")

    print(f"Loaded full dataset with {len(dataset)} rows.")
    #create a df with only prompt text
    df = pd.DataFrame({"prompt": [x["text"] for x in dataset]})
    df = df.rename(columns={"text": "prompt"})

    #save raw data to CSV
    df.to_csv("data/raw/raw_prompts.csv", index=False)
    return df

if __name__ == "__main__":
    load_dataset_raw()

