import os
import pickle

path = "./artifacts/xai-nlp-benchmark-2024-04-23-21-20-02/xai_sentiment"
path = "./artifacts/xai-nlp-benchmark-2024-04-23-21-20-02/xai_gender_no_sub_samp"
records_name = "xai_records.pkl"

out = []
for file in os.listdir(path):
    if file.startswith(records_name):
        with open(os.path.join(path, file), "rb") as f:
            data = pickle.load(f)
            out += data

print(f"Read {len(out)} datapoints")

with open(os.path.join(path, records_name), "wb") as f:
    pickle.dump(out, f)
        
