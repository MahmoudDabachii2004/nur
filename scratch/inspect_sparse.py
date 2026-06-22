import os
import json

DATA_DIR = "/Users/mahmoud/Documents/nur/nur/data"
SPARSE_DIR = os.path.join(DATA_DIR, "sparse")

def main():
    path = os.path.join(SPARSE_DIR, "hadith_sparse.json")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print("Type of data:", type(data))
    keys = list(data.keys())
    print("Number of keys:", len(keys))
    first_key = keys[0]
    print("First key:", first_key)
    print("Value for first key:", type(data[first_key]))
    # Print the first few keys and values of the first item
    first_val = data[first_key]
    first_val_keys = list(first_val.keys())
    print("Inner keys preview (first 10):", first_val_keys[:10])
    for k in first_val_keys[:10]:
        print(f"  {k} ({type(k)}): {first_val[k]}")

if __name__ == "__main__":
    main()
