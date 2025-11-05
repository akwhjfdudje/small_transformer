import os
import urllib.request
import zipfile
import gzip
import shutil

# Directory to store data
DATA_DIR = "data/opensub/en"
os.makedirs(DATA_DIR, exist_ok=True)

URL = "https://object.pouta.csc.fi/OPUS-OpenSubtitles/v1/mono/en.txt.gz"

gz_path = os.path.join(DATA_DIR, "en.txt.gz")
txt_path = os.path.join(DATA_DIR, "en.txt")

# Download the gzipped file if not exists
if not os.path.exists(gz_path):
    print("Downloading OpenSubtitles English corpus...")
    urllib.request.urlretrieve(URL, gz_path)
    print("Download complete.")

# Extract the gz file to plain text
if not os.path.exists(txt_path):
    print("Extracting...")
    with gzip.open(gz_path, 'rb') as f_in:
        with open(txt_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    print(f"Extraction complete. File saved at {txt_path}")

print("Dataset ready. You can now use en.txt for training.")

