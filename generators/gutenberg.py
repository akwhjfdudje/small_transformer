import os
import requests
from tqdm import tqdm
from bs4 import BeautifulSoup

# Directory to save books
OUT_DIR = "data/gutenberg"
os.makedirs(OUT_DIR, exist_ok=True)

BOOK_IDS = [
    1342,  # Pride and Prejudice
    84,    # Frankenstein
    11,    # Alice's Adventures in Wonderland
    2701,  # Moby Dick
    1661,  # Sherlock Holmes
    98,    # A Tale of Two Cities
    4300,  # Ulysses
    2600,  # War and Peace
    76,    # Adventures of Huckleberry Finn
    132    # The Art of War
]

BASE_URL = "https://www.gutenberg.org/files/{id}/{id}-0.txt"

def download_book(book_id, output_dir):
    url = BASE_URL.format(id=book_id)
    try:
        response = requests.get(url, stream=True, timeout=10)
        response.raise_for_status()
        file_path = os.path.join(output_dir, f"{book_id}.txt")
        total = int(response.headers.get('content-length', 0))
        with open(file_path, 'wb') as f, tqdm(
            desc=f"Downloading book {book_id}",
            total=total,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for chunk in response.iter_content(chunk_size=1024):
                f.write(chunk)
                bar.update(len(chunk))
        print(f"Saved book {book_id} to {file_path}")
    except Exception as e:
        print(f"Failed to download book {book_id}: {e}")

def clean_gutenberg_text(file_path):
    """Remove Gutenberg headers and footers"""
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
    start_idx = 0
    end_idx = len(lines)
    # Find start marker
    for i, line in enumerate(lines):
        if line.strip().startswith("*** START OF THIS PROJECT GUTENBERG EBOOK"):
            start_idx = i + 1
            break
    # Find end marker
    for i, line in enumerate(lines):
        if line.strip().startswith("*** END OF THIS PROJECT GUTENBERG EBOOK"):
            end_idx = i
            break
    clean_lines = lines[start_idx:end_idx]
    with open(file_path, 'w', encoding='utf-8') as f:
        f.writelines(clean_lines)
    print(f"Cleaned book {file_path}")

def main():
    for book_id in BOOK_IDS:
        download_book(book_id, OUT_DIR)
        file_path = os.path.join(OUT_DIR, f"{book_id}.txt")
        clean_gutenberg_text(file_path)

if __name__ == "__main__":
    main()
