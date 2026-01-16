import os
import urllib.request

SAMPLES_DIR = os.path.join(os.path.dirname(__file__), '../data/samples')
os.makedirs(SAMPLES_DIR, exist_ok=True)

# Sample arXiv PDF (Material Science / Chemistry related if possible)
# Using a stable arXiv link.
# 2301.00001 is generic. Let's try to find a specific one or just use a dummy.
# A known good open access paper with figures.
URLS = [
    "https://arxiv.org/pdf/2310.16875.pdf", # "Machine Learning for Materials Science" related
]

def download_samples():
    print("Downloading samples...")
    for url in URLS:
        name = url.split('/')[-1]
        path = os.path.join(SAMPLES_DIR, name)
        if not os.path.exists(path):
            try:
                print(f"Downloading {url}...")
                urllib.request.urlretrieve(url, path)
                print(f"Saved to {path}")
            except Exception as e:
                print(f"Failed to download {url}: {e}")
                print("Please manually place a PDF in data/samples/")
        else:
            print(f"Already exists: {path}")

if __name__ == "__main__":
    download_samples()
