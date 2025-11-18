"""Download SAM checkpoint files."""
import urllib.request
import os
from pathlib import Path

def download_with_progress(url, filename):
    """Download file with progress bar."""
    def reporthook(blocknum, blocksize, totalsize):
        downloaded = blocknum * blocksize
        percent = min(downloaded * 100.0 / totalsize, 100)
        print(f"\rDownloading: {percent:.1f}% ({downloaded / 1024 / 1024:.1f} MB / {totalsize / 1024 / 1024:.1f} MB)", end='')
    
    print(f"Downloading {filename}...")
    urllib.request.urlretrieve(url, filename, reporthook)
    print("\nDownload complete!")

if __name__ == "__main__":
    sam_dir = Path(__file__).parent
    os.chdir(sam_dir)
    
    checkpoints = {
        "sam_vit_l_0b3195.pth": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
        # Uncomment to download other models:
        # "sam_vit_h_4b8939.pth": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
        # "sam_vit_b_01ec64.pth": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
    }
    
    for filename, url in checkpoints.items():
        filepath = sam_dir / filename
        if filepath.exists():
            print(f"✓ {filename} already exists (size: {filepath.stat().st_size / 1024 / 1024:.1f} MB)")
        else:
            download_with_progress(url, filename)
