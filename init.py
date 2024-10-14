import os
import urllib.request
import tarfile
import tkinter as tk
from tkinter import ttk, messagebox

# Define the base directory in the user's home directory
base_dir = os.path.join(os.path.expanduser("~"), ".paddleocr", "whl")

# Define the file paths and URLs
files = {
    os.path.join(base_dir, "det", "en", "en_PP-OCRv3_det_infer"): "https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_det_infer.tar",
    os.path.join(base_dir, "rec", "en", "en_PP-OCRv4_rec_infer"): "https://paddleocr.bj.bcebos.com/PP-OCRv4/english/en_PP-OCRv4_rec_infer.tar",
    os.path.join(base_dir, "cls", "ch_ppocr_mobile_v2.0_cls_infer"): "https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar"
}

def download_and_unpack(file_path, url, progress_var, progress_bar, root):
    tar_file_path = file_path + ".tar"
    
    # Create directories if they don't exist
    os.makedirs(os.path.dirname(tar_file_path), exist_ok=True)
    
    # Download the file with progress update
    def reporthook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            percent = downloaded / total_size * 100
            progress_var.set(percent)
            progress_bar.update()
    
    print(f"Downloading {url}...")
    urllib.request.urlretrieve(url, tar_file_path, reporthook)
    print(f"Downloaded {tar_file_path}")
    
    # Unpack the tar file
    if tarfile.is_tarfile(tar_file_path):
        print(f"Unpacking {tar_file_path}...")
        with tarfile.open(tar_file_path, "r") as tar:
            tar.extractall(path=os.path.dirname(file_path))
        print(f"Unpacked {tar_file_path}")
        
        # Delete the tar file after unpacking
        os.remove(tar_file_path)
        print(f"Deleted {tar_file_path}")
    else:
        print(f"{tar_file_path} is not a valid tar file")

def check_and_download_files(files):
    root = tk.Tk()
    root.title("Downloading Models")
    root.geometry("400x100")
    
    progress_var = tk.DoubleVar()
    progress_bar = ttk.Progressbar(root, variable=progress_var, maximum=100)
    progress_bar.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
    
    for file_path, url in files.items():
        if not os.path.exists(file_path):
            download_and_unpack(file_path, url, progress_var, progress_bar, root)
        else:
            print(f"{file_path} already exists")
    
    root.destroy()

# Check and download files
check_and_download_files(files)