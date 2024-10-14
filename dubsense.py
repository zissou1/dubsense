import cv2
import time
import re
import threading
import logging
import asyncio
import aiohttp
import numpy as np
from PIL import ImageGrab, Image, ImageTk
from screeninfo import get_monitors
from paddleocr import PaddleOCR
import tkinter as tk
from tkinter import scrolledtext, messagebox
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from psutil import Process, IDLE_PRIORITY_CLASS, process_iter
import json
import os
import sys
import pystray
from pystray import MenuItem as item
from PIL import Image as PILImage
import subprocess
import urllib.request
import tarfile


# Set process priority to reduce CPU usage
Process().nice(IDLE_PRIORITY_CLASS)

# Global variable to store the latest processed image
latest_image = None
monitoring_active = False

# Config file path
CONFIG_FILE = os.path.join(os.path.dirname(__file__), 'config/config.json')

# Default Configuration
default_config = {
    "search_word": "CTO",
    "check_interval": 2.5,
    "trigger_interval": 15,
    "box_height_percent": 0.22,
    "aspect_ratio": (7, 3),
    "webhook_enabled": True,
    "webhook_url": "http://...",
    "auto_monitor_cod": True,
    "use_gpu": True
}

# Load configuration from file
def load_config():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r') as file:
            return json.load(file)
    else:
        return default_config

# Save configuration to file
def save_config(config):
    os.makedirs(os.path.dirname(CONFIG_FILE), exist_ok=True)
    with open(CONFIG_FILE, 'w') as file:
        json.dump(config, file, indent=4)

# Load initial configuration
CONFIG = load_config()

def update_webhook_enabled():
    CONFIG["webhook_enabled"] = webhook_var.get()
    save_config(CONFIG)

def update_webhook_url():
    CONFIG["webhook_url"] = webhook_entry.get()
    save_config(CONFIG)

def update_auto_monitor_cod():
    toggle_start_stop_buttons()
    CONFIG["auto_monitor_cod"] = auto_start_var.get()
    save_config(CONFIG)

def update_gpu_usage():
    CONFIG["use_gpu"] = use_gpu_var.get()
    save_config(CONFIG)
    # Reinitialize OCR with the new setting
    global ocr
    ocr = initialize_ocr()
    log_message(f"Using {'GPU' if CONFIG['use_gpu'] else 'CPU'}.")

CALL_OF_DUTY_PROCESS_NAME = "cod.exe"  # Update to the actual Call of Duty process name

# Custom logging handler
class StreamToLogger:
    def __init__(self, logger, log_level=logging.INFO):
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ''

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.logger.log(self.log_level, line.rstrip())

    def flush(self):
        pass

# Set logging configuration
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger()

# Redirect stdout and stderr to the logger
sys.stdout = StreamToLogger(logger, logging.INFO)
sys.stderr = StreamToLogger(logger, logging.ERROR)


# Define the base directory in the user's home directory
base_dir = os.path.join(os.path.expanduser("~"), ".paddleocr", "whl")

# Define the file paths and URLs
files = {
    os.path.join(base_dir, "det", "en", "en_PP-OCRv3_det_infer"): "https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_det_infer.tar",
    os.path.join(base_dir, "rec", "en", "en_PP-OCRv4_rec_infer"): "https://paddleocr.bj.bcebos.com/PP-OCRv4/english/en_PP-OCRv4_rec_infer.tar",
    os.path.join(base_dir, "cls", "ch_ppocr_mobile_v2.0_cls_infer"): "https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar"
}

def verify_tar_file(tar_file_path):
    try:
        with tarfile.open(tar_file_path, "r") as tar:
            tar.getmembers()
        return True
    except tarfile.ReadError:
        return False

def download_and_unpack(file_path, url):
    tar_file_path = file_path + ".tar"
    
    # Create directories if they don't exist
    os.makedirs(os.path.dirname(tar_file_path), exist_ok=True)
    
    # Download the file with progress update
    def reporthook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            percent = downloaded / total_size * 100
            #log_message(f"Downloading {url}... {percent:.2f}%")
    
    while True:
        log_message(f"Downloading {url}... Please wait.")
        urllib.request.urlretrieve(url, tar_file_path, reporthook)
        #log_message(f"Downloaded {tar_file_path}")
        
        # Verify the tar file
        if verify_tar_file(tar_file_path):
            break
        else:
            log_message(f"{tar_file_path} is corrupted. Re-downloading...")
            os.remove(tar_file_path)
    
    # Unpack the tar file
    if tarfile.is_tarfile(tar_file_path):
       # log_message(f"Unpacking {tar_file_path}...")
        with tarfile.open(tar_file_path, "r") as tar:
            tar.extractall(path=os.path.dirname(file_path))
        #log_message(f"Unpacked {tar_file_path}")
        
        # Delete the tar file after unpacking
        os.remove(tar_file_path)
        #log_message(f"Deleted {tar_file_path}")
    else:
        log_message(f"{tar_file_path} is not a valid tar file")

def check_and_download_files(files):
    disable_widgets()
    any_downloads = False
    for file_path, url in files.items():
        if not os.path.exists(file_path):
            download_and_unpack(file_path, url)
            any_downloads = True
    if any_downloads:
        log_message("All downloads complete")
    enable_widgets()



# OCR Setup with exception handling
def initialize_ocr():
    try:
        return PaddleOCR(
            use_angle_cls=False,
            use_gpu=CONFIG.get("use_gpu", False),
            lang='en'
        )
    except subprocess.CalledProcessError as e:
        messagebox.showerror("Initialization Error", f"Error initializing OCR: {e}")
        raise
    except Exception as e:
        messagebox.showerror("Initialization Error", f"Error initializing OCR: {e}")
        raise


def capture_screen():
    monitor = get_monitors()[0]
    box_height = int(monitor.height * CONFIG["box_height_percent"])
    box_width = int((box_height * CONFIG["aspect_ratio"][0]) / CONFIG["aspect_ratio"][1])

    offset_x = int(monitor.height * 0.06)
    bbox_x = monitor.x + (monitor.width - box_width) // 2 - offset_x
    bbox_y = monitor.y + (monitor.height - box_height) // 2
    bbox = (bbox_x, bbox_y, bbox_x + box_width, bbox_y + box_height)

    screen = np.array(ImageGrab.grab(bbox=bbox))

    new_width = 300
    resize_factor = new_width / screen.shape[1]
    new_height = int(screen.shape[0] * resize_factor)
    resized_screen = cv2.resize(screen, (new_width, new_height))

    return resized_screen

def process_image(image):
    global latest_image
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    final_width = int(gray_image.shape[1] * 0.5)
    final_height = int(gray_image.shape[0] * 0.5)
    processed_image = cv2.resize(gray_image, (final_width, final_height))
    latest_image = processed_image
    update_image_display()
    return processed_image

def update_image_display():
    if latest_image is not None:
        image_pil = Image.fromarray(latest_image)
        image_tk = ImageTk.PhotoImage(image_pil)
        image_label.config(image=image_tk)
        image_label.image = image_tk

def check_cod_and_monitor():
    while True:
        if auto_start_var.get():
            cod_running = any(proc.info['name'] == CALL_OF_DUTY_PROCESS_NAME for proc in process_iter(['name']))
            if cod_running and not monitoring_active:
                start_monitoring()
            elif not cod_running and monitoring_active:
                stop_monitoring()
        time.sleep(5)

def detect_text(image, search_word):
    results = ocr.ocr(image, cls=False)
    if not results or not results[0]:
        return False
    for line in results[0]:
        text = re.sub(r'[^A-Z]', '', line[1][0].upper())
        if search_word in text.replace("Q", "O"):
            print(f"Detected word: {text}")
            return True
    return False

async def send_webhook(url, retries=3):
    attempt = 0
    while attempt < retries:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url) as response:
                    if response.status == 200:
                        logging.info("Webhook sent successfully.")
                        log_message("Webhook sent successfully!")
                        return
                    else:
                        error_content = await response.text()
                        log_message(f"Failed to send webhook. Status code: {response.status}, Response: {error_content}")
        except aiohttp.ClientError as e:
            log_message(f"Client error sending webhook: {e}")
        except Exception as e:
            log_message(f"Unexpected error sending webhook: {e}")
        attempt += 1
        backoff_time = 2 ** attempt
        log_message(f"Retrying webhook in {backoff_time} seconds... (Attempt {attempt}/{retries})")
        await asyncio.sleep(backoff_time)

    log_message("Failed to send webhook after multiple attempts.")

def test_webhook():
    if CONFIG["webhook_url"]:
        log_message("Testing webhook...")
        asyncio.run(send_webhook(CONFIG["webhook_url"]))
    else:
        log_message("Webhook URL is empty. Please enter a valid URL.")

loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)

def monitor_screen():
    last_webhook_time = 0
    idle_counter = 0
    idle_max = 10

    while monitoring_active:
        try:
            screen = capture_screen()
            processed_image = process_image(screen)
            if detect_text(processed_image, CONFIG["search_word"]):
                idle_counter = 0
                current_time = time.time()
                if current_time - last_webhook_time >= CONFIG["trigger_interval"]:
                    log_message("Victory Detected!")
                    if CONFIG["webhook_enabled"]:
                        loop.run_until_complete(send_webhook(CONFIG["webhook_url"]))
                    last_webhook_time = current_time
            else:
                idle_counter += 1
            time.sleep(CONFIG["check_interval"] if idle_counter < idle_max else CONFIG["check_interval"] * 2)
        except Exception as e:
            log_message(f"Error in monitor: {e}")

def log_error(message):
    log_message(message)
    messagebox.showerror("Error", message)

def start_monitoring():
    global monitoring_active, ocr
    if not monitoring_active:
        monitoring_active = True
        log_message("Monitoring started...")
        toggle_widgets_state("disabled")
        
        # Initialize OCR when monitoring starts
        ocr = initialize_ocr()
        
        monitoring_thread = threading.Thread(target=monitor_screen, daemon=True)
        monitoring_thread.start()

def stop_monitoring():
    global monitoring_active
    if monitoring_active:
        monitoring_active = False
        log_message("Monitoring stopped.")
        toggle_widgets_state("normal")

def toggle_widgets_state(state):
    webhook_checkbox.config(state=state)
    test_button.config(state=state)
    webhook_entry.config(state=state)
    gpu_checkbox.config(state=state)

def toggle_start_stop_buttons():
    if auto_start_var.get():
        start_button.config(state="disabled")
        stop_button.config(state="disabled")
    else:
        start_button.config(state="normal")
        stop_button.config(state="normal")

log_lock = threading.Lock()

def log_message(message):
    with log_lock:
        log_area.config(state=tk.NORMAL)
        log_area.insert(tk.END, f"{message}\n")
        log_area.see(tk.END)
        log_area.config(state=tk.DISABLED)

def disable_widgets():
    for widget in main_frame.winfo_children():
        if widget != log_area:
            try:
                widget.config(state=tk.DISABLED)
            except tk.TclError:
                pass
    log_area.config(state=tk.NORMAL)

def enable_widgets():
    for widget in main_frame.winfo_children():
        if widget != log_area:
            try:
                widget.config(state=tk.NORMAL)
            except tk.TclError:
                pass

app = ttk.Window(themename="darkly")
app.title("DubSense")

icon_path = os.path.join(os.path.dirname(__file__), 'dubsense.ico')
app.iconbitmap(icon_path)
app.resizable(False, False)

main_frame = ttk.Frame(app, padding=(15, 10))
main_frame.grid(row=0, column=0, sticky="nsew")

# Auto Start Monitoring with Call of Duty checkbox
auto_start_var = ttk.BooleanVar(value=CONFIG.get("auto_monitor_cod", True))
auto_start_checkbox = ttk.Checkbutton(main_frame, text="Auto Monitor Call of Duty", variable=auto_start_var, bootstyle="primary-round-toggle", command=update_auto_monitor_cod)
auto_start_checkbox.grid(row=0, column=0, sticky="w", padx=5, pady=5)

# Use GPU checkbox positioned in its own row below "Auto Monitor Call of Duty"
use_gpu_var = ttk.BooleanVar(value=CONFIG.get("use_gpu", False))
gpu_checkbox = ttk.Checkbutton(main_frame, text="Use GPU", variable=use_gpu_var, bootstyle="primary-round-toggle", command=update_gpu_usage)
gpu_checkbox.grid(row=1, column=0, sticky="w", padx=5, pady=5)

# Enable Webhook checkbox positioned below the "Use GPU" checkbox
webhook_var = ttk.BooleanVar(value=CONFIG["webhook_enabled"])
webhook_checkbox = ttk.Checkbutton(main_frame, text="Enable Webhook", variable=webhook_var, bootstyle="primary-round-toggle", command=update_webhook_enabled)
webhook_checkbox.grid(row=2, column=0, sticky="w", padx=5, pady=5)

# Test Webhook button next to the "Enable Webhook" checkbox
test_button = ttk.Button(main_frame, text="Test", command=test_webhook, bootstyle="light")
test_button.grid(row=2, column=1, padx=5, pady=5)

# Webhook URL entry positioned next to the test button
webhook_entry = ttk.Entry(main_frame, width=40)
webhook_entry.insert(0, CONFIG.get("webhook_url", ""))
webhook_entry.bind("<FocusOut>", lambda e: update_webhook_url())
webhook_entry.grid(row=2, column=2, padx=(0, 5), pady=5, sticky="w")

log_area = scrolledtext.ScrolledText(main_frame, width=80, height=10, state=DISABLED, background="#2e2e2e", foreground="white")
log_area.grid(row=3, column=0, columnspan=3, padx=5, pady=10)

image_label = ttk.Label(main_frame)
image_label.grid(row=4, column=0, columnspan=3, padx=5, pady=10)

start_button = ttk.Button(main_frame, text="Start Monitoring", command=start_monitoring, bootstyle="primary")
start_button.grid(row=5, column=0, padx=10, pady=10)
stop_button = ttk.Button(main_frame, text="Stop Monitoring", command=stop_monitoring, bootstyle="primary")
stop_button.grid(row=5, column=2, padx=10, pady=10)

toggle_start_stop_buttons()

# Function to run check_and_download_files in a separate thread
def run_check_and_download_files():
    check_and_download_files(files)

# Start the download process in a separate thread after the GUI is set up
download_thread = threading.Thread(target=run_check_and_download_files, daemon=True)
download_thread.start()

auto_start_thread = threading.Thread(target=check_cod_and_monitor, daemon=True)
auto_start_thread.start()

# Tray Icon Setup
def on_quit(icon=None, item=None):
    global monitoring_active
    monitoring_active = False
    save_config(CONFIG)  # Ensure latest config is saved on exit
    if icon:
        icon.stop()
    app.quit()

def show_window(icon, item):
    icon.stop()
    app.after(0, app.deiconify)

def hide_window():
    app.withdraw()
    image = PILImage.open(icon_path)
    menu = pystray.Menu(
        item('Show', show_window),
        item('Quit', on_quit)
    )
    icon = pystray.Icon("DubSense", image, "DubSense", menu)
    icon.run_detached()

def minimize_to_tray(event=None):
    if app.state() == 'iconic':
        hide_window()

def on_close():
    on_quit()

app.protocol("WM_DELETE_WINDOW", on_close)
app.bind("<Unmap>", minimize_to_tray)
app.mainloop()