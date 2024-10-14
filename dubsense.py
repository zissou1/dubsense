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
from PIL import Image as PILImage
import tarfile
from pathlib import Path

# Global Constants and Configurations
class AppConfig:
    CONFIG_FILE = Path(os.path.dirname(__file__)) / 'config/config.json'
    ICON_PATH = Path(os.path.dirname(__file__)) / 'dubsense.ico'
    CALL_OF_DUTY_PROCESS_NAME = "cod.exe"
    BASE_DIR = Path(os.path.expanduser("~")) / ".paddleocr" / "whl"
    
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

class ConfigManager:
    def __init__(self):
        self.config = AppConfig.default_config.copy()
        self.load_config()
    
    def load_config(self):
        if AppConfig.CONFIG_FILE.exists():
            with open(AppConfig.CONFIG_FILE, 'r') as f:
                self.config.update(json.load(f))
    
    def save_config(self):
        AppConfig.CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(AppConfig.CONFIG_FILE, 'w') as f:
            json.dump(self.config, f, indent=4)

# Configurations
config_manager = ConfigManager()

# Setup Logging 
def configure_logging(log_level=logging.INFO, log_format='%(asctime)s - %(levelname)s - %(message)s'):
    logging.basicConfig(level=log_level, format=log_format)
    logger = logging.getLogger()
    logger.handlers = []  # Clear any default handlers
    
    # Set up logging to GUI
    gui_handler = GuiLoggingHandler()
    gui_handler.setLevel(log_level)
    gui_handler.setFormatter(logging.Formatter(log_format))
    logger.addHandler(gui_handler)

# Setup Logging with direct GUI handler
class GuiLoggingHandler(logging.Handler):
    def emit(self, record):
        try:
            log_entry = self.format(record)
            log_area.config(state=tk.NORMAL)
            log_area.insert(tk.END, f"{log_entry}\n")
            log_area.see(tk.END)
            log_area.config(state=tk.DISABLED)
        except Exception as e:
            pass  # Avoid recursion if logging fails

configure_logging()

# Use IDLE priority to minimize CPU load
Process().nice(IDLE_PRIORITY_CLASS)

# OCR Setup
def initialize_ocr():
    try:
        return PaddleOCR(
            use_angle_cls=False,
            use_gpu=config_manager.config.get("use_gpu", False),
            lang='en'
        )
    except Exception as e:
        messagebox.showerror("Initialization Error", f"Error initializing OCR: {e}")
        raise

ocr = None
monitoring_active = threading.Event()

# Function to log download progress to log_area
def log_download_progress(message):
    log_area.config(state=tk.NORMAL)
    log_area.insert(tk.END, f"{message}\n")
    log_area.see(tk.END)
    log_area.config(state=tk.DISABLED)


# Enhanced Async download function with progress logging and error handling
async def download_file(url, file_path):
    try:
        timeout = aiohttp.ClientTimeout(total=300)  # Set a total timeout for the request
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url) as response:
                response.raise_for_status()  # Check if the response was successful
                file_path.parent.mkdir(parents=True, exist_ok=True)
                total_size = int(response.headers.get('content-length', 0))
                downloaded_size = 0

                with open(file_path, 'wb') as f:
                    async for chunk in response.content.iter_chunked(1024):
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded_size += len(chunk)
                        # Calculate and log download progress
                        if total_size > 0:
                            percent_complete = (downloaded_size / total_size) * 100
                            #log_download_progress(f"Downloading {file_path.name}: {percent_complete:.2f}% complete")

                log_download_progress(f"Download completed for {file_path.name}")

                # Unpack and delete the tar file after downloading
                unpack_tar_file(file_path, file_path.parent)
                os.remove(file_path)  # Clean up the .tar file
                #log_download_progress(f"Deleted {file_path.name}")

    except aiohttp.ClientError as e:
        log_download_progress(f"Failed to download {file_path.name} due to network error: {e}")
    except Exception as e:
        log_download_progress(f"An error occurred while downloading {file_path.name}: {e}")

def file_exists_and_valid(file_path, unpack_path):
    # Check if the unpacked directory exists (assuming unpacking creates this path)
    return unpack_path.exists() and unpack_path.is_dir()

# Enhanced Async download function with file existence check and unpacking
async def download_file_if_needed(url, file_path, unpack_path):
    if file_exists_and_valid(file_path, unpack_path):
        #log_download_progress(f"{unpack_path.name} already exists and is valid. Skipping download.")
        return

    await download_file(url, file_path)

# Function to enable or disable widgets
def set_widgets_state(state):
    auto_start_checkbox.config(state=state)
    gpu_checkbox.config(state=state)
    webhook_checkbox.config(state=state)
    webhook_entry.config(state=state)
    test_button.config(state=state)

# Enhanced Async download function with file existence check and conditional logging
async def download_and_unpack_async():
    # Disable widgets at the start of download
    set_widgets_state("disabled")
    
    # Define the unpacked directory for each downloaded file
    files = {
        (AppConfig.BASE_DIR / "det" / "en" / "en_PP-OCRv3_det_infer"): "https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_det_infer.tar",
        (AppConfig.BASE_DIR / "rec" / "en" / "en_PP-OCRv4_rec_infer"): "https://paddleocr.bj.bcebos.com/PP-OCRv4/english/en_PP-OCRv4_rec_infer.tar",
        (AppConfig.BASE_DIR / "cls" / "ch_ppocr_mobile_v2.0_cls_infer"): "https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar"
    }
    
    downloads_needed = False

    # Prepare the download tasks only if a file is missing or invalid
    tasks = []
    for file_path, url in files.items():
        if not file_exists_and_valid(file_path, file_path):
            downloads_needed = True
            tasks.append(download_file_if_needed(url, file_path.with_suffix('.tar'), file_path))

    # Log and start downloads only if needed
    if downloads_needed:
        log_download_progress("Starting file downloads, please wait...")
    
    if tasks:
        await asyncio.gather(*tasks)
        log_download_progress("All required downloads and unpacking completed.")
    else:
        log_download_progress("All models are downloaded and valid.")

    # Enable widgets after the download is complete
    set_widgets_state("normal")


def unpack_tar_file(tar_path, extract_path):
    try:
        with tarfile.open(tar_path, "r") as tar:
            tar.extractall(path=extract_path)
        #log_download_progress(f"Unpacked {tar_path.name}")
    except tarfile.TarError as e:
        log_download_progress(f"Failed to unpack {tar_path.name}: {e}")

# Start or stop auto-monitoring based on the checkbox setting
def is_cod_running():
    for process in process_iter(['name']):
        if process.info['name'] == AppConfig.CALL_OF_DUTY_PROCESS_NAME:
            return True
    return False

# Monitor Call of Duty process and automatically start/stop monitoring
def auto_start_monitoring():
    while True:
        if auto_start_var.get():  # Check if the checkbox is enabled
            cod_running = is_cod_running()
            if cod_running and not monitoring_active.is_set():
                start_monitoring()
            elif not cod_running and monitoring_active.is_set():
                stop_monitoring()
        time.sleep(5)  # Check every 5 seconds

# Function to enable or disable widgets
def set_widgets_state(state):
    auto_start_checkbox.config(state=state)
    gpu_checkbox.config(state=state)
    webhook_checkbox.config(state=state)
    webhook_entry.config(state=state)
    test_button.config(state=state)

# Define update_auto_monitor function to save the checkbox state
def update_auto_monitor():
    config_manager.config["auto_monitor_cod"] = auto_start_var.get()
    config_manager.save_config()

# GUI and Button Handlers
app = ttk.Window(themename="darkly")
app.title("DubSense")
app.iconbitmap(AppConfig.ICON_PATH)
app.resizable(False, False)
main_frame = ttk.Frame(app, padding=(15, 10))
main_frame.grid(row=0, column=0, sticky="nsew")

auto_start_var = ttk.BooleanVar(value=config_manager.config.get("auto_monitor_cod", True))
use_gpu_var = ttk.BooleanVar(value=config_manager.config.get("use_gpu", False))
webhook_var = ttk.BooleanVar(value=config_manager.config.get("webhook_enabled", True))

def update_gpu_usage():
    # Update configuration
    config_manager.config["use_gpu"] = use_gpu_var.get()
    config_manager.save_config()

    # Set environment variable to enforce GPU or CPU usage
    if config_manager.config["use_gpu"]:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Enables GPU (or specify a device ID)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disables all GPUs, forcing CPU

    # Reinitialize OCR with the new GPU setting
    global ocr
    ocr = initialize_ocr()
    log_message(f"Using {'GPU' if config_manager.config['use_gpu'] else 'CPU'}.")

# Define Test Webhook function
async def test_webhook():
    url = config_manager.config["webhook_url"]
    if not url:
        log_message("Webhook URL is empty. Please enter a valid URL.")
        return

    log_message("Testing webhook...")
    await send_webhook(url)

# Create Widgets
auto_start_checkbox = ttk.Checkbutton(main_frame, text="Auto Monitor Call of Duty", variable=auto_start_var, bootstyle="primary-round-toggle", command=update_auto_monitor)
auto_start_checkbox.grid(row=0, column=0, sticky="w", padx=5, pady=5) 
gpu_checkbox = ttk.Checkbutton(main_frame, text="Use GPU", variable=use_gpu_var, bootstyle="primary-round-toggle",command=update_gpu_usage)  # Ensure GPU setting is applied on change
gpu_checkbox.grid(row=1, column=0, sticky="w", padx=5, pady=5)
webhook_checkbox = ttk.Checkbutton(main_frame, text="Enable Webhook", variable=webhook_var, bootstyle="primary-round-toggle")
webhook_checkbox.grid(row=2, column=0, sticky="w", padx=5, pady=5)
webhook_entry = ttk.Entry(main_frame, width=40)
webhook_entry.insert(0, config_manager.config.get("webhook_url", ""))
webhook_entry.grid(row=2, column=2, padx=(0, 5), pady=5, sticky="w")
test_button = ttk.Button(main_frame, text="Test", command=lambda: asyncio.run(test_webhook()), bootstyle="light")
test_button.grid(row=2, column=1, padx=5, pady=5, sticky="w")

log_area = scrolledtext.ScrolledText(main_frame, width=80, height=10, state=DISABLED, background="#2e2e2e", foreground="white")
log_area.grid(row=3, column=0, columnspan=3, padx=5, pady=10)

image_label = ttk.Label(main_frame)
image_label.grid(row=4, column=0, columnspan=3, padx=5, pady=10)

start_button = ttk.Button(main_frame, text="Start Monitoring", command=lambda: start_monitoring(), bootstyle="primary")
start_button.grid(row=5, column=0, padx=10, pady=10)
stop_button = ttk.Button(main_frame, text="Stop Monitoring", command=lambda: stop_monitoring(), bootstyle="primary")
stop_button.grid(row=5, column=2, padx=10, pady=10)

def log_message(message):
    log_area.config(state=tk.NORMAL)
    log_area.insert(tk.END, f"{message}\n")
    log_area.see(tk.END)
    log_area.config(state=tk.DISABLED)

def update_image_display(image):
    # Convert the processed image to a format suitable for Tkinter display
    image_pil = Image.fromarray(image)
    image_tk = ImageTk.PhotoImage(image_pil)
    image_label.config(image=image_tk)
    image_label.image = image_tk

# Function to enable or disable specific widgets based on monitoring state
def set_monitoring_widgets_state(state):
    webhook_checkbox.config(state=state)
    webhook_entry.config(state=state)
    gpu_checkbox.config(state=state)
    test_button.config(state=state)

# Screen Monitoring Functions
def start_monitoring():
    global ocr
    if not monitoring_active.is_set():
        monitoring_active.set()
        ocr = initialize_ocr()
        log_message("Monitoring started...")
        set_monitoring_widgets_state("disabled")  # Disable widgets when monitoring starts
        threading.Thread(target=monitor_screen, daemon=True).start()

def stop_monitoring():
    if monitoring_active.is_set():
        monitoring_active.clear()
        log_message("Monitoring stopped.")
        set_monitoring_widgets_state("normal")  # Re-enable widgets when monitoring stops

def monitor_screen():
    while monitoring_active.is_set():
        screen = capture_screen()
        processed_image = process_image(screen)
        update_image_display(processed_image)  # Display the processed image
        
        # Detect text and handle logging/webhook within detect_text to respect trigger interval
        detect_text(processed_image, config_manager.config["search_word"])
        
        time.sleep(config_manager.config["check_interval"])

# OCR and Image Processing
def capture_screen():
    monitor = get_monitors()[0]
    box_height = int(monitor.height * config_manager.config["box_height_percent"])
    box_width = int((box_height * config_manager.config["aspect_ratio"][0]) / config_manager.config["aspect_ratio"][1])

    offset_x = int(monitor.height * 0.06)
    bbox_x = monitor.x + (monitor.width - box_width) // 2 - offset_x
    bbox_y = monitor.y + (monitor.height - box_height) // 2
    bbox = (bbox_x, bbox_y, bbox_x + box_width, bbox_y + box_height)

    screen = np.array(ImageGrab.grab(bbox=bbox))

    # First resize to width 300, keeping aspect ratio
    new_width = 150
    resize_factor = new_width / screen.shape[1]
    new_height = int(screen.shape[0] * resize_factor)
    resized_screen = cv2.resize(screen, (new_width, new_height))

    return resized_screen


def calculate_bbox(monitor):
    box_height = int(monitor.height * config_manager.config["box_height_percent"])
    box_width = int(box_height * config_manager.config["aspect_ratio"][0] / config_manager.config["aspect_ratio"][1])
    offset_x = int(monitor.height * 0.06)
    bbox_x = monitor.x + (monitor.width - box_width) // 2 - offset_x
    bbox_y = monitor.y + (monitor.height - box_height) // 2
    return (bbox_x, bbox_y, bbox_x + box_width, bbox_y + box_height)

def process_image(image):
    global latest_image
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Further scale down by half
    final_width = int(gray_image.shape[1] * 1)
    final_height = int(gray_image.shape[0] * 1)
    processed_image = cv2.resize(gray_image, (final_width, final_height))

    latest_image = processed_image
    update_image_display(processed_image)
    return processed_image

def detect_text(image, search_word):
    # Track last detection time within the function scope to ensure proper cooldown
    if not hasattr(detect_text, "last_detection_time"):
        detect_text.last_detection_time = 0  # Initialize attribute if it doesn't exist

    current_time = time.time()
    results = ocr.ocr(image, cls=False)

    # Check if results are empty
    if not results or not results[0]:
        #log_message("No text detected.")
        return False

    for line in results[0]:
        text = re.sub(r'[^A-Z]', '', line[1][0].upper())
        if search_word in text.replace("Q", "O"):
            # Check cooldown based on trigger_interval
            if current_time - detect_text.last_detection_time >= config_manager.config["trigger_interval"]:
                log_message("Victory Detected!")  # Log only once per interval
                detect_text.last_detection_time = current_time  # Update last detection time
                
                # Send webhook if enabled
                if config_manager.config["webhook_enabled"]:
                    asyncio.run(send_webhook(config_manager.config["webhook_url"]))
            return True

    return False


# Webhook Sending
async def send_webhook(url, retries=3):
    async with aiohttp.ClientSession() as session:
        for attempt in range(retries):
            try:
                async with session.post(url) as response:
                    if response.status == 200:
                        log_message("Webhook sent successfully!")
                        return
                    else:
                        await asyncio.sleep(2 ** attempt)
            except aiohttp.ClientError:
                await asyncio.sleep(2 ** attempt)
        log_message("Failed to send webhook after multiple attempts.")

# Function to start the download asynchronously in a separate thread
def start_download():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(download_and_unpack_async())
    loop.close()

# Start the download in a new thread as soon as the program starts
download_thread = threading.Thread(target=start_download, daemon=True)
download_thread.start()

# Start the auto-monitoring thread
auto_monitor_thread = threading.Thread(target=auto_start_monitoring, daemon=True)
auto_monitor_thread.start()

# Start Application
app.protocol("WM_DELETE_WINDOW", lambda: stop_monitoring() or app.quit())
app.mainloop()
