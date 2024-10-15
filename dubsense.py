import cv2
import time
import threading
import logging
import asyncio
import aiohttp
import numpy as np
from PIL import ImageGrab, Image, ImageTk
from screeninfo import get_monitors
import pytesseract
import tkinter as tk
from tkinter import scrolledtext, messagebox
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from psutil import Process, IDLE_PRIORITY_CLASS, process_iter
import json
import os
from PIL import Image 
from pathlib import Path
import re

# Global Constants and Configurations
class AppConfig:
    CONFIG_FILE = Path.home() / '.dubsense' / 'config.json'
    ICON_PATH = Path(os.path.dirname(__file__)) / 'dubsense.ico'
    CALL_OF_DUTY_PROCESS_NAME = "cod.exe"
    BASE_DIR = Path(os.path.expanduser("~")) / ".paddleocr" / "whl"
    
    default_config = {
        "search_word": "CTO",
        "check_interval": 1,
        "trigger_interval": 15,
        "box_height_percent": 0.22,
        "aspect_ratio": (7, 3),
        "webhook_enabled": True,
        "webhook_url": "http://...",
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
        # Tesseract OCR does not require initialization like PaddleOCR
        return pytesseract
    except Exception as e:
        messagebox.showerror("Initialization Error", f"Error initializing OCR: {e}")
        raise

ocr = initialize_ocr()
monitoring_active = threading.Event()

# Function to enable or disable widgets
def set_widgets_state(state):
    webhook_checkbox.config(state=state)
    webhook_entry.config(state=state)
    test_button.config(state=state)

# Function to enable or disable widgets
def set_widgets_state(state):
    webhook_checkbox.config(state=state)
    webhook_entry.config(state=state)
    test_button.config(state=state)

# GUI and Button Handlers
app = ttk.Window(themename="darkly")
app.title("DubSense")
app.iconbitmap(AppConfig.ICON_PATH)
app.resizable(False, False)
main_frame = ttk.Frame(app, padding=(15, 10))
main_frame.grid(row=0, column=0, sticky="nsew")

webhook_var = ttk.BooleanVar(value=config_manager.config.get("webhook_enabled", True))

# Define Test Webhook function
async def test_webhook():
    url = config_manager.config["webhook_url"]
    if not url:
        log_message("Webhook URL is empty. Please enter a valid URL.")
        return

    log_message("Testing webhook...")
    await send_webhook(url)

# Function to update and save the webhook URL
def update_webhook_url(event=None):
    new_url = webhook_entry.get().strip()
    if new_url:
        config_manager.config["webhook_url"] = new_url
        config_manager.save_config()

# Create Widgets
webhook_checkbox = ttk.Checkbutton(main_frame, text="Enable Webhook", variable=webhook_var, bootstyle="primary-round-toggle")
webhook_checkbox.grid(row=2, column=0, sticky="w", padx=5, pady=5)
webhook_entry = ttk.Entry(main_frame, width=40)
webhook_entry.insert(0, config_manager.config.get("webhook_url", ""))
webhook_entry.grid(row=2, column=2, padx=(0, 5), pady=5, sticky="w")

# Bind the webhook entry to save the URL when focus is lost (e.g., user presses Enter or clicks away)
webhook_entry.bind("<FocusOut>", update_webhook_url)

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
        
        time.sleep(config_manager.config["check_interval"])  # Use the check_interval from the configuration

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
        # Apply thresholding to binarize the image
    # Apply thresholding to binarize the image
    _, binary_image = cv2.threshold(gray_image, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Apply denoising to remove noise
    denoised_image = cv2.fastNlMeansDenoising(binary_image, h=30)

    # Further scale down by half
    final_width = int(denoised_image.shape[1] * 1)
    final_height = int(denoised_image.shape[0] * 1)
    processed_image = cv2.resize(gray_image, (final_width, final_height))

    latest_image = processed_image
    update_image_display(processed_image)
    return processed_image

def detect_text(image, search_word):
    # Track last detection time within the function scope to ensure proper cooldown
    if not hasattr(detect_text, "last_detection_time"):
        detect_text.last_detection_time = 0  # Initialize attribute if it doesn't exist

    current_time = time.time()
    try:
        custom_config = r'--oem 3 --psm 6'
        text = pytesseract.image_to_string(image, config=custom_config)
        #log_message(f"Detected text: {text}")  # Log the detected text
    except Exception as e:
        log_message(f"Error during OCR: {e}")
        return False

    # Check if results are empty
    if not text:
        #log_message("No text detected.")
        return False

    # Use regular expression to check for the exact match of the search word
    if re.search(r'\b' + re.escape(search_word) + r'\b', text):
        # Check cooldown based on trigger_interval
        if current_time - detect_text.last_detection_time >= config_manager.config["trigger_interval"]:
            log_message("Victory Detected!")  # Log only once per interval
            detect_text.last_detection_time = current_time  # Update last detection time
            
            # Send webhook if enabled and checkbox is checked
            if config_manager.config["webhook_enabled"] and webhook_var.get():
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

# Start Application
app.protocol("WM_DELETE_WINDOW", lambda: stop_monitoring() or app.quit())
app.mainloop()
