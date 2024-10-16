import cv2
import time
import threading
import logging
import asyncio
import aiohttp
import numpy as np
from PIL import Image
from screeninfo import get_monitors
import pytesseract
import tkinter as tk
from tkinter import scrolledtext, messagebox, BooleanVar
import customtkinter as ctk
from psutil import Process, IDLE_PRIORITY_CLASS, process_iter
import json
import os
from pathlib import Path
import re
import pystray
import ctypes
import mss
import psutil


# Global Constants and Configurations
class AppConfig:
    CONFIG_FILE = Path.home() / '.dubsense' / 'config.json'
    ICON_PATH = Path(os.path.dirname(__file__)) / 'dubsense.ico'
    CALL_OF_DUTY_PROCESS_NAME = "cod.exe"
    BASE_DIR = Path(os.path.expanduser("~")) / ".paddleocr" / "whl"
    
    default_config = {
        "webhook_enabled": True,
        "webhook_url": "http://...",
        "auto_monitor_cod": True,
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
search_word = "CTO"
trigger_interval = 15
box_height_percent = 0.22
aspect_ratio = (7, 3)

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

def set_process_affinity():
    p = psutil.Process()
    p.cpu_affinity([0])  # Use only the first CPU core

# Call this function at the beginning of your program
set_process_affinity()

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
    auto_start_checkbox.configure(state=state)
    webhook_checkbox.configure(state=state)
    webhook_entry.configure(state=state)
    test_button.configure(state=state)

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

# Define update_auto_monitor function to save the checkbox state
def update_auto_monitor():
    config_manager.config["auto_monitor_cod"] = auto_start_var.get()
    config_manager.save_config()

# GUI and Button Handlers
ctk.set_appearance_mode("Dark")  # Set the appearance mode to 'Dark'
ctk.set_default_color_theme("dark-blue")  # You can choose a different theme

app = ctk.CTk()
app.title("DubSense")
app.iconbitmap(AppConfig.ICON_PATH)
app.resizable(False, False)

main_frame = ctk.CTkFrame(app)
main_frame.grid(row=0, column=0, sticky="nsew", padx=15, pady=10)

auto_start_var = BooleanVar(value=config_manager.config.get("auto_monitor_cod", True))
webhook_var = BooleanVar(value=config_manager.config.get("webhook_enabled", True))

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
        detect_text(processed_image, search_word)
        
        time.sleep(2.5)

# Create Widgets
auto_start_checkbox = ctk.CTkSwitch(main_frame, text="Auto Monitor Call of Duty", variable=auto_start_var, command=update_auto_monitor)
auto_start_checkbox.grid(row=0, column=0, sticky="w", padx=5, pady=5) 

webhook_checkbox = ctk.CTkSwitch(main_frame, text="Enable Webhook", variable=webhook_var)
webhook_checkbox.grid(row=2, column=0, sticky="w", padx=5, pady=5)

webhook_entry = ctk.CTkEntry(main_frame, width=300)
webhook_entry.insert(0, config_manager.config.get("webhook_url", ""))
webhook_entry.grid(row=2, column=2, padx=(0, 5), pady=5, sticky="w")

# Bind the webhook entry to save the URL when focus is lost (e.g., user presses Enter or clicks away)
webhook_entry.bind("<FocusOut>", update_webhook_url)

test_button = ctk.CTkButton(main_frame, text="Test", command=lambda: asyncio.run(test_webhook()))
test_button.grid(row=2, column=1, padx=5, pady=5, sticky="w")

log_area = scrolledtext.ScrolledText(main_frame, width=80, height=10, state=tk.DISABLED, background="#2e2e2e", foreground="white")
log_area.grid(row=3, column=0, columnspan=3, padx=5, pady=10)

image_label = ctk.CTkLabel(main_frame, text='')
image_label.grid(row=4, column=0, columnspan=3, padx=5, pady=10)

start_button = ctk.CTkButton(main_frame, text="Start Monitoring", command=start_monitoring)
start_button.grid(row=5, column=0, padx=10, pady=10)
stop_button = ctk.CTkButton(main_frame, text="Stop Monitoring", command=stop_monitoring)
stop_button.grid(row=5, column=2, padx=10, pady=10)

def log_message(message):
    log_area.config(state=tk.NORMAL)
    log_area.insert(tk.END, f"{message}\n")
    log_area.see(tk.END)
    log_area.config(state=tk.DISABLED)

def update_image_display(image):
    # Convert the processed image to a format suitable for CTkImage
    image_pil = Image.fromarray(image)
    image_ctk = ctk.CTkImage(image_pil, size=(150, (150 * (aspect_ratio[1]) / aspect_ratio[0])))
    image_label.configure(image=image_ctk)
    image_label.image = image_ctk  # Keep a reference to prevent garbage collection

# Function to enable or disable specific widgets based on monitoring state
def set_monitoring_widgets_state(state):
    webhook_checkbox.configure(state=state)
    webhook_entry.configure(state=state)
    test_button.configure(state=state)

def capture_screen():
    with mss.mss() as sct:
        monitor = get_monitors()[0]
        box_height = int(monitor.height * box_height_percent)
        box_width = int((box_height * aspect_ratio[0]) / aspect_ratio[1])

        offset_x = int(monitor.height * 0.06)
        bbox_x = monitor.x + (monitor.width - box_width) // 2 - offset_x
        bbox_y = monitor.y + (monitor.height - box_height) // 2
        bbox = {'left': bbox_x, 'top': bbox_y, 'width': box_width, 'height': box_height}

        sct_img = sct.grab(bbox)
        screen = np.array(sct_img)

        # First resize to width 150, keeping aspect ratio
        new_width = 150
        resize_factor = new_width / screen.shape[1]
        new_height = int(screen.shape[0] * resize_factor)
        resized_screen = cv2.resize(screen, (new_width, new_height))

        return resized_screen


def process_image(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    update_image_display(gray_image)
    return gray_image

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
        if current_time - detect_text.last_detection_time >= trigger_interval:
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

# Start the auto-monitoring thread
auto_monitor_thread = threading.Thread(target=auto_start_monitoring, daemon=True)
auto_monitor_thread.start()

# System Tray Integration
icon = None  # Global variable to hold the system tray icon

def show_window(tray_icon=None, item=None):
    app.after(0, app.deiconify)

def hide_window(tray_icon=None, item=None):
    app.withdraw()

def exit_app(tray_icon=None, item=None):
    stop_monitoring()
    icon.stop()
    app.quit()

def create_tray_icon():
    global icon
    image = Image.open(AppConfig.ICON_PATH)
    menu = pystray.Menu(
        pystray.MenuItem('Show', show_window, default=True),  # Default action on double-click
        pystray.MenuItem('Hide', hide_window),
        pystray.MenuItem('Exit', exit_app)
    )
    icon = pystray.Icon("DubSense", image, "DubSense", menu)
    icon.run_detached()

# Create the system tray icon at startup
create_tray_icon()

# Hide the window from the taskbar
def remove_from_taskbar():
    hwnd = ctypes.windll.user32.GetParent(app.winfo_id())
    GWL_EXSTYLE = -20
    WS_EX_APPWINDOW = 0x00040000
    ex_style = ctypes.windll.user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
    ex_style = ex_style & ~WS_EX_APPWINDOW  # Remove WS_EX_APPWINDOW style
    ctypes.windll.user32.SetWindowLongW(hwnd, GWL_EXSTYLE, ex_style)
    # Update the window's appearance
    SWP_NOSIZE = 0x0001
    SWP_NOMOVE = 0x0002
    SWP_NOZORDER = 0x0004
    SWP_FRAMECHANGED = 0x0020
    ctypes.windll.user32.SetWindowPos(hwnd, None, 0, 0, 0, 0,
        SWP_NOMOVE | SWP_NOSIZE | SWP_NOZORDER | SWP_FRAMECHANGED)

# Apply the change after the window is initialized
app.after(0, remove_from_taskbar)

# Start Application
app.protocol("WM_DELETE_WINDOW", hide_window)  # Hide window on close
#import cProfile
#cProfile.run('app.mainloop()', 'profiling_results.prof')
app.mainloop()
