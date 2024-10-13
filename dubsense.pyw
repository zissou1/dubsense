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
import onnxruntime as ort
import json
import os
import pystray
from pystray import MenuItem as item
from PIL import Image as PILImage

# Set process priority to reduce CPU usage
Process().nice(IDLE_PRIORITY_CLASS)

# Global variable to store the latest processed image
latest_image = None
monitoring_active = False

# Config file path
CONFIG_FILE = os.path.join(os.path.dirname(__file__), 'config\config.json')

# Constants for OCR and Monitoring Configurations
default_config = {
    "search_word": "CTO",
    "check_interval": 2.5,
    "trigger_interval": 15,
    "box_height_percent": 0.22,
    "aspect_ratio": (7, 3),
    "webhook_enabled": True,
    "webhook_url": "http://10.0.1.47:8123/api/webhook/dubsense-ANjl3h7PSIZKDVdHuD4CNqU6",
    "auto_monitor_cod": True
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
    # Ensure the directory exists before saving the file
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

CALL_OF_DUTY_PROCESS_NAME = "cod.exe"  # Update to the actual Call of Duty process name

# Set logging configuration
logging.basicConfig(level=logging.ERROR)


# OCR Setup with exception handling
def initialize_ocr():
    try:
        execution_provider = get_onnx_execution_provider()
        return PaddleOCR(
            use_angle_cls=False,
            use_gpu=(execution_provider != 'CPUExecutionProvider'),
            lang='en',
            execution_providers=[execution_provider]
        )
    except Exception as e:
        messagebox.showerror("Initialization Error", f"Error initializing OCR: {e}")
        raise

def get_onnx_execution_provider():
    providers = ['CUDAExecutionProvider', 'ROCmExecutionProvider', 'DirectMLExecutionProvider']
    available = ort.get_available_providers()
    return next((p for p in providers if p in available), 'CPUExecutionProvider')

ocr = initialize_ocr()

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
    latest_image = processed_image  # Update the latest image
    update_image_display()  # Update the image in the GUI
    return processed_image

def update_image_display():
    if latest_image is not None:
        # Convert the processed image to an ImageTk format for Tkinter
        image_pil = Image.fromarray(latest_image)
        image_tk = ImageTk.PhotoImage(image_pil)

        # Update the label with the new image
        image_label.config(image=image_tk)
        image_label.image = image_tk  # Keep reference to avoid garbage collection

def check_cod_and_monitor():
    while True:
        if auto_start_var.get():
            cod_running = any(proc.info['name'] == CALL_OF_DUTY_PROCESS_NAME for proc in process_iter(['name']))
            if cod_running and not monitoring_active:
                start_monitoring()
            elif not cod_running and monitoring_active:
                stop_monitoring()
        time.sleep(5)  # Check every 5 seconds

def detect_text(image, search_word):
    results = ocr.ocr(image, cls=False)
    if not results or not results[0]:
        return False
    for line in results[0]:
        text = re.sub(r'[^A-Z]', '', line[1][0].upper())  # Filter and convert to uppercase letters
        if search_word in text.replace("Q", "O"):
            print(f"Detected word: {text}")  # Print the detected word to the console
            return True
    return False

# Enhanced async webhook function with detailed logging and retries
async def send_webhook(url, retries=3):
    attempt = 0
    while attempt < retries:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url) as response:
                    # Log successful webhook
                    if response.status == 200:
                        logging.info("Webhook sent successfully.")
                        log_message("Webhook sent successfully!")
                        return  # Exit if successful
                    else:
                        # Log specific failure details
                        error_content = await response.text()
                        logging.warning(f"Failed to send webhook. Status code: {response.status}")
                        logging.warning(f"Response content: {error_content}")
                        log_message(f"Failed to send webhook. Status code: {response.status}, Response: {error_content}")

        except aiohttp.ClientError as e:
            # Log detailed network-related error
            logging.error(f"Client error sending webhook: {e}")
            log_message(f"Client error sending webhook: {e}")
        except Exception as e:
            # Log any other unexpected errors
            logging.error(f"Unexpected error sending webhook: {e}")
            log_message(f"Unexpected error sending webhook: {e}")

        attempt += 1
        backoff_time = 2 ** attempt  # Exponential backoff
        log_message(f"Retrying webhook in {backoff_time} seconds... (Attempt {attempt}/{retries})")
        await asyncio.sleep(backoff_time)

    # If it fails after retries, log final failure
    log_message("Failed to send webhook after multiple attempts.")


# Test webhook function called by the Test button
def test_webhook():
    if CONFIG["webhook_url"]:
        log_message("Testing webhook...")
        asyncio.run(send_webhook(CONFIG["webhook_url"]))
    else:
        log_message("Webhook URL is empty. Please enter a valid URL.")

# Create the event loop globally so it can be reused
loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)

# Monitor Screen with async webhook and sleep adjustment
def monitor_screen():
    last_webhook_time = 0
    idle_counter = 0
    idle_max = 10  # Increase check interval after 10 consecutive idle checks

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
                        # Use the persistent event loop to send the webhook asynchronously
                        loop.run_until_complete(send_webhook(CONFIG["webhook_url"]))
                    last_webhook_time = current_time
            else:
                idle_counter += 1

            time.sleep(CONFIG["check_interval"] if idle_counter < idle_max else CONFIG["check_interval"] * 2)
        except Exception as e:
            log_message(f"Error in monitor: {e}")

def log_error(message):
    log_message(message)  # Log to GUI
    messagebox.showerror("Error", message)  # Show popup

# Start/Stop Monitoring Functions
def start_monitoring():
    global monitoring_active
    if not monitoring_active:  # Prevent multiple threads from starting
        monitoring_active = True
        log_message("Monitoring started...")
        toggle_widgets_state("disabled")  # Disable webhook-related widgets
        monitoring_thread = threading.Thread(target=monitor_screen, daemon=True)
        monitoring_thread.start()

def stop_monitoring():
    global monitoring_active
    if monitoring_active:  # Only stop if already monitoring
        monitoring_active = False
        log_message("Monitoring stopped.")
        toggle_widgets_state("normal")  # Re-enable webhook-related widgets

def toggle_widgets_state(state):
    # Toggle the state of webhook-related widgets between "disabled" and "normal"
    webhook_checkbox.config(state=state)
    test_button.config(state=state)
    webhook_entry.config(state=state)

# Function to toggle the Start/Stop buttons based on the checkbox state
def toggle_start_stop_buttons():
    if auto_start_var.get():  # If Auto Start is enabled, disable the buttons
        start_button.config(state="disabled")
        stop_button.config(state="disabled")
    else:  # Otherwise, enable the buttons
        start_button.config(state="normal")
        stop_button.config(state="normal")

# GUI and Logging
log_lock = threading.Lock()

def log_message(message):
    with log_lock:
        log_area.config(state=tk.NORMAL)
        log_area.insert(tk.END, f"{message}\n")
        log_area.see(tk.END)
        log_area.config(state=tk.DISABLED)

# GUI Setup
app = ttk.Window(themename="darkly")  # Using a dark theme
app.title("DubSense")

# Set the icon for the window (update with the path to your icon file)
icon_path = r"C:\Users\arnqv\iCloudDrive\Programming\dubsense\dubsense.ico"
app.iconbitmap(icon_path)

# Make the window non-resizable
app.resizable(False, False)

# Create a main frame with padding to hold all content
main_frame = ttk.Frame(app, padding=(15, 10))  # Add padding around the main content
main_frame.grid(row=0, column=0, sticky="nsew")

# Auto Start Monitoring with Call of Duty checkbox with additional internal padding
auto_start_var = ttk.BooleanVar(value=CONFIG.get("auto_monitor_cod", True))
auto_start_checkbox = ttk.Checkbutton(main_frame, text="Auto Monitor Call of Duty", variable=auto_start_var, bootstyle="primary-round-toggle", command=update_auto_monitor_cod)
auto_start_checkbox.grid(row=0, column=0, sticky="w", padx=5, pady=5)

# Enable Webhook checkbox with padding and binding to the update function
webhook_var = ttk.BooleanVar(value=CONFIG["webhook_enabled"])  # Set initial state
webhook_checkbox = ttk.Checkbutton(main_frame, text="Enable Webhook", variable=webhook_var, bootstyle="primary-round-toggle", command=update_webhook_enabled)
webhook_checkbox.grid(row=1, column=0, sticky="w", padx=5, pady=5)

# Test Webhook button
test_button = ttk.Button(main_frame, text="Test", command=test_webhook, bootstyle="light")
test_button.grid(row=1, column=1, padx=5, pady=5)

# Webhook URL entry
webhook_entry = ttk.Entry(main_frame, width=40)
webhook_entry.insert(0, CONFIG.get("webhook_url", ""))
webhook_entry.bind("<FocusOut>", lambda e: update_webhook_url())
webhook_entry.grid(row=1, column=2, padx=(0, 10), pady=5, sticky="w")

# Log area with increased width for better readability and padding
log_area = scrolledtext.ScrolledText(main_frame, width=80, height=10, state=DISABLED, background="#2e2e2e", foreground="white")
log_area.grid(row=2, column=0, columnspan=3, padx=5, pady=10)

# Image display area below the log area with padding
image_label = ttk.Label(main_frame)
image_label.grid(row=3, column=0, columnspan=3, padx=5, pady=10)

# Start and Stop Monitoring buttons with padding
start_button = ttk.Button(main_frame, text="Start Monitoring", command=start_monitoring, bootstyle="primary")
start_button.grid(row=4, column=0, padx=10, pady=10)
stop_button = ttk.Button(main_frame, text="Stop Monitoring", command=stop_monitoring, bootstyle="primary")
stop_button.grid(row=4, column=2, padx=10, pady=10)

# Call toggle_start_stop_buttons immediately to set the initial state of buttons
toggle_start_stop_buttons()

# Start background thread for Call of Duty process monitoring
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