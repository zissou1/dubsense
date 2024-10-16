import subprocess
import os

# Define the project directory where dubsense.py is located
project_dir = os.path.dirname(os.path.abspath(__file__))
script_path = os.path.join(project_dir, 'dubsense.py')

# Define the PyInstaller command components
pyinstaller_command = [
    'pyinstaller',
    '--onefile',
    '--noconsole',
    #'--additional-hooks-dir=.',
    '--icon=dubsense.ico',
    '--add-binary',
    os.path.join(os.path.dirname(__file__), 'dubsense.ico') + ';.',
    #'--add-binary',
    #os.path.join(os.getenv('LOCALAPPDATA'), 'Programs', 'Python', 'Python310', 'Lib', 'site-packages', 'paddle', 'libs', 'mklml.dll') + ';.',
    script_path
]

# Run the PyInstaller command
try:
    subprocess.run(pyinstaller_command, check=True, cwd=project_dir)
    print("Build completed successfully.")
except subprocess.CalledProcessError as e:
    print(f"An error occurred during the build: {e}")