import subprocess
import sys
import os
import importlib

def install_packages(requirements_file="requirements.txt"):
    # Check if file exists
    print("Installing required packages....")
    if not os.path.exists(requirements_file):
        print(f"No {requirements_file} file found.")
        return

    with open(requirements_file) as f:
        packages = [line.strip() for line in f if line.strip()]

    for package in packages:
        try:
            importlib.import_module(package.split("==")[0].split(">=")[0])
        except ImportError:
            print(f"Installing missing package: {package}")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    print("Installation completed.")


