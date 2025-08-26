from utils.logger import get_logger

import subprocess
import sys
import os
import importlib

#Initialize logger
logger = get_logger('Package Installer')


def install_packages(requirements_file="requirements.txt"):
    # Check if file exists
    logger.info(f"Installing required packages.")
    if not os.path.exists(requirements_file):
        logger.critical(f"No {requirements_file} file found.")
        return

    with open(requirements_file) as f:
        packages = [line.strip() for line in f if line.strip()]

    for package in packages:
        try:
            importlib.import_module(package.split("==")[0].split(">=")[0])
        except ImportError:
            logger.info(f"Installing missing package: {package}")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    logger.info(f"Installation completed.")


