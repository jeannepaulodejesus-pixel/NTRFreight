import argparse
from utils.logger import get_logger

#Initialize logger
logger = get_logger('Argument Parser')

def get_args():
    parser = argparse.ArgumentParser(description="Convert NTR Freight tariffs to standard format")

    # Define arguments
    parser.add_argument("--in", dest="input_file", required=True, help="Path to input Excel file")
    parser.add_argument("--out", dest="output_file", required=True, help="Path to output Excel file")

    args = parser.parse_args()

    logger.info(f"Input file: {args.input_file}")
    logger.info(f"Output file: {args.output_file}")
    
    return args
