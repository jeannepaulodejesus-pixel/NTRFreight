import argparse

def get_args():
    parser = argparse.ArgumentParser(description="Convert NTR Freight tariffs to standard format")

    # Define arguments
    parser.add_argument("--in", dest="input_file", required=True, help="Path to input Excel file")
    parser.add_argument("--out", dest="output_file", required=True, help="Path to output Excel file")

    args = parser.parse_args()

    print(f"Input file: {args.input_file}")
    print(f"Output file: {args.output_file}")
    
    return args
