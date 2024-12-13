import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as pl
from joblib import load

# Add the root directory to the Python path
current_file = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file)
root_directory = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, root_directory)

from src.utils.plot_utils import DramPlotter

def main():
    parser = argparse.ArgumentParser(description="Create and save a plot from Bayesian results.")
    parser.add_argument("--file", type=str, help="Base data file name (e.g., DEGT0) for the joblib.")
    parser.add_argument("--individual", type=int, help="Individual index in the data.")
    parser.add_argument("--lh_model", type=str, help="Likelihood model used (e.g., LhNormal).")
    parser.add_argument("--b_value", type=float, help="The b value used in the Bayesian fitting.")
    parser.add_argument("--joblib_path", type=str, help="Full path to the joblib file (overrides other parameters).")
    parser.add_argument("--output_dir", type=str, default="data", help="Directory to save the plot.")
    parser.add_argument("--title", type=str, default="Gompertz model", help="Title for the plot.")
    parser.add_argument("--ylabel", type=str, default="Tumor volume ($mm^3$)", help="Label for the Y-axis.")
    parser.add_argument("--xlabel", type=str, default="Days", help="Label for the X-axis.")
    args = parser.parse_args()

    # Determine joblib path
    if args.joblib_path:
        joblib_path = args.joblib_path
    else:
        if not args.file or args.individual is None or not args.lh_model or args.b_value is None:
            raise ValueError("If --joblib_path is not provided, you must specify --file, --individual, --lh_model, and --b_value.")
        joblib_filename = f"bayes_results_{args.file}_{args.individual}_{args.lh_model}_b={args.b_value}.joblib"
        joblib_path = os.path.join(root_directory, "data", joblib_filename)

    # Load Bayesian results
    if not os.path.exists(joblib_path):
        raise FileNotFoundError(f"Joblib file not found: {joblib_path}")
    print(f"Loading Bayesian results from: {joblib_path}")
    bayes_results = load(joblib_path)

    # Extract time (t) and data (y) from results
    t = bayes_results['t']
    y = bayes_results['data']

    # Create the plotter
    dram_plotter = DramPlotter(bayes_results)

    # Generate the plot
    fig, ax = pl.subplots()
    ax.plot(t, y, '*', label='Data')
    ax = dram_plotter.plot(ax, t, label='Fit', width=3)
    ax.legend(loc='upper left')
    ax.set_ylabel(args.ylabel)
    ax.set_xlabel(args.xlabel)
    ax.set_title(args.title)

    # Save the plot
    os.makedirs(args.output_dir, exist_ok=True)
    plot_filename = f"plot_{args.file}_{args.individual}_{args.lh_model}_b={args.b_value}.png"
    plot_path = os.path.join(args.output_dir, plot_filename)
    pl.savefig(plot_path, facecolor=None, bbox_inches='tight', dpi=1200)
    print(f"Plot saved to: {plot_path}")

if __name__ == "__main__":
    main()
