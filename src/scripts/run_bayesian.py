import argparse
import numpy as np
from joblib import load, dump
import os
import sys

# Add the project root to the Python path
current_file = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file)
root_directory = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, root_directory)

from src.models.tumor_models import GompertzI
from src.models.lh_models import (
    LhNormal, LhNormalProp, LhBenz, LhStudent, LhStudentProp
)
from src.bayesian.bayes_fit import DramFitter, bf_from_2_log_marginal_lihelihoods
from src.bayesian.bayes_assess import DramTiMarginalLikelihoodEstimator, TrajectoryPEstimator
from src.models.lh_models import LhModelCreator


# Add the project root to the Python path
current_file = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file)
root_directory = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, root_directory)


# Define available likelihood models
lh_models = {
    'LhNormal': LhNormal,
    'LhNormalProp': LhNormalProp,
    'LhBenz': LhBenz,
    'LhStudent': LhStudent,
    'LhStudentProp': LhStudentProp
}

def main():
    parser = argparse.ArgumentParser(description="Run Bayesian Fitting for Tumor Models.")
    parser.add_argument("--file", type=str, choices=["DEGT0.txt", "DFGT0.txt"], default="DEGT0.txt",
                        help="Data file to process.")
    parser.add_argument("--individual", type=int, default=1, help="Individual column to process.")
    parser.add_argument("--lh_model", type=str, choices=list(lh_models.keys()), default="LhNormal",
                        help="Likelihood model to use.")
    parser.add_argument("--maxiter", type=int, default=50000, help="Maximum number of iterations for Bayesian fitting.")
    parser.add_argument("--b_values", type=float, nargs="+", default=[1, 0.1, 0.35, 0.5, 0.7],
                        help="List of b values to use for Bayesian fitting.")
    parser.add_argument("--n_temperatures", type=int, default=15, help="Number of temperatures for TI.")
    parser.add_argument("--ti_n", type=int, default=1, help="Temperature exponent for TI.")
    parser.add_argument("--output_dir", type=str, default="data",
                        help="Directory to save and load intermediate results (default: data).")
    parser.add_argument("--theta_limits", type=str,
                    help="Initial theta limits as a JSON string, e.g., '[[0,1],[95,105],[0.5,2],[1,10]]'")
    parser.add_argument("--theta_names", type=str,
                        help="Comma-separated list of theta names, e.g., 'param1,param2,param3,param4'")

    args = parser.parse_args()

    # Ensure the output directory exists
    output_dir = os.path.join(root_directory, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Processing file: {args.file}, Individual: {args.individual}, Likelihood model: {args.lh_model}")
    print(f"b values: {args.b_values}")
    print(f"Output directory: {output_dir}")

    # Load data
    data_path = os.path.join(root_directory, "data", args.file)
    dat = np.loadtxt(data_path)
    t = dat[:, 0]
    y = dat[:, args.individual]
    t = np.array(t)
    y = np.array(y)
    
    # Initialize model and likelihood estimator
    lh_estimator_class = lh_models[args.lh_model]
    model = GompertzI
    
    # Add these hardcoded defaults
    lh_model_config = {
        'LhNormal': {
            'theta_limits': [[0, 1], [y[-1], y[-1] * 3], [y[0] - 0.3 * y[0], y[0] + 0.3 * y[0]], [95, 105]],
            'theta_names': ['r', 'K', 'V(0)', 'sigma']
        },
        'LhNormalProp': {
            'theta_limits': [[0, 1], [y[-1], y[-1] * 3], [y[0] - 0.3 * y[0], y[0] + 0.3 * y[0]], [0, 1]],
            'theta_names': ['r', 'K', 'V(0)', 'sigma_coeff']
        }
        # Add other models as needed
    }

    # Load MLE results
    mle_file_name = f"mle_results_{args.file[:-4]}_{args.individual}_{args.lh_model}.joblib"
    mle_results_path = os.path.join(output_dir, mle_file_name)
    if not os.path.exists(mle_results_path):
        raise FileNotFoundError(f"MLE results file not found: {mle_results_path}")
    mle_results = load(mle_results_path)
    mle_theta = mle_results['mle_theta']

    # Prepare parameters
    # theta_limits = [[0, 1], [y[-1], y[-1] * 3], [y[0] - 0.3 * y[0], y[0] + 0.3 * y[0]], [95, 105]]
    # theta_names = ['r', 'K', 'V(0)', 'sigma']
    
    
    # Retrieve defaults
    config = lh_model_config[args.lh_model]
    theta_limits = config['theta_limits']
    theta_names = config['theta_names']
    
    # Override defaults with CLI inputs if provided
    if args.theta_limits:
        import json
        theta_limits = json.loads(args.theta_limits)
    if args.theta_names:
        theta_names = args.theta_names.split(',')
        
    init_theta = mle_theta
    model_dict = {'model': model, 'data': y, 't': t, 'lh_estimator_class': lh_estimator_class}
    ml_dict = {'n_temperatures': args.n_temperatures, 'ti_N': args.ti_n}
    

    # Priors
    def flat_prior(theta):
        return 1
    prior_funs = [flat_prior, flat_prior, flat_prior, flat_prior]

    # Instances
    lh_creator = LhModelCreator()
    lh_estimator = lh_creator.lh_model_create(model_dict)
    trajectory_p_estimator_instance = TrajectoryPEstimator()

    log_marginal_likelihoods = {}

    for b in args.b_values:
        print(f"Starting Bayesian fitting with b = {b}")

        # Output file name
        bayes_file_name = f"bayes_results_{args.file[:-4]}_{args.individual}_{args.lh_model}_b={b}.joblib"
        bayes_results_path = os.path.join(output_dir, bayes_file_name)

        fit_dict = {
            'init_theta': init_theta,
            'LH_model': lh_estimator,
            'theta_names': theta_names,
            'maxiter': args.maxiter,
            'theta_limits': theta_limits,
            'prior': prior_funs,
            'b': b
        }

        dram_fitter = DramFitter(fit_dict)
        dram_ml_estimator = DramTiMarginalLikelihoodEstimator(ml_dict, dram_fitter)
        result_dict = dram_ml_estimator.get_ml()

        output = {
            'dram_results': dram_fitter.results,
            'ml_estimator.results': dram_ml_estimator.results,
            'data': y,
            't': t
        }

        log_marginal_likelihoods[b] = dram_ml_estimator.results['log_marginal_likelihood']

        if b == 1:
            dram_fitter.process()
            ps = trajectory_p_estimator_instance.get_all_ps(y, dram_fitter.results['all_trajectories'])
            output['ps'] = ps

        # Save results
        dump(output, bayes_results_path)
        print(f"Results saved to {bayes_results_path}")

    # Compute FBF for each b-value relative to b=1
    if 1 in log_marginal_likelihoods and len(log_marginal_likelihoods) > 1:
        print("\nFractional Bayes Factor (FBF) Estimates:")
        base_log_ml = log_marginal_likelihoods[1]
        for b, log_ml in log_marginal_likelihoods.items():
            if b != 1:
                fbf = bf_from_2_log_marginal_lihelihoods(base_log_ml, log_ml)
                print(f"FBF for b={b} relative to b=1: {fbf:.5f}")

if __name__ == "__main__":
    main()


# # Define available likelihood models
# lh_models = {
#     'LhNormal': LhNormal,
#     'LhNormalProp': LhNormalProp,
#     'LhBenz': LhBenz,
#     'LhStudent': LhStudent,
#     'LhStudentProp': LhStudentProp
# }

# def main():
#     parser = argparse.ArgumentParser(description="Run Bayesian Fitting for Tumor Models.")
#     parser.add_argument("--file", type=str, choices=["DEGT0.txt", "DFGT0.txt"], default="DEGT0.txt",
#                         help="Data file to process.")
#     parser.add_argument("--individual", type=int, default=1, help="Individual column to process.")
#     parser.add_argument("--lh_model", type=str, choices=list(lh_models.keys()), default="LhNormal",
#                         help="Likelihood model to use.")
#     parser.add_argument("--maxiter", type=int, default=50000, help="Maximum number of iterations for Bayesian fitting.")
#     parser.add_argument("--b_values", type=float, nargs="+", default=[1, 0.1, 0.35, 0.5, 0.7],
#                         help="List of b values to use for Bayesian fitting.")
#     parser.add_argument("--n_temperatures", type=int, default=15, help="Number of temperatures for TI.")
#     parser.add_argument("--ti_n", type=int, default=1, help="Temperature exponent for TI.")
#     args = parser.parse_args()

#     print(f"Processing file: {args.file}, Individual: {args.individual}, Likelihood model: {args.lh_model}")
#     print(f"b values: {args.b_values}")

#     # Load data
#     data_path = os.path.join(root_directory, "data", args.file)
#     dat = np.loadtxt(data_path)
#     t = dat[:, 0]
#     y = dat[:, args.individual]
#     t = np.array(t)
#     y = np.array(y)

#     # Initialize model and likelihood estimator
#     lh_estimator_class = lh_models[args.lh_model]
#     model = GompertzI

#     # Load MLE results
#     mle_file_name = f"mle_results_{args.file[:-4]}_{args.individual}_{args.lh_model}.joblib"
#     mle_results = load(os.path.join(root_directory, "data", mle_file_name))
#     mle_theta = mle_results['mle_theta']

#     # Prepare parameters
#     theta_limits = [[0, 1], [y[-1], y[-1] * 3], [y[0] - 0.3 * y[0], y[0] + 0.3 * y[0]], [95, 105]]
#     init_theta = mle_theta
#     model_dict = {'model': model, 'data': y, 't': t, 'lh_estimator_class': lh_estimator_class}
#     ml_dict = {'n_temperatures': args.n_temperatures, 'ti_N': args.ti_n}
#     theta_names = ['r', 'K', 'V(0)', 'sigma']

#     # Priors
#     def flat_prior(theta):
#         return 1
#     prior_funs = [flat_prior, flat_prior, flat_prior, flat_prior]

#     # Instances
#     lh_creator = LhModelCreator()
#     lh_estimator = lh_creator.lh_model_create(model_dict)
#     trajectory_p_estimator_instance = TrajectoryPEstimator()

#     for b in args.b_values:
#         print(f"Starting Bayesian fitting with b = {b}")

#         # Output file name
#         file_name = f"bayes_results_{args.file[:-4]}_{args.individual}_{args.lh_model}_b={b}.joblib"

#         fit_dict = {
#             'init_theta': init_theta,
#             'LH_model': lh_estimator,
#             'theta_names': theta_names,
#             'maxiter': args.maxiter,
#             'theta_limits': theta_limits,
#             'prior': prior_funs,
#             'b': b
#         }

#         dram_fitter = DramFitter(fit_dict)
#         dram_ml_estimator = DramTiMarginalLikelihoodEstimator(ml_dict, dram_fitter)
#         result_dict = dram_ml_estimator.get_ml()

#         output = {
#             'dram_results': dram_fitter.results,
#             'ml_estimator.results': dram_ml_estimator.results
#         }

#         if b == 1:
#             dram_fitter.process()
#             ps = trajectory_p_estimator_instance.get_all_ps(y, dram_fitter.results['all_trajectories'])
#             output['ps'] = ps

#         # Save results
#         dump(output, os.path.join(root_directory, "data", file_name))
#         print(f"Results saved to {file_name}")

# def main():
#     parser = argparse.ArgumentParser(description="Run Bayesian Fitting for Tumor Models.")
#     parser.add_argument("--file", type=str, choices=["DEGT0.txt", "DFGT0.txt"], default="DEGT0.txt",
#                         help="Data file to process.")
#     parser.add_argument("--individual", type=int, default=1, help="Individual column to process.")
#     parser.add_argument("--lh_model", type=str, choices=list(lh_models.keys()), default="LhNormal",
#                         help="Likelihood model to use.")
#     parser.add_argument("--maxiter", type=int, default=50000, help="Maximum number of iterations for Bayesian fitting.")
#     parser.add_argument("--b_values", type=float, nargs="+", default=[1, 0.1, 0.35, 0.5, 0.7],
#                         help="List of b values to use for Bayesian fitting.")
#     parser.add_argument("--n_temperatures", type=int, default=15, help="Number of temperatures for TI.")
#     parser.add_argument("--ti_n", type=int, default=1, help="Temperature exponent for TI.")
#     parser.add_argument("--output_dir", type=str, default="data",
#                         help="Directory to save and load intermediate results (default: data).")
#     args = parser.parse_args()

#     # Ensure the output directory exists
#     output_dir = os.path.join(root_directory, args.output_dir)
#     os.makedirs(output_dir, exist_ok=True)

#     print(f"Processing file: {args.file}, Individual: {args.individual}, Likelihood model: {args.lh_model}")
#     print(f"b values: {args.b_values}")
#     print(f"Output directory: {output_dir}")

#     # Load data
#     data_path = os.path.join(root_directory, "data", args.file)
#     dat = np.loadtxt(data_path)
#     t = dat[:, 0]
#     y = dat[:, args.individual]
#     t = np.array(t)
#     y = np.array(y)

#     # Load MLE results
#     mle_file_name = f"mle_results_{args.file[:-4]}_{args.individual}_{args.lh_model}.joblib"
#     mle_results_path = os.path.join(output_dir, mle_file_name)
#     if not os.path.exists(mle_results_path):
#         raise FileNotFoundError(f"MLE results file not found: {mle_results_path}")
#     mle_results = load(mle_results_path)
#     mle_theta = mle_results['mle_theta']

#     # Fit using Bayesian methods
#     for b in args.b_values:
#         print(f"Starting Bayesian fitting with b = {b}")

#         # Output file name
#         bayes_file_name = f"bayes_results_{args.file[:-4]}_{args.individual}_{args.lh_model}_b={b}.joblib"
#         bayes_results_path = os.path.join(output_dir, bayes_file_name)

#         # Save results
#         dump(output, bayes_results_path)
#         print(f"Results saved to {bayes_results_path}")

if __name__ == "__main__":
    main()


