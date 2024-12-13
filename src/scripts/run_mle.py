import argparse
import numpy as np
from joblib import dump
import os
import sys

# Determine the absolute path to the current file
current_file = os.path.abspath(__file__)
# Get the directory containing the current file
current_dir = os.path.dirname(current_file)
# Navigate up two levels to reach the root directory
root_directory = os.path.dirname(os.path.dirname(current_dir))

# Add the root directory to the Python path
sys.path.insert(0, root_directory)

print(f'Current file: {current_file}')
print(f'Root directory: {root_directory}')


from src.models.tumor_models import GompertzI
from src.models.lh_models import (
    LhNormal, LhNormalProp, LhBenz, LhStudent, LhStudentProp
)
from src.mle.mle_fit import MlcFitterCreator
from src.models.lh_models import LhModelCreator


#%%

# Define available likelihood models
lh_models = {
    'LhNormal': LhNormal,
    'LhNormalProp': LhNormalProp,
    'LhBenz': LhBenz,
    'LhStudent': LhStudent,
    'LhStudentProp': LhStudentProp
}





def main():
    parser = argparse.ArgumentParser(description="Run Maximum Likelihood Estimation (MLE).")
    parser.add_argument("--file", type=str, choices=["DEGT0.txt", "DFGT0.txt"], default="DEGT0.txt",
                        help="Data file to process.")
    parser.add_argument("--individual", type=int, default=1, help="Individual column to process.")
    parser.add_argument("--lh_model", type=str, choices=list(lh_models.keys()), default="LhNormal",
                        help="Likelihood model to use.")
    parser.add_argument("--maxiter", type=int, default=1000, help="Maximum number of iterations for fitting.")
    
    parser.add_argument("--theta_limits", type=str,
                    help="Initial theta limits as a JSON string, e.g., '[[0,1],[95,105],[0.5,2],[1,10]]'")
    parser.add_argument("--theta_names", type=str,
                        help="Comma-separated list of theta names, e.g., 'param1,param2,param3,param4'")

    
    args = parser.parse_args()

    print(f"Processing file: {args.file}, Individual: {args.individual}, Likelihood model: {args.lh_model}")

    # Load data
    data_path = f"data/{args.file}"
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
    

    # Prepare initial parameters
    # init_theta = [[0, 1], [y[-1], y[-1] * 3], [y[0] - 0.3 * y[0], y[0] + 0.3 * y[0]], [95, 105]]
    
    # Retrieve defaults
    config = lh_model_config[args.lh_model]
    init_theta = config['theta_limits']
    # theta_names = config['theta_names']
    
    
    # Override defaults with CLI inputs if provided
    if args.theta_limits:
        import json
        init_theta = json.loads(args.theta_limits)
    # if args.theta_names:
    #     theta_names = args.theta_names.split(',')

    
    model_dict = {'model': model, 'data': y, 't': t, 'lh_estimator_class': lh_estimator_class}

    # Create likelihood estimator
    lh_creator = LhModelCreator()
    lh_estimator = lh_creator.lh_model_create(model_dict)

    # Fit using MLE
    fit_dict = {'init_theta': init_theta, 'LH_model': lh_estimator, 'maxiter': args.maxiter}
    mle_fitter_creator = MlcFitterCreator()
    mle_fitter = mle_fitter_creator.mlc_fit_create(fit_dict)

    # Perform fitting and calculate metrics
    mle_fitter.fit()
    mle_fitter.bic()
    mle_fitter.r2()
    mle_fitter.shapiro()

    # Print results
    print('MLE Results:')
    print('MLE Theta: ', mle_fitter.results['mle_theta'])
    print('BIC: ', mle_fitter.results['bic'])
    print('R2: ', mle_fitter.results['R2'])
    print('Shapiro-Wilk Test Results: ', mle_fitter.results['shapiro_results'])

   
    # Save results to a designated directory
    output_dir = os.path.join(root_directory, "data")
    os.makedirs(output_dir, exist_ok=True)  # Create the directory if it doesn't exist
    output_file = os.path.join(output_dir, f"mle_results_{args.file[:-4]}_{args.individual}_{args.lh_model}.joblib")
    dump(mle_fitter.results, output_file)
    print(f"Results saved to {output_file}")


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
#     parser = argparse.ArgumentParser(description="Run Maximum Likelihood Estimation (MLE).")
#     parser.add_argument("--file", type=str, choices=["DEGT0.txt", "DFGT0.txt"], default="DEGT0.txt",
#                         help="Data file to process.")
#     parser.add_argument("--individual", type=int, default=1, help="Individual column to process.")
#     parser.add_argument("--lh_model", type=str, choices=list(lh_models.keys()), default="LhNormal",
#                         help="Likelihood model to use.")
#     parser.add_argument("--maxiter", type=int, default=1000, help="Maximum number of iterations for fitting.")
#     args = parser.parse_args()

#     print(f"Processing file: {args.file}, Individual: {args.individual}, Likelihood model: {args.lh_model}")

#     # Load data
#     data_path = f"data/{args.file}"
#     dat = np.loadtxt(data_path)
#     t = dat[:, 0]
#     y = dat[:, args.individual]
#     t = np.array(t)
#     y = np.array(y)

#     # Initialize model and likelihood estimator
#     lh_estimator_class = lh_models[args.lh_model]
#     model = GompertzI

#     # Prepare initial parameters
#     init_theta = [[0, 1], [y[-1], y[-1] * 3], [y[0] - 0.3 * y[0], y[0] + 0.3 * y[0]], [95, 105]]
#     model_dict = {'model': model, 'data': y, 't': t, 'lh_estimator_class': lh_estimator_class}

#     # Create likelihood estimator
#     lh_creator = LhModelCreator()
#     lh_estimator = lh_creator.lh_model_create(model_dict)

#     # Fit using MLE
#     fit_dict = {'init_theta': init_theta, 'LH_model': lh_estimator, 'maxiter': args.maxiter}
#     mle_fitter_creator = MlcFitterCreator()
#     mle_fitter = mle_fitter_creator.mlc_fit_create(fit_dict)

#     # Perform fitting and calculate metrics
#     mle_fitter.fit()
#     mle_fitter.bic()
#     mle_fitter.r2()
#     mle_fitter.shapiro()

#     # Print results
#     print('MLE Results:')
#     print('MLE Theta: ', mle_fitter.results['mle_theta'])
#     print('BIC: ', mle_fitter.results['bic'])
#     print('R2: ', mle_fitter.results['R2'])
#     print('Shapiro-Wilk Test Results: ', mle_fitter.results['shapiro_results'])

   
#     # Save results to a designated directory
#     output_dir = os.path.join(root_directory, "data")
#     os.makedirs(output_dir, exist_ok=True)  # Create the directory if it doesn't exist
#     output_file = os.path.join(output_dir, f"mle_results_{args.file[:-4]}_{args.individual}_{args.lh_model}.joblib")
#     dump(mle_fitter.results, output_file)
#     print(f"Results saved to {output_file}")


# if __name__ == "__main__":
#     main()

