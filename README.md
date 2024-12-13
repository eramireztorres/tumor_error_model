# Tumor Growth Model Likelihood Comparison

This repository provides a framework to perform Maximum Likelihood Estimation (MLE) and Bayesian fitting on tumor growth models, allowing for comparison of different likelihood models. The code is designed to reproduce the results of a manuscript and includes functionalities to:

1. Perform Maximum Likelihood Estimation using various likelihood models.
2. Run Bayesian fitting with flexible parameter configurations.
3. Generate and save plots of the fitted models.

The repository is structured to ensure reproducibility and flexibility, with user-friendly CLI scripts for all major tasks.

### **Features**
- Supports multiple likelihood models and tumor growth models.
- Customizable fitting parameters via CLI arguments.
- Flexible options for generating and saving plots of results.
- Results saved in a structured directory for easy access.

### **Target Audience**
This repository is intended for reviewers and researchers interested in reproducing the results of the manuscript or applying the framework to new data.

### **Prerequisites**
- Python >= 3.8
- Basic familiarity with Python and CLI usage.

## Setup Instructions

Follow these steps to set up the environment and run the provided scripts:

### Step 1: Clone the Repository
Clone the repository to your local machine:
```bash
git clone https://github.com/eramireztorres/tumor-error-model.git
cd tumor-growth-model
```

### Step 2: Create a Virtual Environment 
It is recommended to use a virtual environment to manage dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies 
Install the required Python packages listed in requirements.txt:
```bash
pip install -r requirements.txt
```

### Notes 
Ensure your Python version is >= 3.8.

## Step-by-Step Usage for CLI Scripts

### Step 1: Run Maximum Likelihood Estimation (MLE)
The first step is to perform Maximum Likelihood Estimation (MLE) on the tumor growth data. Use the `run_mle.py` script for this.

#### Example Command:
```bash
python src/scripts/run_mle.py --file DEGT0.txt --individual 1 --lh_model LhNormal --maxiter 1000
```

#### Explanation of Arguments:
- `--file`: Specify the data file to use (e.g., `DEGT0.txt` or `DFGT0.txt`).
- `--individual`: Column index corresponding to the individual in the dataset.
- `--lh_model`: Likelihood model to use (e.g., `LhNormal`, `LhStudent`).
- `--maxiter`: Maximum number of iterations for the fitting process.

The results will be saved in the data/ directory with a name like:

```bash
mle_results_DEGT0_1_LhNormal.joblib
```

### Step 2: Run Bayesian Fitting
After obtaining the MLE results, run Bayesian fitting using the `run_bayesian.py` script.

#### Example Command:
```bash
python src/scripts/run_bayesian.py --file DEGT0.txt --individual 1 --lh_model LhNormal --maxiter 50000 --b_values 1 0.1 0.35 0.5 0.7 --n_temperatures 15 --ti_n 1
```

#### Explanation of Arguments:
- `--file`: Specify the data file to use (e.g., `DEGT0.txt` or `DFGT0.txt`).
- `--individual`: Column index corresponding to the individual in the dataset.
- `--lh_model`: Likelihood model to use (e.g., `LhNormal`, `LhStudent`).
- `--maxiter`: Maximum number of iterations for the Bayesian fitting process.
- `--b_values`: List of `b` values to explore during Bayesian fitting.
- `--n_temperatures`: Number of temperatures for Thermodynamic Integration (TI).
- `--ti_n`: Temperature exponent for TI.


The results will be saved in the data/ directory with a name like:

```bash
bayes_results_DEGT0_1_LhNormal_b=1.0.joblib
```

### Step 3: Generate and Save Plots

To visualize the Bayesian fitting results, use the `run_plot_bayesian.py` script. This script can operate in two modes:

#### Option 1: Specify Parameters
If the Bayesian results follow the naming convention, you can specify the parameters directly:
```bash
python src/scripts/run_plot_bayesian.py --file DEGT0 --individual 1 --lh_model LhNormal --b_value 1.0
```

#### Option 2: Provide Joblib Path
If the results file is located elsewhere or uses a custom name, specify the full path to the joblib file:
```bash
python src/scripts/run_plot_bayesian.py --joblib_path data/bayes_results_DEGT0_1_LhNormal_b=1.0.joblib
```

## License
Tumor Growth Model Likelihood Comparison is licensed under the MIT License.