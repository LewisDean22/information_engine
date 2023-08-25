#!/bin/bash
#SBATCH --time=04:00:00
#SBATCH --account=def-dsivak
#SBATCH --output=job_output.out
#SBATCH --cpus-per-task=20
#SBATCH --mem-per-cpu=16G

#If running a script with joblib imported, module load quast/5.0.2
#I believe this breaks scipy.stats import in power_calculator

module purge && module load gcc/9.3.0 && module save
module load python/3.11.2
source virtual_environment/bin/activate
module load scipy-stack/2023a
module load StdEnv/2020
python setup.py build_ext --inplace
python correlation_time_analysis.py
python gaussian_bath_low_power_limit.py
