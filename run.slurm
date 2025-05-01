#!/bin/bash
#------------------------------------------
# SLURM Job Script to Run Python Script
#------------------------------------------
#SBATCH --cluster chip-cpu
#SBATCH --job-name=TNCC-experiments
#SBATCH --output=tncc-%j.out
#SBATCH --error=tncc-%j.err
#SBATCH --time=01-00:00:00
#SBATCH --mem=16000
#SBATCH --cpus-per-task=4
#SBATCH --account=pi_rajangul
#SBATCH --qos=medium
#SBATCH --partition=general

#------------------------------------------
# Load Python module or Anaconda
#------------------------------------------
module load python/3.9  # adjust version as needed

#------------------------------------------
# Set up virtual environment
#------------------------------------------
echo "Creating virtual environment..."
python3 -m venv venv

echo "Activating virtual environment..."
source venv/bin/activate

#------------------------------------------
# Install required packages
#------------------------------------------
echo "Installing Python dependencies..."
pip install --upgrade pip
pip install numpy pandas scikit-learn matplotlib seaborn tqdm

#------------------------------------------
# Run your Python script
#------------------------------------------
echo "Running Python script..."
python dry_run.py  # replace with actual script name

#------------------------------------------
# Deactivate environment (optional cleanup)
#------------------------------------------
deactivate
echo "Job completed."
