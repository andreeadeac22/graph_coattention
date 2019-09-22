#!/bin/bash
#SBATCH --qos=unkillable                             # Ask for unkillable job
#SBATCH --cpus-per-task=2                     # Ask for 2 CPUs
#SBATCH --gres=gpu:titanx:1                          # Ask for 1 GPU
#SBATCH --mem=32G                             # Ask for 10 GB of RAM
#SBATCH --time=30:00:00                        # The job will run for 3 hours
#SBATCH -o /network/tmp1/deacandr/slurm-%j.out  # Write the log on tmp1

# 0. Set script variables
scratch=/network/tmp1/deacandr
project_root=/network/tmp1/deacandr/graph_coattention
arch=approx1
exp_id=1
experiment_name="graph-coattn-$arch-$exp_id"

# 1. Load your environment
module load python/3.6
# 1.1 Create & install virtualenv
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip
pip install --no-index -r requirements.txt


source tesa/bin/activate
pip freeze > requirements.txt
mkdir dependencies
cd dependencies
pip download -r requirements.txt
#  add at top of requirements.txt
-f ./dependencies/

# 2. Copy your dataset on the compute node
mkdir $SLURM_TMPDIR/cora
cp $scratch/graph-ode/data/cora/* $SLURM_TMPDIR/cora
mkdir $SLURM_TMPDIR/$experiment_name

# 3. Launch your job, tell it to save the model in $SLURM_TMPDIR
#    and look for the dataset into $SLURM_TMPDIR
python train.py QM9 data/qm9/dsgdb9nsd/

# 4. Copy whatever you want to save on $SCRATCH
mkdir -p $scratch/graph-coattn/$experiment_name
cp -r $SLURM_TMPDIR/$experiment_name/* $scratch/graph-coattn/$experiment_name/
rm -rf $SLURM_TMPDIR/env