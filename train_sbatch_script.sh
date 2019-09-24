#!/bin/bash
#SBATCH --qos=unkillable                             # Ask for unkillable job
#SBATCH --cpus-per-task=2                     # Ask for 2 CPUs
#SBATCH --gres=gpu:titanx:1                          # Ask for 1 GPU
#SBATCH --exclude=leto52
#SBATCH --mem=32G                             # Ask for 10 GB of RAM
#SBATCH --time=30:00:00                        # The job will run for 3 hours
#SBATCH -o /network/tmp1/deacandr/slurm-%j.out  # Write the log on tmp1

export HOME=`getent passwd deacandr | cut -d':' -f6`
source ~/.bashrc
export PYTHONUNBUFFERED=1
echo Running pwdon deacandr

# 0. Set script variables
scratch=/network/tmp1/deacandr/graph_coattention
project_root=/network/tmp1/deacandr/graph_coattention/
arch=randompair
exp_id=5
experiment_name="graph-coattn-$arch-$exp_id"

# 1. Load your environment
echo $PATH
module load cuda/10.0
echo $PATH
echo Checking gpu
nvidia-smi
# 1.1 Create & install conda

# 2. Copy your dataset on the compute node
#mkdir $SLURM_TMPDIR/data
#cp -r $scratch/data/ $SLURM_TMPDIR/data
#mkdir $SLURM_TMPDIR/$experiment_name

# 3. Launch your job, tell it to save the model in $SLURM_TMPDIR
#    and look for the dataset into $SLURM_TMPDIR
pwd
cd graph_coattention/
python train.py QM9 $scratch/data/qm9/dsgdb9nsd/ --d_hid 50 --d_readout 200 --qm9_pairing_repetitions 5 --patience 8 --batch_size 120 --memo $SLURM_TMPDIR/randompair-hid50-readout200-repetitions5-patience8-batch120

# 4. Copy whatever you want to save on $SCRATCH
mkdir -p $scratch/$experiment_name
ls -l $SLURM_TMPDIR
cp $SLURM_TMPDIR/* $scratch/$experiment_name/
