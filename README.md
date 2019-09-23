Skeleton:
- data_processing

1. python data_download.py QM9 DECAGON
2. python data_preprocess.py QM9 DECAGON
3. python split_cv_data.py QM9
   python split_cv_data.py DECAGON

-train

sbatch train_sbatch_script.sh

-test

sbatch test_sbatch_script.sh
