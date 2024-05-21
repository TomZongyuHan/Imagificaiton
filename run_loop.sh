#!/bin/sh
############################ CHECK BEFORE RUN ##########################
# 1 --run_dataset is correct?
# 2 root dir settings in all_args.py are correct?
########################################################################

python3 main_cpm.py --run_dataset TabulaMuris_Tongue_FACS-RawCount-new.csv
python3 main_cpm.py --run_dataset darmanis-RawCount-new.csv
python3 main_cpm.py --run_dataset TabulaMuris_Tongue_FACS-RawCount-new.csv
python3 main_cpm.py --run_dataset darmanis-RawCount-new.csv
python3 main_raw.py --run_dataset TabulaMuris_Heart_10X-RowCount.csv
python3 main_cpm.py --run_dataset TabulaMuris_Heart_10X-RowCount.csv
python3 main_raw.py --run_dataset TabulaMuris_Heart_10X-RowCount.csv
python3 main_cpm.py --run_dataset TabulaMuris_Heart_10X-RowCount.csv

# train + inference
# choose datasets to run
datasets=("TabulaMuris_Trachea_FACS-RowCount.csv" "TabulaMuris_Tongue_FACS-RawCount-new.csv" "darmanis-RawCount-new.csv" "TabulaMuris_Heart_10X-RowCount.csv")
# choose scripts to run
scripts=("main_raw.py" "main_cpm.py")

for i in {1..8}; do
    for dataset in "${datasets[@]}"; do
        for script in "${scripts[@]}"; do
            echo "Running: python3 $script --run_dataset $dataset"
            python3 $script --run_dataset $dataset
        done
    done
done
