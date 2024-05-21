#!/bin/sh
############################ CHECK BEFORE RUN ##########################
# 1 --run_dataset is correct?
# 2 root dir settings in all_args.py are correct?
########################################################################

# train + inference
python3 main_raw.py --run_dataset TabulaMuris_Trachea_FACS-RowCount.csv
python3 main_cpm.py --run_dataset TabulaMuris_Trachea_FACS-RowCount.csv
