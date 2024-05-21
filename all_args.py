import argparse
import yaml

def model_config(parser):
    model_yaml = 'configs/model_config.yaml'

    with open(model_yaml, 'r') as f:
        default_arg = yaml.safe_load(f)

    parser.set_defaults(**default_arg)
    model_config = parser.parse_args()
    f.close()
    
    return model_config


def run_shell_options():
    parser = argparse.ArgumentParser(description="the options of run.sh")

    # dataset related
    parser.add_argument('--data_dir',default="Datasets/", type=str, help='')
    parser.add_argument('--run_dataset',type=str, default="deng-reads-RawCount.csv", help=['\
                                    TabulaMuris_Liver_10X-RowCount.csv', 
                                    'TabulaMuris_Heart_10X-RowCount.csv', 
                                    'TabulaMuris_Marrow_10X-RowCount.csv', 
                                    'tasic-rpkms-RowCount.csv', 
                                    'yan-RowCount.csv', 
                                    'TabulaMuris_Trachea_FACS-RowCount.csv', 
                                    'TabulaMuris_Thymus_10X-RowCount.csv', 
                                    'TabulaMuris_Marrow_FACS-RowCount.csv', 
                                    'campbell-RawCount-new.csv', 
                                    'TabulaMuris_Spleen_10X-RawCount-new.csv', 
                                    'marques-RawCount-new.csv', 
                                    'TabulaMuris_Thymus_10X-RawCount-new.csv', 
                                    'TabulaMuris_Tongue_FACS-RawCount-new.csv', 
                                    'darmanis-RawCount-new.csv', 
                                    'TabulaMuris_Colon_FACS-RawCount-new.csv', 
                                    'manno_mouse-RawCount.csv', 
                                    'deng-reads-RawCount (1).csv', 
                                    'deng-reads-RawCount.csv', 
                                    'baron-mouse-RawCount.csv', 
                                    'manno_human-RawCount.csv'])
    
    # CNN model config related
    parser.add_argument('--monitor',default='val_loss', type=str, help='')
    parser.add_argument('--es_patience',default=10, type=int, help='EarlyStopping patience')
    parser.add_argument('--verbose',default=1, type=int, help='')
    parser.add_argument('--mode',default='auto', type=str, help='')
    parser.add_argument('--save_best_only',default=True, type=bool, help='')
    parser.add_argument('--factor',default=0.1, type=float, help='')
    parser.add_argument('--lr_patience',default=8, type=int, help='reduce_lr_loss patience')
    parser.add_argument('--lr',default=1e-4, type=float, help='')
    parser.add_argument('--batch_size',default=32, type=int, help='')
    parser.add_argument('--epochs',default=90, type=int, help='')
    parser.add_argument('--validation_split',default=0.2, type=float, help='')

    # GPU related
    parser.add_argument('--ngpu', default=1, type=int, help='')

    # stage related
    parser.add_argument('--start_stage', default=1, type=int, help='')
    parser.add_argument('--stop_stage', default=4, type=int, help='')

    # output dir related
    parser.add_argument('--out_dir',default="./Output", type=str, help='')

    # update model 
    all_args = model_config(parser)

    return all_args

