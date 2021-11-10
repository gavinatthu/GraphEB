import os
import argparse
from model import PBG
from utils import *





def main(args):
    if not os.path.exists(args.data_format):
        print('generate format data')
        df = dataformat(args)
    
    #stat(args)

    DATA_DIR = './tiny_PBG/data/'
    MODEL_DIR = './tiny_PBG/model/'

    model = PBG(DATA_DIR, MODEL_DIR, args.data_format)
    model.train()
    plot(DATA_DIR, MODEL_DIR)


    return None




if __name__ == "__main__":
    
    args = argparse.ArgumentParser(description='Decision Stump')
    args.add_argument('--data_in', default='D:/data/KMcall/tiny_data.xlsx', type=str,
                    help='path to the training input .xlsx file')
    args.add_argument('--data_format', default='D:/data/KMcall/tiny_format.tsv', type=str,
            help='path to the training input .xlsx file')     
    args.add_argument('--out_dir', default='D:/data/KMcall/tiny_PBG/', type=str,
                    help='path to the result data file')

    args = args.parse_args()
    main(args)
