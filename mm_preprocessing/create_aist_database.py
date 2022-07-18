import os
import argparse
from create_database import main

if __name__ == "__main__":
    DATA_DIR = r"D:\Research\Carousel\workspace\rinu\variational-dance-motion-models\data"
    motion_path = DATA_DIR +os.sep +  "AIST_motion" #"idle_motion" #
    ignore_list_filename = DATA_DIR +os.sep + r"ignore_list.txt"
    out_filename = "data" +os.sep + "database_dance4.bin"
    parser = argparse.ArgumentParser(description="Create motion matching database")
    parser.add_argument("--motion_path", type=str,  default=motion_path)
    parser.add_argument("--ignore_list_filename", type=str, default=ignore_list_filename)
    parser.add_argument("--out_filename", type=str, default=out_filename)
    parser.add_argument('--evaluate', "-e", default=False, dest='evaluate', action='store_true')
    parser.add_argument('--n_max_files', type=int,  default=4)
    parser.add_argument('--skeleton_type', type=str,  default="raw")
    args = parser.parse_args()
    main(**vars(args))
    



    
    