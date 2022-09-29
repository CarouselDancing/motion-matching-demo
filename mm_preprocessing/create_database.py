import os
import argparse
from motion_matching.mm_database import MMDatabase
from motion_matching.preprocessing_pipeline import PreprocessingPipeline, load_ignore_list
from motion_matching.settings import SETTINGS


def load_ignore_list(filename):
    ignore_list = []
    with open(filename, "rt") as in_file:
        ignore_list.append(in_file.readline())
    return ignore_list


def main(**kwargs):
    out_filename= kwargs["out_filename"]
    motion_path= kwargs["motion_path"]
    n_max_files = kwargs["n_max_files"]
    skeleton_type = kwargs["skeleton_type"]
    kwargs["ignore_list"] = list()
    if kwargs["ignore_list_filename"] is not None:
        kwargs["ignore_list"] = load_ignore_list(kwargs["ignore_list_filename"])
    kwargs.update(SETTINGS[skeleton_type])
    pipeline = PreprocessingPipeline(**kwargs)
    if not kwargs["evaluate"]:
        db = pipeline.create_db(motion_path, n_max_files)
        db.write(out_filename)
        #db.print_shape()

    db = MMDatabase()
    db.load(out_filename)
    db.print_shape()

    

if __name__ == "__main__":
    DATA_DIR = r"D:\Research\Carousel\data"
    motion_path = DATA_DIR + os.sep + r"m11\retarget\raw\combined"
    out_path = "D:\Research\Carousel\workspace\motion_matching_demo\mm_demo\Assets\Resources"
    out_filename = out_path + os.sep + "database_merengue_raw3.bin.txt"
    parser = argparse.ArgumentParser(description="Create motion matching database")
    parser.add_argument("--motion_path", type=str,  default=motion_path)
    parser.add_argument("--ignore_list_filename", type=str, default=None)
    parser.add_argument("--out_filename", type=str, default=out_filename)
    parser.add_argument('--evaluate', "-e", default=False, dest='evaluate', action='store_true')
    parser.add_argument('--n_max_files', type=int, default=20)
    parser.add_argument('--skeleton_type', type=str, default="raw")
    args = parser.parse_args()
    main(**vars(args))
    



    
    