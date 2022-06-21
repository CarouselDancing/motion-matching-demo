import os
import argparse
from motion_database import MotionDatabase
from preprocessing_pipeline import PreprocessingPipeline, get_settings, load_ignore_list



def main(**kwargs):
    out_filename= kwargs["out_filename"]
    motion_path= kwargs["motion_path"]
    n_max_files = kwargs["n_max_files"]
    kwargs["ignore_list"] = load_ignore_list(kwargs["ignore_list_filename"])
    kwargs.update(get_settings())
    #print(kwargs["bone_map"], len(kwargs["bone_map"]))
    pipeline = PreprocessingPipeline(**kwargs)
    if not kwargs["evaluate"]:
        db = pipeline.create_db(motion_path, n_max_files)
        db.write(out_filename)
        #db.print_shape()

    db = MotionDatabase()
    db.load(out_filename)
    db.print_shape()

    

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
    parser.add_argument('--n_max_files', type=int,  default=4, dest='n_max_files')
    args = parser.parse_args()
    main(**vars(args))
    



    
    