import sys
import os
sys.path.append(os.sep.join([".."])+os.sep)
import argparse
from motion_matching.mm_database import MMDatabase
from motion_matching.mm_database_binary_io import MMDatabaseBinaryIO
from motion_matching.preprocessing_pipeline import PreprocessingPipeline, load_ignore_list
from motion_matching.settings import SETTINGS, DEFAULT_FEATURES



def create_db(feature_descs=DEFAULT_FEATURES, **kwargs):
    out_filename= kwargs["out_filename"]
    motion_path= kwargs["motion_path"]
    n_max_files = kwargs["n_max_files"]
    skeleton_type = kwargs["skeleton_type"]
    file_format = kwargs["file_format"]
    convert_coordinate_system = kwargs["convert_cs"]
    kwargs["ignore_list"] = list()
    if kwargs["ignore_list_filename"] is not None:
        kwargs["ignore_list"] = load_ignore_list(kwargs["ignore_list_filename"])
    kwargs.update(SETTINGS[skeleton_type])
    pipeline = PreprocessingPipeline(**kwargs)
    db = pipeline.create_db(motion_path, n_max_files)
    db.concatenate_data()
    if feature_descs is not None:
        print("create features")
        db.calculate_features(feature_descs, convert_coordinate_system=convert_coordinate_system, normalize=False)
        db.calculate_neighbors(normalize=True)

    if file_format == "npy":
        db.write_to_numpy(out_filename, False)
    else:
        MMDatabaseBinaryIO.write(db, out_filename)

    load_db(**kwargs)


def load_db(**kwargs):
    file_format = kwargs["file_format"]
    out_filename= kwargs["out_filename"]
    if file_format == "npy":
        db = MMDatabase()
        db.load_from_numpy(out_filename)
    else:
        db = MMDatabaseBinaryIO.load(out_filename)
    db.print_shape()

if __name__ == "__main__":
    data_dir = r"D:\Research\Carousel\data"
    out_path = r"..\data"
    motion_path = data_dir + os.sep + r"m11\retarget\raw\combined"
    out_filename = out_path + os.sep + "database_merengue_raw20_gl_2.npz"
    skeleton_type = "raw"
    parser = argparse.ArgumentParser(description="Create motion matching database")
    parser.add_argument("--motion_path", type=str,  default=motion_path)
    parser.add_argument("--ignore_list_filename", type=str, default=None)
    parser.add_argument("--out_filename", type=str, default=out_filename)
    parser.add_argument('--evaluate', "-e", default=False, dest='evaluate', action='store_true')
    parser.add_argument('--n_max_files', type=int, default=120)
    parser.add_argument('--skeleton_type', type=str, default=skeleton_type)
    parser.add_argument('--file_format', type=str, default="npy")
    parser.add_argument('--convert_cs', type=bool, default=False)
    args = parser.parse_args()
    args = vars(args)
    create_db(**args)
    



    
    