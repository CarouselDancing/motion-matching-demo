import os
import argparse
from motion_database import MotionDatabase
from preprocessing_pipeline import PreprocessingPipeline, get_aist_settings, load_ignore_list
from .settings import SETTINGS


def main(**kwargs):
    out_filename= kwargs["out_filename"]
    motion_path= kwargs["motion_path"]
    audio_path= kwargs["audio_path"]
    skeleton_type = kwargs["skeleton_type"]
    kwargs["ignore_list"] = load_ignore_list(kwargs["ignore_list_filename"])
    kwargs.update(SETTINGS[skeleton_type])
    pipeline = PreprocessingPipeline(**kwargs)
    if not kwargs["evaluate"]:
        db = pipeline.create_db_with_audio(motion_path, audio_path, 4)
        db.write(out_filename)
        #db.print_shape()

    db = MotionDatabase()
    db.load(out_filename)
    db.print_shape()

    

if __name__ == "__main__":
    DATA_DIR = r"D:\Research\Carousel\workspace\rinu\variational-dance-motion-models\data"
    motion_path = DATA_DIR +os.sep +  "AIST_motion"
    audio_path = DATA_DIR +os.sep + "AIST_music"
    ignore_list_filename = DATA_DIR +os.sep + r"ignore_list.txt"
    out_path = "D:\Research\Carousel\workspace\motion_matching_demo\mm_demo\Assets\Resources"
    out_filename = out_path +os.sep + "database_merengue.bin"
    parser = argparse.ArgumentParser(description="Create motion matching database")
    parser.add_argument("--motion_path", type=str,  default=motion_path)
    parser.add_argument("--audio_path", type=str, default=audio_path)
    parser.add_argument("--ignore_list_filename", type=str, default=ignore_list_filename)
    parser.add_argument("--n_mels", type=int, default=27)#96
    parser.add_argument("--sampling_rate",  type=float, default=16000)
    parser.add_argument("--out_filename", type=str, default=out_filename)
    parser.add_argument('--evaluate', "-e", default=False, dest='evaluate', action='store_true')
    args = parser.parse_args()
    main(**vars(args))
    



    
    