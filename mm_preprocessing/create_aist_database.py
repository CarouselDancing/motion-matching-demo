import os
import argparse
from motion_database import MotionDatabase
from preprocessing_pipeline import PreprocessingPipeline, load_ignore_list
from utils import UHumanBodyBones



def main(**kwargs):
    out_filename= kwargs["out_filename"]
    motion_path= kwargs["motion_path"]
    n_max_files = kwargs["n_max_files"]
    kwargs["ignore_list"] = load_ignore_list(kwargs["ignore_list_filename"])
    kwargs.update(get_aist_settings())
    #print(kwargs["bone_map"], len(kwargs["bone_map"]))
    pipeline = PreprocessingPipeline(**kwargs)
    if not kwargs["evaluate"]:
        db = pipeline.create_db(motion_path, n_max_files)
        db.write(out_filename)
        #db.print_shape()

    db = MotionDatabase()
    db.load(out_filename)
    db.print_shape()

    
def get_aist_settings():
    settings = dict()
    joint_names = ["root", "lhip", "lknee", "lankle", "ltoes",
                   "rhip", "rknee", "rankle", "rtoes", 
                   "spine", "belly", "chest", "neck", "head",
                   "linshoulder","lshoulder","lelbow", "lwrist", "lhand",
                    "rinshoulder", "rshoulder", "relbow", "rwrist", "rhand"]
    bone_map = [UHumanBodyBones.LastBone,UHumanBodyBones.Hips,
    UHumanBodyBones.LeftUpperLeg, UHumanBodyBones.LeftLowerLeg, UHumanBodyBones.LeftFoot,  UHumanBodyBones.LeftToes,
    UHumanBodyBones.RightUpperLeg, UHumanBodyBones.RightLowerLeg, UHumanBodyBones.RightFoot,  UHumanBodyBones.RightToes,
    UHumanBodyBones.Spine, UHumanBodyBones.Chest, UHumanBodyBones.UpperChest, UHumanBodyBones.Neck, UHumanBodyBones.Head,
    UHumanBodyBones.LeftShoulder,UHumanBodyBones.LeftUpperArm, UHumanBodyBones.LeftLowerArm,UHumanBodyBones.LeftHand, UHumanBodyBones.LeftIndexDistal,
    UHumanBodyBones.RightShoulder,UHumanBodyBones.RightUpperArm, UHumanBodyBones.RightLowerArm,UHumanBodyBones.RightHand, UHumanBodyBones.RightIndexDistal
    ]
    settings["scale"] = 0.1
    settings["toes"] = ["rtoes", "ltoes"]
    settings["joint_names"] = joint_names
    settings["bone_map"] = bone_map
    settings["sim_position_joint_name"] = "chest"
    settings["sim_rotation_joint_name"] = "root"
    settings["left_prefix"] = "l"
    settings["right_prefix"] = "r"
    return settings

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
    



    
    