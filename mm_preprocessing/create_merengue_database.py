import os
import argparse
import collections
from motion_database import MotionDatabase
from preprocessing_pipeline import PreprocessingPipeline, load_ignore_list
from utils import UHumanBodyBones


def get_merengue_settings():
    settings = dict()
    joint_map = collections.OrderedDict({"Hips": UHumanBodyBones.Hips,
                    "Spine" : UHumanBodyBones.Spine, 
                    "Spine1": UHumanBodyBones.Jaw,
                    "Spine2": UHumanBodyBones.Chest, 
                    "Spine3": UHumanBodyBones.UpperChest,
                   "Spine4": UHumanBodyBones.Jaw,
                    "Neck": UHumanBodyBones.Neck,
                     "Head": UHumanBodyBones.Head,
                      "LeftShoulder": UHumanBodyBones.LeftShoulder, 
                   "LeftArm": UHumanBodyBones.LeftUpperArm,
                    "LeftForeArm": UHumanBodyBones.LeftLowerArm, 
                    "LeftHand": UHumanBodyBones.LeftHand, 
                    "RightShoulder": UHumanBodyBones.RightUpperArm,
                     "RightArm": UHumanBodyBones.RightUpperArm,
                   "RightForeArm": UHumanBodyBones.RightLowerArm,
                   "RightHand": UHumanBodyBones.RightHand,
                   "LeftUpLeg": UHumanBodyBones.LeftUpperLeg,
                    "LeftLeg": UHumanBodyBones.LeftLowerLeg,
                     "LeftFoot": UHumanBodyBones.LeftFoot,
                    "LeftToeBase": UHumanBodyBones.LeftToes, 
                    "RightUpLeg": UHumanBodyBones.RightUpperLeg,
                     "RightLeg": UHumanBodyBones.RightLowerLeg, 
                     "RightFoot": UHumanBodyBones.RightFoot,
                      "RightToeBase": UHumanBodyBones.RightToes})
    
    settings["scale"] = 0.01
    settings["toes"] = ["RightToeBase", "LeftToeBase"]
    settings["joint_names"] = list(joint_map.keys())
    settings["bone_map"] = [UHumanBodyBones.LastBone]+list(joint_map.values())
    settings["sim_position_joint_name"] = "Spine3"
    settings["sim_rotation_joint_name"] = "Hips"
    settings["left_prefix"] = "Left"
    settings["right_prefix"] = "Right"
    return settings

def load_ignore_list(filename):
    ignore_list = []
    with open(filename, "rt") as in_file:
        ignore_list.append(in_file.readline())
    return ignore_list


def main(**kwargs):
    out_filename= kwargs["out_filename"]
    motion_path= kwargs["motion_path"]
    n_max_files = kwargs["n_max_files"]
    kwargs["ignore_list"] = list()
    if kwargs["ignore_list_filename"] is not None:
        kwargs["ignore_list"] = load_ignore_list(kwargs["ignore_list_filename"])
    kwargs.update(get_merengue_settings())
    pipeline = PreprocessingPipeline(**kwargs)
    if not kwargs["evaluate"]:
        db = pipeline.create_db(motion_path, n_max_files)
        db.write(out_filename)
        #db.print_shape()

    db = MotionDatabase()
    db.load(out_filename)
    db.print_shape()

    

if __name__ == "__main__":
    DATA_DIR = r"D:\Research\Carousel\data"
    motion_path = DATA_DIR +os.sep + "m11\\bvh_split" #"idle_motion" #
    ignore_list_filename = None#DATA_DIR +os.sep + r"ignore_list.txt"
    out_filename = "out" +os.sep + "database_merengue_full.bin.txt"
    parser = argparse.ArgumentParser(description="Create motion matching database")
    parser.add_argument("--motion_path", type=str,  default=motion_path)
    parser.add_argument("--ignore_list_filename", type=str, default=ignore_list_filename)
    parser.add_argument("--out_filename", type=str, default=out_filename)
    parser.add_argument('--evaluate', "-e", default=False, dest='evaluate', action='store_true')
    parser.add_argument('--n_max_files', type=int,  default=10, dest='n_max_files')
    args = parser.parse_args()
    main(**vars(args))
    



    
    