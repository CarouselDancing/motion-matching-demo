import collections
from utils import UHumanBodyBones


def get_raw_settings():
    settings = dict()
    joint_map = collections.OrderedDict({"Hips": UHumanBodyBones.Hips,
    
                    "Spine" : UHumanBodyBones.Spine, 
                    "Spine1": UHumanBodyBones.UpperChest,
                    "Neck": UHumanBodyBones.Neck,
                     "Head": UHumanBodyBones.Head,
                      "LeftShoulder": UHumanBodyBones.LeftShoulder, 
                   "LeftArm": UHumanBodyBones.LeftUpperArm,
                    "LeftForeArm": UHumanBodyBones.LeftLowerArm, 
                    "LeftHand": UHumanBodyBones.LeftHand, 
                    "RightShoulder": UHumanBodyBones.RightShoulder,
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
                      "RightToeBase": UHumanBodyBones.RightToes,

                    })
    
    settings["scale"] = 0.01
    settings["toes"] = ["RightToeBase", "LeftToeBase"]
    settings["joint_names"] = list(joint_map.keys())
    settings["bone_map"] = [UHumanBodyBones.LastBone]+list(joint_map.values())
    settings["sim_position_joint_name"] = "Spine1"
    settings["sim_rotation_joint_name"] = "Hips"
    settings["left_prefix"] = "Left"
    settings["right_prefix"] = "Right"
    return settings

    
def get_captury_settings():
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
                    "RightShoulder": UHumanBodyBones.RightShoulder,
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


def get_cmu_settings():
    settings = dict()
    joint_map = collections.OrderedDict({"root": UHumanBodyBones.Hips,
    
                   "lfemur": UHumanBodyBones.LeftUpperLeg,
                    "ltibia": UHumanBodyBones.LeftLowerLeg,
                     "lfoot": UHumanBodyBones.LeftFoot,
                    "ltoes": UHumanBodyBones.LeftToes, 
                    "rfemur": UHumanBodyBones.RightUpperLeg,
                     "rtibia": UHumanBodyBones.RightLowerLeg, 
                     "rfoot": UHumanBodyBones.RightFoot,
                      "rtoes": UHumanBodyBones.RightToes,

                    "lowerback" : UHumanBodyBones.Spine, 
                    "upperback": UHumanBodyBones.Chest, 
                    "thorax": UHumanBodyBones.UpperChest,
                   "lowerneck": UHumanBodyBones.Jaw,
                    "upperneck": UHumanBodyBones.Neck,
                     "head": UHumanBodyBones.Head,
                      "lclavicle": UHumanBodyBones.LeftShoulder, 
                   "lhumerus": UHumanBodyBones.LeftUpperArm,
                    "lradius": UHumanBodyBones.LeftLowerArm, 
                    "lwrist": UHumanBodyBones.LeftHand, 
                    "lhand": UHumanBodyBones.Jaw, 
                    "lfingers": UHumanBodyBones.Jaw, 
                    "lthumb": UHumanBodyBones.Jaw, 
                    "rclavicle": UHumanBodyBones.RightShoulder,
                     "rhumerus": UHumanBodyBones.RightUpperArm,
                   "rradius": UHumanBodyBones.RightLowerArm,
                   "rwrist": UHumanBodyBones.RightHand,
                   
                    "rhand": UHumanBodyBones.Jaw, 
                    "rfingers": UHumanBodyBones.Jaw, 
                    "rthumb": UHumanBodyBones.Jaw, 
                    })
    
    settings["scale"] = 0.1
    settings["toes"] = ["rtoes", "ltoes"]
    settings["joint_names"] = list(joint_map.keys())
    settings["bone_map"] = [UHumanBodyBones.LastBone]+list(joint_map.values())
    settings["sim_position_joint_name"] = "thorax"
    settings["sim_rotation_joint_name"] = "root"
    settings["left_prefix"] = "l"
    settings["right_prefix"] = "r"
    return settings


SETTINGS = dict()
SETTINGS["raw"] = get_raw_settings()
SETTINGS["captury"] = get_captury_settings()
SETTINGS["aist"] = get_aist_settings()
SETTINGS["cmu"] = get_cmu_settings()