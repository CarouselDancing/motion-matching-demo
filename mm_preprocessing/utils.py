
from pathlib import Path
import quat
import bvh
import numpy as np
import librosa


def get_files(dir, suffix=".bvh"):
    files = []
    for f in Path(dir).iterdir():
        f = str(f)
        if  f.endswith(suffix):
            files.append(f)
    return files

def contains_element_in_list(name, str_list):
    print(str_list)
    return len([v for v in str_list if name in v]) > 0

""" Basic function for mirroring animation data with this particular skeleton structure """

def animation_mirror(lrot, lpos, names, parents, left_prefix = "Left", right_prefix= "Right"):
    #ignore root
    joints_mirror = np.array([0] + [(
        names.index(left_prefix+n[len(right_prefix):]) if n.startswith(right_prefix) else (
        names.index(right_prefix+n[len(left_prefix):]) if n.startswith(left_prefix) else 
        names.index(n))) for n in names[1:]])

    mirror_pos = np.array([-1, 1, 1])
    mirror_rot = np.array([[-1, -1, 1], [1, 1, -1], [1, 1, -1]])

    grot, gpos = quat.fk(lrot, lpos, parents)

    gpos_mirror = mirror_pos * gpos[:,joints_mirror]
    grot_mirror = quat.from_xform(mirror_rot * quat.to_xform(grot[:,joints_mirror]))
    
    return quat.ik(grot_mirror, gpos_mirror, parents)


def load_file(filename, scale):
    """ Load Data """   
    
    bvh_data = bvh.load(filename)
    bvh_data['positions'] = bvh_data['positions']
    bvh_data['rotations'] = bvh_data['rotations']
    
    positions = bvh_data['positions']
    rotations = quat.unroll(quat.from_euler(np.radians(bvh_data['rotations']), order=bvh_data['order']))

    
    positions *= scale 
    
    return positions, rotations,  bvh_data['names'], bvh_data["parents"]




def extract_audio_features(audio_file, n_frames, sampling_rate, n_mels):

    audio_data, sr = librosa.load(audio_file, sr=sampling_rate)
    hop_len = np.floor(len(audio_data) / n_frames)
    C = librosa.feature.melspectrogram(y=audio_data, sr=sr, n_fft=2048, 
                                    hop_length=int(hop_len), n_mels=n_mels, fmin=0.0, fmax=8000)
    
    return C.T[:n_frames]


from enum import IntEnum
class UHumanBodyBones(IntEnum):
    Hips = 0
    LeftUpperLeg = 1
    RightUpperLeg = 2
    LeftLowerLeg = 3
    RightLowerLeg = 4
    LeftFoot = 5
    RightFoot = 6
    Spine = 7
    Chest = 8
    Neck = 9
    Head = 10
    LeftShoulder = 11
    RightShoulder = 12
    LeftUpperArm = 13
    RightUpperArm = 14
    LeftLowerArm = 15
    RightLowerArm = 16
    LeftHand = 17
    RightHand = 18
    LeftToes = 19
    RightToes = 20
    LeftEye = 21
    RightEye = 22
    Jaw = 23
    LeftThumbProximal = 24
    LeftThumbIntermediate = 25
    LeftThumbDistal = 26
    LeftIndexProximal = 27
    LeftIndexIntermediate = 28
    LeftIndexDistal = 29
    LeftMiddleProximal = 30
    LeftMiddleIntermediate = 31
    LeftMiddleDistal = 32
    LeftRingProximal = 33
    LeftRingIntermediate = 34
    LeftRingDistal = 35
    LeftLittleProximal = 36
    LeftLittleIntermediate = 37
    LeftLittleDistal = 38
    RightThumbProximal = 39
    RightThumbIntermediate = 40
    RightThumbDistal = 41
    RightIndexProximal = 42
    RightIndexIntermediate = 43
    RightIndexDistal = 44
    RightMiddleProximal = 45
    RightMiddleIntermediate = 46
    RightMiddleDistal = 47
    RightRingProximal = 48
    RightRingIntermediate = 49
    RightRingDistal = 50
    RightLittleProximal = 51
    RightLittleIntermediate = 52
    RightLittleDistal = 53
    UpperChest = 54
    LastBone = 55