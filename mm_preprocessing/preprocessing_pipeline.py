
from pathlib import Path
import quat
from scipy.interpolate import griddata
import scipy.signal as signal
import scipy.ndimage as ndimage
import numpy as np
from motion_database import MotionDatabase
from utils import load_file, animation_mirror, extract_audio_features, contains_element_in_list, UHumanBodyBones


def swing_twist_decomposition(qs, twist_axis):
    """ code by janis sprenger based on
        Dobrowsolski 2015 Swing-twist decomposition in Clifford algebra. https://arxiv.org/abs/1506.05481
    """
    #twist_axis = np.array((q * offset))[0]
    swing_qs, twist_qs = np.zeros(qs.shape), np.zeros(qs.shape)
    print(qs.shape)
    for i, q in enumerate(qs):
        q = q[0]
        projection = np.dot(twist_axis, np.array([q[1], q[2], q[3]])) * twist_axis
        twist_q = np.array([q[0], projection[0], projection[1],projection[2]])
        if np.linalg.norm(twist_q) == 0:
            twist_q = np.array([1,0,0,0])
        twist_qs[i][0] = quat.normalize(twist_q)
        swing_qs[i][0] = quat.mul(q, quat.inv(twist_q))#q * quaternion_inverse(twist)
    return swing_qs, twist_qs

class PreprocessingPipeline:
    def __init__(self, **kwargs):
        self.joint_names = kwargs.get("joint_names", None)
        self.toes = kwargs.get("toes", [] )
        self.contact_velocity_threshold = kwargs.get("contact_velocity_threshold", 0.15)
        self.sim_position_joint_name = kwargs.get("sim_position_joint_name", "Spine2" )
        self.sim_rotation_joint_name = kwargs.get("sim_rotation_joint_name", "Hips" )
        self.scale = kwargs.get("scale", 1)
        self.left_prefix = kwargs.get("left_prefix", "Left")
        self.right_prefix = kwargs.get("right_prefix", "Right")
        self.grounding_offset = None
        self.ground_motion = kwargs.get("ground_motion", "True")
        self.sampling_rate = kwargs.get("sampling_rate", 1600)
        self.n_mels = kwargs.get("n_mels", 17)
        self.ignore_list = kwargs.get("ignore_list", [])
        self.bone_map = kwargs.get("bone_map", [])

    def process_motion(self, positions, rotations, names, parents, mirror):
        """ Supersample """
        
        nframes = positions.shape[0]
        nbones = positions.shape[1]
        
        # Supersample data to 60 fps
        original_times = np.linspace(0, nframes - 1, nframes)
        sample_times = np.linspace(0, nframes - 1, int(0.9 * (nframes * 2 - 1))) # Speed up data by 10%
        
        # This does a cubic interpolation of the data for supersampling and also speeding up by 10%
        positions = griddata(original_times, positions.reshape([nframes, -1]), sample_times, method='cubic').reshape([len(sample_times), nbones, 3])
        rotations = griddata(original_times, rotations.reshape([nframes, -1]), sample_times, method='cubic').reshape([len(sample_times), nbones, 4])
        
        # Need to re-normalize after super-sampling
        rotations = quat.normalize(rotations)
        
        """ Extract Simulation Bone """
        ref_dir = np.array([0, 0, 1])
        # First compute world space positions/rotations
        global_rotations, global_positions = quat.fk(rotations, positions, parents)
        
        # Specify joints to use for simulation bone 
        sim_position_joint = names.index(self.sim_position_joint_name)
        sim_rotation_joint = names.index(self.sim_rotation_joint_name)
        
        # Position comes from spine joint
        sim_position = np.array([1.0, 0.0, 1.0]) * global_positions[:,sim_position_joint:sim_position_joint+1]
        sim_position = signal.savgol_filter(sim_position, 31, 3, axis=0, mode='interp')
        
        # Direction comes from projected hip forward direction
        
        #sim_direction = np.array([1.0, 0.0, 1.0]) * quat.mul_vec(global_rotations[:,sim_rotation_joint:sim_rotation_joint+1], np.array([0.0, 1.0, 0.0]))
        _, twist_qs = swing_twist_decomposition(global_rotations[:,sim_rotation_joint:sim_rotation_joint+1],np.array([0.0, 1.0, 0.0]))
        
        sim_direction = quat.mul_vec(twist_qs, ref_dir)
  
        # We need to re-normalize the direction after both projection and smoothing
        sim_direction = sim_direction / np.sqrt(np.sum(np.square(sim_direction), axis=-1))[...,np.newaxis]
        sim_direction = signal.savgol_filter(sim_direction, 61, 3, axis=0, mode='interp')
        sim_direction = sim_direction / np.sqrt(np.sum(np.square(sim_direction), axis=-1)[...,np.newaxis])
            
        # initial frame to reference 
        initial_offset_rotation = quat.normalize(quat.between(sim_direction[0:1], ref_dir))
        #print(sim_direction[0:1].shape, initial_offset_rotation.shape)
        rotations[:,0:1] = quat.mul(initial_offset_rotation[0:1], rotations[:,0:1])
        #for i in range(len(rotations)):
        #    q = quat.mul(initial_offset_rotation[0:1], rotations[i,0:1])
        #    #print(initial_offset_rotation.shape, rotations[i,0:1].shape, q)
        #    rotations[i,0:1] = q
        print("before", sim_direction[0])
        sim_direction = quat.mul_vec(initial_offset_rotation, sim_direction)
        print("after", sim_direction[0])
        # Extract rotation from direction
        sim_rotation = quat.normalize(quat.between(ref_dir, sim_direction))
        print("sim_rotation", quat.to_euler(sim_rotation[0]))

        # Transform first joints to be local to sim and append sim as root bone
        
        print("before", sim_rotation_joint, sim_direction[0], quat.to_euler(sim_rotation[0]), quat.to_euler(rotations[0,0]))
        positions[:,0:1] = quat.mul_vec(quat.inv(sim_rotation), positions[:,0:1] - sim_position)
        rotations[:,0:1] = quat.mul(quat.inv(sim_rotation), rotations[:,0:1])
        if mirror:
            rotations[:,0:1] =quat.mul(quat.from_angle_axis(np.pi, [0,1,0]), rotations[:,0:1])

        print("after", quat.to_euler(rotations[0,0]))
        
        positions = np.concatenate([sim_position, positions], axis=1)
        rotations = np.concatenate([sim_rotation, rotations], axis=1)
        
        bone_parents = np.concatenate([[-1], parents + 1])
        
        bone_names = ['Simulation'] + names
        
        """ Compute Velocities """
        
        # Compute velocities via central difference
        velocities = np.empty_like(positions)
        velocities[1:-1] = (
            0.5 * (positions[2:  ] - positions[1:-1]) * 60.0 +
            0.5 * (positions[1:-1] - positions[ :-2]) * 60.0)
        velocities[ 0] = velocities[ 1] - (velocities[ 3] - velocities[ 2])
        velocities[-1] = velocities[-2] + (velocities[-2] - velocities[-3])
        
        # Same for angular velocities
        angular_velocities = np.zeros_like(positions)
        angular_velocities[1:-1] = (
            0.5 * quat.to_scaled_angle_axis(quat.abs(quat.mul_inv(rotations[2:  ], rotations[1:-1]))) * 60.0 +
            0.5 * quat.to_scaled_angle_axis(quat.abs(quat.mul_inv(rotations[1:-1], rotations[ :-2]))) * 60.0)
        angular_velocities[ 0] = angular_velocities[ 1] - (angular_velocities[ 3] - angular_velocities[ 2])
        angular_velocities[-1] = angular_velocities[-2] + (angular_velocities[-2] - angular_velocities[-3])

        """ Compute Contact Data """ 

        global_rotations, global_positions, global_velocities, global_angular_velocities = quat.fk_vel(
            rotations, 
            positions, 
            velocities,
            angular_velocities,
            bone_parents)
        
        contact_velocity = np.sqrt(np.sum(global_velocities[:,np.array([
            bone_names.index(self.toes[0]), 
            bone_names.index(self.toes[1])])]**2, axis=-1))
        
        # Contacts are given for when contact bones are below velocity threshold
        contacts = contact_velocity < self.contact_velocity_threshold
        
        # Median filter here acts as a kind of "majority vote", and removes
        # small regions  where contact is either active or inactive
        for ci in range(contacts.shape[1]):
        
            contacts[:,ci] = ndimage.median_filter(
                contacts[:,ci], 
                size=6, 
                mode='nearest')

        return positions, velocities, rotations, angular_velocities, bone_names, bone_parents, contacts

    def estimate_ground_offset(self, positions, rotations, parents):
        toe_idx = self.joint_names.index(self.toes[0])
        global_rotations, global_positions = quat.fk(rotations, positions, parents)
        mininum_y = np.min(global_positions[:, :, 1])
        return np.array([0.0,-mininum_y,0.0], dtype=positions.dtype)

    def process_motion_file(self, filename, mirror):
        positions, rotations, bone_names, bone_parents = load_file(filename, self.scale)
        #print(bone_names, len(self.joint_names))
        bone_names = self.joint_names
        self.grounding_offset = self.estimate_ground_offset(positions, rotations, bone_parents)
        if self.ground_motion:
            print(positions[:,0].shape, self.grounding_offset)
            positions[:,0] += self.grounding_offset
        if mirror:
            rotations, positions = animation_mirror(rotations, positions, bone_names, bone_parents, self.left_prefix, self.right_prefix)
            rotations = quat.unroll(rotations)
        positions, velocities, rotations, angular_velocities, bone_names, bone_parents, contacts = self.process_motion(positions, rotations, bone_names, bone_parents, mirror)
        return positions, velocities, rotations, angular_velocities, bone_names, bone_parents, contacts



    def is_valid_file(self, filename):
        return filename.endswith("bvh") and not contains_element_in_list(filename, self.ignore_list)


    def create_db(self, motion_path, n_max_files=-1):
        db = MotionDatabase()
        n_files = 0
        for filename in Path(motion_path).iterdir():
            filename = str(filename)
            if not self.is_valid_file(filename):
                continue
            for mirror in [False, True]:
                print('Loading "%s" %s...' % (filename, "(Mirrored)" if mirror else ""))
                positions, velocities, rotations, angular_velocities, bone_names, bone_parents, contacts = self.process_motion_file(filename, mirror)
                db.append(positions, velocities, rotations, angular_velocities, contacts)
            n_files+=1
            if n_files >= n_max_files and n_max_files > 0:
                break
        db.set_skeleton(bone_names, bone_parents, self.bone_map)
        return db

    def create_db_with_audio(self, motion_path, audio_path, n_max_files=-1):
        db = MotionDatabase()
        n_files = 0
        for filename in Path(motion_path).iterdir():
            filename = str(filename)
            if not self.is_valid_file(filename):
                continue
            for mirror in [False]:
                print('Loading "%s" %s...' % (filename, "(Mirrored)" if mirror else ""))
                positions, velocities, rotations, angular_velocities, bone_names, bone_parents, contacts = self.process_motion_file(filename, mirror)
                #audio_file = os.path.join(audio_path, os.path.split(filename)[-1].replace('.bvh', '.mp3'))
                #if not os.path.exists(audio_file):
                #    raise FileExistsError("Cannot find audio file",audio_file)
                #audio_data = extract_audio_features(audio_file, len(positions), self.sampling_rate, self.n_mels)
                #print("audio",audio_data.shape)
                audio_data = None
                db.append(positions, velocities, rotations, angular_velocities, contacts, audio_data)
            n_files+=1
            if n_files >= n_max_files and n_max_files > 0:
                break
        db.set_skeleton(bone_names, bone_parents, self.bone_map)
        return db



def get_settings():
    settings = dict()
    scale = 0.01 # Convert from cm to m 
    toes = ["LeftToeBase", "RightToeBase"]
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

def load_ignore_list(filename):
    ignore_list = []
    with open(filename, "rt") as in_file:
        ignore_list.append(in_file.readline())
    return ignore_list
