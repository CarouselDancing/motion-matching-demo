
from pathlib import Path
import os
from . import quat
import json
from scipy.interpolate import griddata
import scipy.signal as signal
import scipy.ndimage as ndimage
import numpy as np
from .mm_database import MMDatabase
from .utils import load_file, animation_mirror, extract_audio_features, contains_element_in_list
from scipy.signal import savgol_filter


import matplotlib.pyplot as plt

class AnnotationMatrixBuilder:
    """ build keys x frames matrix
    """
    keys = []
    values = []
    db_annotation_matrix = None

    def __init__(self, annotation_dict):
        self.db_annotation_matrix = None
        self.init_keys_and_values(annotation_dict)

    def init_keys_and_values(self, annotation_dict):
        _keys = []
        _values = ["ignore"] # values written in the matrix
        for f in annotation_dict:
            for k in annotation_dict[f]["annotations"]:
                if k not in _keys:
                    _keys.append(k)
                for v in annotation_dict[f]["annotations"][k]:
                    if v not in _values:
                        _values.append(v)
        self.values = list(_values)
        self.keys = list(_keys)

    def build_clip_annotation_matrix(self, clip_annotation, n_frames):

        clip_annotation_matrix =np.zeros((n_frames, len(self.keys)), dtype=np.int32)
        for k in clip_annotation:
            key_index = self.keys.index(k)
            for v in clip_annotation[k]:
                value_idx = self.values.index(v)
                for start_idx, end_idx in clip_annotation[k][v]:
                    print(key_index, start_idx, end_idx, value_idx)
                    clip_annotation_matrix[start_idx:end_idx, key_index] = value_idx
        return clip_annotation_matrix

        

def plot_data(data, title=""):
    fig, ax = plt.subplots()
    n_frames = len(data)
    if title !="": plt.title(title)
    ax.plot(data)
    #ax.bar(list(range(len(data))), data)
    plt.tight_layout()
    plt.show()



def load_json_file(file_path):
    if os.path.isfile(file_path):
        with open(file_path, "r") as in_file:
            return json.load(in_file)


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
        self.ground_motion = kwargs.get("ground_motion", True)
        self.sampling_rate = kwargs.get("sampling_rate", 1600)
        self.n_mels = kwargs.get("n_mels", 17)
        self.ignore_list = kwargs.get("ignore_list", [])
        self.bone_map = kwargs.get("bone_map", [])
        self.offset_rot = None#np.pi/2
        self.fps = kwargs.get("fps", 60) 
        self.src_fps = kwargs.get("src_fps", 60) 
        self.ref_dir = np.array([0, 0, 1])
        self.resample_rate = self.fps/ self.src_fps #1.8
        self.clip_annotation_dict = None
        self.annotation_matrix_builder = None
        self.remove_broken_frames = kwargs.get("remove_broken_frames", False)
        self.debug_plot = kwargs.get("debug_plot", False)
        self.mask_indices_dict = None

    def resample_motion(self, positions, rotations, annotations=None, sample_rate=1.8):
        """ Supersample 
        sample_rate * n_frames
        """
        
        nframes = positions.shape[0]
        nbones = positions.shape[1]
        # Supersample data to 60 fps
        original_times = np.linspace(0, nframes - 1, nframes)
        sample_times = np.linspace(0, nframes - 1, int((sample_rate * nframes)- 1))
        
        # This does a cubic interpolation of the data for supersampling and also speeding up by 10%
        positions = griddata(original_times, positions.reshape([nframes, -1]), sample_times, method='cubic').reshape([len(sample_times), nbones, 3])
        rotations = griddata(original_times, rotations.reshape([nframes, -1]), sample_times, method='cubic').reshape([len(sample_times), nbones, 4])
        # Need to re-normalize after super-sampling
        rotations = quat.normalize(rotations)

        if annotations is not None:
            n_annotations = annotations.shape[1]
            print("before", annotations.shape)
            annotations = griddata(original_times, annotations.reshape([nframes, -1]), sample_times, method='cubic').reshape([len(sample_times), n_annotations]).astype(np.int32)
            print("after", annotations.shape)
        return positions, rotations, annotations

    def extract_sim_bone(self, positions, rotations, names, parents):
        """ Extract Simulation Bone """
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
        sim_direction = quat.mul_vec(twist_qs, self.ref_dir)
  
        # We need to re-normalize the direction after both projection and smoothing
        sim_direction = sim_direction / np.sqrt(np.sum(np.square(sim_direction), axis=-1))[...,np.newaxis]
        sim_direction = signal.savgol_filter(sim_direction, 61, 3, axis=0, mode='interp')
        sim_direction = sim_direction / np.sqrt(np.sum(np.square(sim_direction), axis=-1)[...,np.newaxis])
            
        # align to reference rotation
        # this caused issues when checking the original poses
        initial_offset_rotation = quat.normalize(quat.between(sim_direction[0:1], self.ref_dir))
        #print(sim_direction[0:1].shape, initial_offset_rotation.shape)
        rotations[:,0:1] = quat.mul(initial_offset_rotation[0:1], rotations[:,0:1])
        #align positions as well
        positions[:,0:1] = quat.mul_vec(initial_offset_rotation[0:1], positions[:,0:1])
        sim_position = quat.mul_vec(initial_offset_rotation[0:1],sim_position)


        #for i in range(len(rotations)):
        #    q = quat.mul(initial_offset_rotation[0:1], rotations[i,0:1])
        #    #print(initial_offset_rotation.shape, rotations[i,0:1].shape, q)
        #    rotations[i,0:1] = q
        print("before", sim_direction[0])
        sim_direction = quat.mul_vec(initial_offset_rotation, sim_direction)
        print("after", sim_direction[0])
        # Extract rotation from direction
        sim_rotation = quat.normalize(quat.between(self.ref_dir, sim_direction))
        print("sim_rotation", quat.to_euler(sim_rotation[0]))

        # Transform first joints to be local to sim and append sim as root bone
        
        #print("before", sim_rotation_joint, sim_direction[0], quat.to_euler(sim_rotation[0]), quat.to_euler(rotations[0,0]))
        positions[:,0:1] = quat.mul_vec(quat.inv(sim_rotation), positions[:,0:1] - sim_position)
        rotations[:,0:1] = quat.mul(quat.inv(sim_rotation), rotations[:,0:1])
        #if mirror:
        if self.offset_rot is not None:
            rotations[:,0:1] =quat.mul(quat.from_angle_axis(self.offset_rot, [0,1,0]), rotations[:,0:1])
            #sim_rotation =quat.mul(quat.from_angle_axis(self.offset_rot, [0,1,0]), sim_rotation)

        #print("after", quat.to_euler(rotations[0,0]))
        
        positions = np.concatenate([sim_position, positions], axis=1)
        rotations = np.concatenate([sim_rotation, rotations], axis=1)

        bone_parents = np.concatenate([[-1], parents + 1])
        
        bone_names = ['Simulation'] + names
        return positions, rotations, bone_names, bone_parents


    def compute_velocities(self, positions, rotations):
        # Compute velocities via central difference
        velocities = np.empty_like(positions)
        velocities[1:-1] = (
            0.5 * (positions[2:  ] - positions[1:-1]) +
            0.5 * (positions[1:-1] - positions[ :-2]) )
        velocities[ 0] = velocities[ 1] - (velocities[ 3] - velocities[ 2])
        velocities[-1] = velocities[-2] + (velocities[-2] - velocities[-3])
        
        # Same for angular velocities
        angular_velocities = np.zeros_like(positions)
        angular_velocities[1:-1] = (
            0.5 * quat.to_scaled_angle_axis(quat.abs(quat.mul_inv(rotations[2:  ], rotations[1:-1])))  +
            0.5 * quat.to_scaled_angle_axis(quat.abs(quat.mul_inv(rotations[1:-1], rotations[ :-2]))))
        angular_velocities[ 0] = angular_velocities[ 1] - (angular_velocities[ 3] - angular_velocities[ 2])
        angular_velocities[-1] = angular_velocities[-2] + (angular_velocities[-2] - angular_velocities[-3])
        return velocities, angular_velocities

    def compute_contacts(self, positions, rotations, velocities, angular_velocities, bone_names, bone_parents):
        
        """ Compute Contact Data """ 

        _, _, global_velocities, _ = quat.fk_vel(
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
        return contacts
        

    def process_motion(self, positions, rotations, names, parents):
        positions, rotations, bone_names, bone_parents = self.extract_sim_bone(positions, rotations, names, parents)
        velocities, angular_velocities = self.compute_velocites(positions, rotations)
        contacts = self.compute_contacts(positions, rotations, velocities, angular_velocities, bone_names, bone_parents)


        return positions, velocities, rotations, angular_velocities, bone_names, bone_parents, contacts

    def estimate_ground_offset(self, positions, rotations, parents):
        #toe_idx = self.joint_names.index(self.toes[0])
        global_rotations, global_positions = quat.fk(rotations, positions, parents)
        mininum_y = np.min(global_positions[:, :, 1])
        return np.array([0.0,-mininum_y,0.0], dtype=positions.dtype)

    def resample_data(self, data, sample_rate):
        n_frames = len(data)
        original_times = np.linspace(0, n_frames - 1, n_frames)
        sample_times = np.linspace(0, n_frames - 1, int((sample_rate * n_frames)- 1))
        data = griddata(original_times, data.reshape([n_frames, -1]), sample_times, method='cubic').reshape([len(sample_times)])
        return data
            

    def process_motion_file(self, filename, mirror):
        clip_name = filename.stem + filename.suffix
        positions, rotations, bone_names, bone_parents = load_file(str(filename), self.scale)
        n_original_frames = len(positions)

        clip_annotation = None
        if self.annotation_matrix_builder is not None and len(self.annotation_matrix_builder.keys)>0:
            assert clip_name in self.clip_annotation_dict
            annotation = self.clip_annotation_dict[clip_name]["annotations"]
            n_frames = len(positions)
            clip_annotation = self.annotation_matrix_builder.build_clip_annotation_matrix(annotation, n_frames)
        #print(positions.shape)
        #print(bone_names, len(self.joint_names))
        #bone_names = self.joint_names
        self.grounding_offset = self.estimate_ground_offset(positions, rotations, bone_parents)
        if self.ground_motion:
            #print(positions[:,0].shape, self.grounding_offset)
            positions[:,0] += self.grounding_offset
        
        if mirror:
            rotations, positions = animation_mirror(rotations, positions, bone_names, bone_parents, self.left_prefix, self.right_prefix)
            rotations = quat.unroll(rotations)
            #self.offset_rot = np.pi
        #self.resample_rate = 1.8
        positions, rotations,clip_annotation = self.resample_motion(positions, rotations, clip_annotation, self.resample_rate)

        positions, velocities, rotations, angular_velocities, bone_names, bone_parents, contacts = self.process_motion(positions, rotations, bone_names, bone_parents)
        
        if self.mask_indices_dict is not None:
            if self.remove_broken_frames and clip_name in self.mask_indices_dict:
                mask_indices = np.array(self.mask_indices_dict[clip_name])
                m_start_indices, m_end_indices = [], []
                if len(mask_indices) > 0:
                    mask = np.ones(n_original_frames, np.bool)
                    mask[mask_indices] = 0
                    mask = self.resample_data(mask, self.resample_rate).astype(np.bool)
                    positions = positions[mask,:]
                    velocities = velocities[mask,:]
                    rotations = rotations[mask,:]
                    angular_velocities = angular_velocities[mask,:]
                    contacts = contacts[mask,:]
                    if clip_annotation is not None:
                        clip_annotation = clip_annotation[mask,:]
        return positions, velocities, rotations, angular_velocities, bone_names, bone_parents, contacts, clip_annotation


    def load_annotation(self, motion_path):
        filename = motion_path+os.sep+"annotation.json"
        if not os.path.isfile(filename):
            return
        self.clip_annotation_dict = load_json_file(filename)
        self.annotation_matrix_builder = AnnotationMatrixBuilder(self.clip_annotation_dict)
        self.mask_indices_dict = dict()
        for f in self.clip_annotation_dict:
            self.mask_indices_dict[f] = []
            if "ignore_frames" in self.clip_annotation_dict[f]:
                for start_idx, end_idx in self.clip_annotation_dict[f]["ignore_frames"]:
                    self.mask_indices_dict[f] += list(range(start_idx, end_idx))

        

    def get_file_list(self, motion_path, n_max_files=-1):
        self.load_annotation(motion_path)
        filenames = []
        n_files = 0
        for filename in Path(motion_path).iterdir():
            filename_str =  filename.stem + filename.suffix
            if not filename_str.endswith("bvh"):
                continue
            if contains_element_in_list(filename_str, self.ignore_list):
                continue
            if self.clip_annotation_dict is not None and filename_str not in self.clip_annotation_dict:
                continue
            n_files+=1
            print("add", filename_str)
            filenames.append(filename)
            if n_files >= n_max_files and n_max_files > 0:
                break
        return filenames

        

    def create_db(self, motion_path, n_max_files=-1):
        db = MMDatabase()
        for filename in self.get_file_list(motion_path, n_max_files):
            mirror = False
            print('Loading "%s" %s...' % (str(filename), "(Mirrored)" if mirror else ""))
            positions, velocities, rotations, angular_velocities, bone_names, bone_parents, contacts, clip_annotation = self.process_motion_file(filename, mirror)

            x = np.linspace(0, len(positions)*1/self.fps, len(positions))
            phase_data = np.abs(np.sin(x))
            
            db.append(positions, velocities, rotations, angular_velocities, contacts, phase_data)
            if clip_annotation is not None:                
                db.append_clip_annotation(self.annotation_matrix_builder.keys, self.annotation_matrix_builder.values, clip_annotation)
        db.set_skeleton(bone_names, bone_parents, self.bone_map)
        return db


def load_ignore_list(filename):
    ignore_list = []
    with open(filename, "rt") as in_file:
        ignore_list.append(in_file.readline())
    return ignore_list
