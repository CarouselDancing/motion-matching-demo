import numpy as np
import struct
from transformations import quaternion_matrix, quaternion_inverse
from mm_features import MMFeature, MMFeatureType, calculate_normalized_features
from utils import UHumanBodyBones


class MotionDatabase:
    fps = 60
    bone_positions = []
    bone_velocities = []
    bone_rotations = []
    bone_angular_velocities = []
    bone_parents = []
    range_starts = []
    range_stops = []
    contact_states = []
    phase_data = []
    bone_names = []
    bone_map = []
    annotation_keys = []
    annotation_values = []
    annotation_matrix = None
    features = None
    features_mean = None
    features_scale = None
    feature_descs = []

    def append_clip_annotation(self, keys, values, clip_annotation_matrix):
        self.annotation_keys = keys
        self.annotation_values = values
        print(clip_annotation_matrix)
        if self.annotation_matrix is None:
            self.annotation_matrix = clip_annotation_matrix.astype(np.int32)
        else:
            self.annotation_matrix = np.concatenate([self.annotation_matrix, clip_annotation_matrix], axis=0).astype(np.int32)
        print("clip---------------------",clip_annotation_matrix.shape)
        print("full----------------",self.annotation_matrix.shape)

    def set_skeleton(self, bone_names, bone_parents, bone_map):
        self.bone_names = bone_names
        self.bone_parents = bone_parents
        print(len(bone_map))
        self.bone_map = np.array(list(map(int,bone_map)), dtype=np.int32)
        print(len(self.bone_map))

    def append(self,positions, velocities, rotations, angular_velocities, contacts, phase_data=None):
        self.bone_positions.append(positions)
        self.bone_velocities.append(velocities)
        self.bone_rotations.append(rotations)
        self.bone_angular_velocities.append(angular_velocities)
        
        offset = 0 if len(self.range_starts) == 0 else self.range_stops[-1] 

        self.range_starts.append(offset)
        self.range_stops.append(offset + len(positions))
        self.contact_states.append(contacts)
        if phase_data is not None:
            self.phase_data.append(phase_data)

    def concatenate_data(self, convert_coodinate_system=False):
        self.bone_positions = np.concatenate(self.bone_positions, axis=0).astype(np.float32)
        
        self.bone_velocities = np.concatenate(self.bone_velocities, axis=0).astype(np.float32)
        self.bone_rotations = np.concatenate(self.bone_rotations, axis=0).astype(np.float32)
        self.bone_angular_velocities = np.concatenate(self.bone_angular_velocities, axis=0).astype(np.float32)
        self.bone_parents = self.bone_parents.astype(np.int32)
        if convert_coodinate_system:
            self.bone_positions[:, : ,0 ] *= -1
            self.bone_rotations[:, : ,:2] *= -1 # 0 and 1
            self.bone_velocities[:, : ,0 ] *= -1
            self.bone_angular_velocities[:, : ,0 ] *= -1

        self.range_starts = np.array(self.range_starts).astype(np.int32)
        self.range_stops = np.array(self.range_stops).astype(np.int32)
        self.contact_states = np.concatenate(self.contact_states, axis=0).astype(np.uint8)
        if len(self.phase_data) > 0:# and len(self.phase_data.shape) > 1:
            #print(self.phase_data.shape)
            self.phase_data = np.concatenate(self.phase_data, axis=0).astype(np.float32)


    def write(self, filename, concatenate=True):
                
        if concatenate: self.concatenate_data()

        """ Write Database """
        print("Writing Database...")

        with open(filename, 'wb') as f:

            nframes = self.bone_positions.shape[0]
            nbones = self.bone_positions.shape[1]
            nranges = self.range_starts.shape[0]
            ncontacts = self.contact_states.shape[1]

            f.write(struct.pack('II', nframes, nbones) + self.bone_positions.ravel().tobytes())
            f.write(struct.pack('II', nframes, nbones) + self.bone_velocities.ravel().tobytes())
            f.write(struct.pack('II', nframes, nbones) + self.bone_rotations.ravel().tobytes())
            f.write(struct.pack('II', nframes, nbones) + self.bone_angular_velocities.ravel().tobytes())
            f.write(struct.pack('I', nbones) + self.bone_parents.ravel().tobytes())
            
            f.write(struct.pack('I', nranges) + self.range_starts.ravel().tobytes())
            f.write(struct.pack('I', nranges) + self.range_stops.ravel().tobytes())
            
            f.write(struct.pack('II', nframes, ncontacts) + self.contact_states.ravel().tobytes())

            self.save_string_list(self.bone_names, f)
            f.write(struct.pack('I', nbones) + self.bone_map.ravel().tobytes())

            print("save bone map",len(self.bone_map), self.bone_map.dtype)
            if len(self.phase_data) > 0:
                #self.phase_data = np.concatenate(self.phase_data, axis=0).astype(np.float32)
                n_phase_dims = 1
                #if len(self.phase_data.shape) >1:
                #    n_phase_dims = self.phase_data.shape[1]
                f.write(struct.pack('II', nframes, n_phase_dims) + self.phase_data.ravel().tobytes())
            
            if self.annotation_matrix is not None:
                self.save_string_list(self.annotation_keys, f)
                self.save_string_list(self.annotation_values, f)
                n_annotations = self.annotation_matrix.shape[1]
                f.write(struct.pack('II', nframes, n_annotations) + self.annotation_matrix.ravel().tobytes())

    def write_to_numpy(self, filename, concatenate=True):
        
        if concatenate: self.concatenate_data()
        data = self.to_dict()
        #np.save(filename, data)
        print("save", filename)
        np.savez_compressed(filename, **data)
        #np.savez(filename, **data)
        

    def load_from_numpy(self, filename):
        data = np.load(filename, allow_pickle=True)
        self.from_dict(data)

    def to_dict(self):

        nFrames = self.bone_positions.shape[0] 
        nBones = self.bone_positions.shape[1]
        data = dict()
        data["meta_data_keys"] = self.string_list_to_int_list(["nFrames", "nBones", "fps"])
        data["meta_data_values"] = np.array([nFrames, nBones, self.fps]).astype(np.float32) 
        #data["nFrames"] = np.array([nFrames]).astype(np.int32) 
        #data["nBones"] = np.array([nBones]).astype(np.int32) 
        #data["fps"] = np.array([self.fps]).astype(np.float32) 
        data["bone_positions"] = self.bone_positions
        data["bone_velocities"] =self.bone_velocities
        data["bone_rotations"] = self.bone_rotations
        data["bone_angular_velocities"] = self.bone_angular_velocities
        data["bone_parents"] = self.bone_parents
        data["range_starts"] = np.array(self.range_starts).astype(np.int32)
        data["range_stops"] = np.array(self.range_stops).astype(np.int32)
        data["contact_states"] = np.array(self.contact_states).astype(np.int32)
        data["bone_names"] = self.string_list_to_int_list(self.bone_names)#np.array([ord(c) for c in self.concat_str_list(self.bone_names)]).astype(np.int32)
        
        data["bone_map"] = np.array(self.bone_map).astype(np.int32)
        print("bone_map", data["bone_names"])
        
        if len(self.phase_data) > 0:
            data["phase_data"] = self.phase_data
        if self.annotation_matrix is not None:
            data["annotation_keys"] = self.string_list_to_int_list(self.annotation_keys)# str.encode(self.concat_str_list(self.annotation_keys), 'utf-8')
            data["annotation_values"] = self.string_list_to_int_list(self.annotation_values)#str.encode(self.concat_str_list(self.annotation_values), 'utf-8')
            data["annotation_matrix"] = self.annotation_matrix

        if self.features is not None:
            data["features"] = np.array(self.features).astype(np.float32) 
            data["features_mean"] = np.array(self.features_mean).astype(np.float32)
            data["features_scale"] = np.array(self.features_scale).astype(np.float32)
            data["feature_types"] = np.array([int(f.ftype) for f in self.feature_descs], np.int32)
            data["feature_bones"] = np.array([int(f.bone) for f in self.feature_descs], np.int32)
            data["feature_weights"] = np.array([f.weight for f in self.feature_descs], np.float32)
        return data

    def from_dict(self, data):
        meta_data_keys =self.int_list_to_string_list(data["meta_data_keys"])
        meta_data_values = data["meta_data_values"]
        fps_index = meta_data_keys.index("fps")
        self.fps = meta_data_values[fps_index]
        self.bone_positions = data["bone_positions"]
        self.bone_velocities = data["bone_velocities"]
        self.bone_rotations = data["bone_rotations"]
        self.bone_angular_velocities = data["bone_angular_velocities"]
        self.bone_parents = data["bone_parents"]
        self.range_starts = data["range_starts"]
        self.range_stops = data["range_stops"]
        self.contact_states = np.array(data["contact_states"]).astype(np.int32) 
        #print(data["bone_names"])
        #sys.exit()
       
        self.bone_names =self.int_list_to_string_list( data["bone_names"])
        self.bone_map = data["bone_map"]
        if "phase_data" in data:
            self.phase_data =  data["phase_data"]
        if "annotation_keys" in data:
            self.annotation_keys =  self.int_list_to_string_list(data["annotation_keys"])
            self.annotation_values = self.int_list_to_string_list(data["annotation_values"])
            self.annotation_matrix =  data["annotation_matrix"]
        if "features" in data:
            self.features = data["features"]
            self.features_mean = data["features_mean"]
            self.features_scale = data["features_scale"]
            self.feature_descs = []
            feature_bones = data["feature_bones"]
            feature_types = data["feature_types"]
            feature_weights = data["feature_weights"]
            for bone, ftype, weight in zip(feature_bones, feature_types, feature_weights):
                f_desc = MMFeature(UHumanBodyBones(bone), MMFeatureType(ftype), weight)
                self.feature_descs.append(f_desc)
                print(f_desc.bone)
                print(f_desc.ftype)
                print(f_desc.weight)

    def string_list_to_int_list(self, names):
        return np.array([ord(c) for c in self.concat_str_list(names)]).astype(np.int32)


    def int_list_to_string_list(self, int_list):
        concat_str =  "".join([chr(c) for c in int_list])
        return concat_str.split(",")

    def concat_str_list(self, string_list):
        concat_str = ""
        for key in string_list: 
            concat_str += key +","
        concat_str = concat_str[:-1]
        return concat_str

    def save_string_list(self, string_list, outfile):
        concat_str = self.concat_str_list(string_list)
        outfile.write(struct.pack('I', len(concat_str))+str.encode(concat_str, 'utf-8'))

            
    def load(self, filename, load_phase=False, load_annotation=False):
        with open(filename, 'rb') as f:
            

            nframes, nbones = struct.unpack('II', f.read(8))
            self.bone_positions = np.frombuffer(f.read(nframes*nbones*3*4), dtype=np.float32, count=nframes*nbones*3).reshape([nframes, nbones, 3])
            print("loaded", nframes, nbones)
            
            nframes, nbones = struct.unpack('II', f.read(8))
            self.bone_velocities = np.frombuffer(f.read(nframes*nbones*3*4), dtype=np.float32, count=nframes*nbones*3).reshape([nframes, nbones, 3])
            
            nframes, nbones = struct.unpack('II', f.read(8))
            self.bone_rotations = np.frombuffer(f.read(nframes*nbones*4*4), dtype=np.float32, count=nframes*nbones*4).reshape([nframes, nbones, 4])
            
            nframes, nbones = struct.unpack('II', f.read(8))
            self.bone_angular_velocities = np.frombuffer(f.read(nframes*nbones*3*4), dtype=np.float32, count=nframes*nbones*3).reshape([nframes, nbones, 3])
            
            print("loaded", nframes, nbones)
            nbones = struct.unpack('I', f.read(4))[0]
            self.bone_parents = np.frombuffer(f.read(nbones*4), dtype=np.int32, count=nbones).reshape([nbones])
            
            nranges = struct.unpack('I', f.read(4))[0]
            self.range_starts = np.frombuffer(f.read(nranges*4), dtype=np.int32, count=nranges).reshape([nranges])
            
            nranges = struct.unpack('I', f.read(4))[0]
            self.range_stops = np.frombuffer(f.read(nranges*4), dtype=np.int32, count=nranges).reshape([nranges])
            
            nframes, ncontacts = struct.unpack('II', f.read(8))
            self.contact_states = np.frombuffer(f.read(nframes*ncontacts), dtype=np.int8, count=nframes*ncontacts).reshape([nframes, ncontacts])

            self.bone_names = self.load_string_list(f)
            
            nbones = struct.unpack('I', f.read(4))[0]
            self.bone_map = np.frombuffer(f.read(nbones*4), dtype=np.int32, count=nbones).reshape([nbones])

            if load_phase:
                nframes, n_phase_dims = struct.unpack('II', f.read(8))
                self.phase_data = np.frombuffer(f.read(nframes*n_phase_dims*4), dtype=np.float32, count=nframes*n_phase_dims).reshape([nframes, n_phase_dims])
                if load_annotation:
                    self.annotation_keys = self.load_string_list(f)
                    self.annotation_values = self.load_string_list(f)
                    nframes, n_annotations = struct.unpack('II', f.read(8))
                    self.annotation_matrix = np.frombuffer(f.read(nframes*n_annotations*4), dtype=np.int32, count=nframes*n_annotations).reshape([nframes, n_annotations])
                    print(self.annotation_keys)
                    print(self.annotation_values)
        print("loaded", nframes, self.bone_names, len(self.bone_names))

    def split_str(self, concat_bytes):
        print(concat_bytes)
        concat_str = str(concat_bytes,"utf-8")
        return concat_str.split(",")
    
    def load_string_list(self, infile):
        str_len = struct.unpack('I',infile.read(4))[0]
        concat_str = self.split_str(infile.read(str_len))# str(infile.read(str_len),"utf-8")
        return concat_str.split(",")
            
    def print_shape(self):
        print("bone_positions",self.bone_positions.shape)
        print("bone_rotations",self.bone_rotations.shape)
        print("bone_angular_velocities",self.bone_angular_velocities.shape)
        print("bone_parents",self.bone_parents.shape)
        print("range_starts",self.range_starts.shape)
        print("range_stops",self.range_stops.shape)
        print("contact_states",self.contact_states.shape)
        if self.features is not None:
            print("features", self.features.shape)


    def get_relative_bone_positions(self, bone_idx):
        n_frames = len(self.bone_positions)
        positions = np.zeros((n_frames, 3))
        for frame_idx in range(n_frames):
            p, r = self.fk(frame_idx, bone_idx)
            p = p - self.bone_positions[frame_idx, 0]
            inv_root_m = quaternion_matrix(quaternion_inverse(self.bone_rotations[frame_idx, 0]))[:3, :3]
            p = np.dot(inv_root_m, p)
            positions[frame_idx] = p
        return positions

    def get_relative_bone_velocities(self, bone_idx):
        n_frames = len(self.bone_positions)
        velocities = np.zeros((n_frames, 3))
        for frame_idx in range(n_frames):
            p, lv, r, av = self.fk_velocity(frame_idx, bone_idx)
            inv_root_m = quaternion_matrix(quaternion_inverse(self.bone_rotations[frame_idx, 0]))[:3, :3]
            lv = np.dot(inv_root_m, lv)
            velocities[frame_idx] = lv
        return velocities

    def trajectory_index_clamp(self, frame_idx, offset):
        n_frames = len(self.bone_positions)
        frame_range_idx = -1
        for i in range(len(self.range_starts)):
            if frame_idx >= self.range_starts[i] and frame_idx < self.range_stops[i]:
                frame_range_idx = i
                break
        
        if frame_range_idx > 0:
            frame_idx = max(self.range_starts[frame_range_idx], min(frame_idx+offset, self.range_stops[frame_range_idx]-1))
        if frame_idx < 0: 
            frame_idx = 0
        elif frame_idx > n_frames:
             frame_idx = n_frames-1

        return frame_idx

    def get_position_trajectories(self, bone_idx):
        n_frames = len(self.bone_positions)
        pos_trajectories = np.zeros((n_frames, 6))
        for frame_idx in range(n_frames):
            t0 = self.trajectory_index_clamp(frame_idx, 20)
            t1 = self.trajectory_index_clamp(frame_idx, 40)
            t2 = self.trajectory_index_clamp(frame_idx, 60)
            pos_trajectories[frame_idx] = self.get_position_trajectory(frame_idx, t0, t1, t2, bone_idx)
        return pos_trajectories

    def get_position_trajectory(self, frame_idx, t0, t1, t2, bone_idx):
        pos_trajectory = np.zeros(6)
        frame_pos, frame_rot = self.fk(frame_idx, bone_idx)
        inv_frame_rot = np.linalg.inv(frame_rot)
        frame_pos = self.bone_positions[frame_idx, bone_idx]
        t0_pos, _ = self.fk(t0, bone_idx)
        t1_pos, _ = self.fk(t1, bone_idx)
        t2_pos, _ = self.fk(t2, bone_idx)
        delta0 = np.dot(inv_frame_rot, t0_pos- frame_pos)
        delta1 = np.dot(inv_frame_rot, t1_pos- frame_pos)
        delta2 = np.dot(inv_frame_rot, t2_pos- frame_pos)
        pos_trajectory[0] = delta0[0]
        pos_trajectory[1] = delta0[2]
        pos_trajectory[2] = delta1[0]
        pos_trajectory[3] = delta1[2]
        pos_trajectory[4] = delta2[0]
        pos_trajectory[5] = delta2[2]
        return pos_trajectory

    def get_direction_trajectories(self, bone_idx):
        n_frames = len(self.bone_positions)
        pos_trajectories = np.zeros((n_frames, 6))
        for frame_idx in range(n_frames):
            t0 = self.trajectory_index_clamp(frame_idx, 20)
            t1 = self.trajectory_index_clamp(frame_idx, 40)
            t2 = self.trajectory_index_clamp(frame_idx, 60)
            pos_trajectories[frame_idx] = self.get_direction_trajectory(frame_idx, t0, t1, t2, bone_idx)
        return pos_trajectories

    def get_direction_trajectory(self, frame_idx, t0, t1, t2, bone_idx):
        pos_trajectory = np.zeros(6)
        frame_pos, frame_rot = self.fk(frame_idx, bone_idx)
        inv_frame_rot = np.linalg.inv(frame_rot)
        _, t0_rot = self.fk(t0, bone_idx)
        _, t1_rot = self.fk(t1, bone_idx)
        _, t2_rot = self.fk(t2, bone_idx)
        delta0 = np.dot(inv_frame_rot, np.dot(t0_rot, [0,0,1]))
        delta1 = np.dot(inv_frame_rot, np.dot(t1_rot, [0,0,1]))
        delta2 = np.dot(inv_frame_rot, np.dot(t2_rot, [0,0,1]))
        pos_trajectory[0] = delta0[0]
        pos_trajectory[1] = delta0[2]
        pos_trajectory[2] = delta1[0]
        pos_trajectory[3] = delta1[2]
        pos_trajectory[4] = delta2[0]
        pos_trajectory[5] = delta2[2]
        return pos_trajectory


    def fk(self, frame_idx, bone_idx):
        if self.bone_parents[bone_idx] != -1:
            parent_idx = self.bone_parents[bone_idx]
            parent_pos, parent_rot = self.fk(frame_idx, parent_idx)
            bone_m = quaternion_matrix(self.bone_rotations[frame_idx, bone_idx])[:3, :3]
            global_pos = parent_pos+ np.dot(parent_rot, self.bone_positions[frame_idx, bone_idx])
            global_rot = np.dot(parent_rot, bone_m)
            return global_pos, global_rot

        else:
            return self.bone_positions[frame_idx, bone_idx], quaternion_matrix(self.bone_rotations[frame_idx, bone_idx])[:3, :3]

    def fk_velocity(self, frame_idx, bone_idx):
        if self.bone_parents[bone_idx] != -1:
            parent_idx = self.bone_parents[bone_idx]
            parent_pos, parent_lv, parent_rot, parent_av = self.fk_velocity(frame_idx, parent_idx)
            bone_m = quaternion_matrix(self.bone_rotations[frame_idx, bone_idx])[:3, :3]
            global_pos = parent_pos+ np.dot(parent_rot, self.bone_positions[frame_idx, bone_idx])
            global_vel = parent_lv + np.dot(parent_rot, self.bone_velocities[frame_idx, bone_idx]) + np.cross(parent_av, np.dot(parent_rot, self.bone_positions[frame_idx, bone_idx]))
            global_rot = np.dot(parent_rot, bone_m)
            global_av = np.dot(parent_rot, (self.bone_angular_velocities[frame_idx, bone_idx] + parent_av))
            return global_pos, global_vel, global_rot, global_av

        else:
            return self.bone_positions[frame_idx, bone_idx], \
                self.bone_velocities[frame_idx, bone_idx], \
                quaternion_matrix(self.bone_rotations[frame_idx, bone_idx])[:3, :3], \
                self.bone_angular_velocities[frame_idx, bone_idx]


    def calculate_features(self, feature_descs):
        self.feature_descs = feature_descs
        self.features, self.features_mean, self.features_scale = calculate_normalized_features(self, feature_descs)
        print("finished calculating features")
