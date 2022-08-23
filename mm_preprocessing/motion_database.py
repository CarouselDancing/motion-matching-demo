import numpy as np
import struct



class MotionDatabase:
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

    def append_clip_annotation(self, keys, values, clip_annotation_matrix):
        self.annotation_keys = keys
        self.annotation_values = values
        print(clip_annotation_matrix)
        if self.annotation_matrix is None:
            self.annotation_matrix = clip_annotation_matrix.astype(np.int32)
        else:
            self.annotation_matrix = np.concatenate([self.annotation_matrix, clip_annotation_matrix], axis=0).astype(np.int32)

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

    def write(self, filename):
                
        """ Concatenate Data """

        self.bone_positions = np.concatenate(self.bone_positions, axis=0).astype(np.float32)
        self.bone_velocities = np.concatenate(self.bone_velocities, axis=0).astype(np.float32)
        self.bone_rotations = np.concatenate(self.bone_rotations, axis=0).astype(np.float32)
        self.bone_angular_velocities = np.concatenate(self.bone_angular_velocities, axis=0).astype(np.float32)
        self.bone_parents = self.bone_parents.astype(np.int32)

        self.range_starts = np.array(self.range_starts).astype(np.int32)
        self.range_stops = np.array(self.range_stops).astype(np.int32)

        self.contact_states = np.concatenate(self.contact_states, axis=0).astype(np.uint8)

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
                self.phase_data = np.concatenate(self.phase_data, axis=0).astype(np.float32)
                n_phase_dims = 1
                if len(self.phase_data.shape) >1:
                    n_phase_dims = self.phase_data.shape[1]
                f.write(struct.pack('II', nframes, n_phase_dims) + self.phase_data.ravel().tobytes())
            
            if self.annotation_matrix is not None:
                self.save_string_list(self.annotation_keys, f)
                self.save_string_list(self.annotation_values, f)
                n_annotations = self.annotation_matrix.shape[1]
                f.write(struct.pack('II', nframes, n_annotations) + self.annotation_matrix.ravel().tobytes())


    def save_string_list(self, string_list, outfile):
        concat_str = ""
        for key in string_list: 
            concat_str += key +","
        concat_str = concat_str[:-1]
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
        print("loaded", nframes, self.bone_names, len(self.bone_names))
    
    def load_string_list(self, infile):
        str_len = struct.unpack('I',infile.read(4))[0]
        concat_str = str(infile.read(str_len),"utf-8")
        return concat_str.split(",")


            
    def print_shape(self):
        print("bone_positions",self.bone_positions.shape)
        print("bone_rotations",self.bone_rotations.shape)
        print("bone_angular_velocities",self.bone_angular_velocities.shape)
        print("bone_parents",self.bone_parents.shape)
        print("range_starts",self.range_starts.shape)
        print("range_stops",self.range_stops.shape)
        print("contact_states",self.contact_states.shape)


