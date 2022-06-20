from importlib_metadata import re
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
    audio_data = []
    bone_names = []
    bone_map = []

    def set_skeleton(self, bone_names, bone_parents, bone_map):
        self.bone_names = bone_names
        self.bone_parents = bone_parents
        import sys
        print(len(bone_map))
        self.bone_map = np.array(list(map(int,bone_map)), dtype=np.int32)
        print(len(self.bone_map))
        #sys.exit()

    def append(self,positions, velocities, rotations, angular_velocities, contacts, audio_data):
            self.bone_positions.append(positions)
            self.bone_velocities.append(velocities)
            self.bone_rotations.append(rotations)
            self.bone_angular_velocities.append(angular_velocities)
            
            offset = 0 if len(self.range_starts) == 0 else self.range_stops[-1] 

            self.range_starts.append(offset)
            self.range_stops.append(offset + len(positions))
            self.contact_states.append(contacts)
            self.audio_data.append(audio_data)

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
        # self.audio_data = np.concatenate(self.audio_data, axis=0).astype(np.float32)

        """ Write Database """
        print("Writing Database...")

        with open(filename, 'wb') as f:

            nframes = self.bone_positions.shape[0]
            nbones = self.bone_positions.shape[1]
            nranges = self.range_starts.shape[0]
            ncontacts = self.contact_states.shape[1]
            #n_audio_dims = self.audio_data.shape[1]
            
            f.write(struct.pack('II', nframes, nbones) + self.bone_positions.ravel().tobytes())
            f.write(struct.pack('II', nframes, nbones) + self.bone_velocities.ravel().tobytes())
            f.write(struct.pack('II', nframes, nbones) + self.bone_rotations.ravel().tobytes())
            f.write(struct.pack('II', nframes, nbones) + self.bone_angular_velocities.ravel().tobytes())
            f.write(struct.pack('I', nbones) + self.bone_parents.ravel().tobytes())
            
            f.write(struct.pack('I', nranges) + self.range_starts.ravel().tobytes())
            f.write(struct.pack('I', nranges) + self.range_stops.ravel().tobytes())
            
            f.write(struct.pack('II', nframes, ncontacts) + self.contact_states.ravel().tobytes())
            name_str = ""
            for name in self.bone_names: 
                name_str += name +","
            name_str = name_str[:-1]
            len_names_str = len(name_str)
            f.write(struct.pack('I', len_names_str)+str.encode(name_str, 'utf-8'))
            f.write(struct.pack('I', nbones) + self.bone_map.ravel().tobytes())
            print("save bone map",len(self.bone_map), self.bone_map.dtype)
            #f.write(struct.pack('II', nframes, n_audio_dims) + self.audio_data.ravel().tobytes())
            
            
            
    def load(self, filename):
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

            len_names_str = struct.unpack('I', f.read(4))[0]
            name_str = str(f.read(len_names_str),"utf-8")
            self.bone_names = name_str.split(",")
            print(len_names_str)
            
            nbones = struct.unpack('I', f.read(4))[0]
            self.bone_map = np.frombuffer(f.read(nbones*4), dtype=np.int32, count=nbones).reshape([nbones])
            print("loaded bone map",len(self.bone_map), self.bone_map)

        print("loaded", nframes, self.bone_names, len(self.bone_names))
            
    def print_shape(self):
        print("bone_positions",self.bone_positions.shape)
        print("bone_rotations",self.bone_rotations.shape)
        print("bone_angular_velocities",self.bone_angular_velocities.shape)
        print("bone_parents",self.bone_parents.shape)
        print("range_starts",self.range_starts.shape)
        print("range_stops",self.range_stops.shape)
        print("contact_states",self.contact_states.shape)


