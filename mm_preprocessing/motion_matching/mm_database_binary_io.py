
import numpy as np
import struct
from .mm_database import MMDatabase

class MMDatabaseBinaryIO:
    @staticmethod
    def load(filename, load_phase=False, load_annotation=False):
        db = MMDatabase()
        with open(filename, 'rb') as f:

            nframes, nbones = struct.unpack('II', f.read(8))
            db.bone_positions = np.frombuffer(f.read(nframes*nbones*3*4), dtype=np.float32, count=nframes*nbones*3).reshape([nframes, nbones, 3])
            print("loaded", nframes, nbones)
            
            nframes, nbones = struct.unpack('II', f.read(8))
            db.bone_velocities = np.frombuffer(f.read(nframes*nbones*3*4), dtype=np.float32, count=nframes*nbones*3).reshape([nframes, nbones, 3])
            
            nframes, nbones = struct.unpack('II', f.read(8))
            db.bone_rotations = np.frombuffer(f.read(nframes*nbones*4*4), dtype=np.float32, count=nframes*nbones*4).reshape([nframes, nbones, 4])
            
            nframes, nbones = struct.unpack('II', f.read(8))
            db.bone_angular_velocities = np.frombuffer(f.read(nframes*nbones*3*4), dtype=np.float32, count=nframes*nbones*3).reshape([nframes, nbones, 3])
            
            print("loaded", nframes, nbones)
            nbones = struct.unpack('I', f.read(4))[0]
            db.bone_parents = np.frombuffer(f.read(nbones*4), dtype=np.int32, count=nbones).reshape([nbones])
            
            nranges = struct.unpack('I', f.read(4))[0]
            db.range_starts = np.frombuffer(f.read(nranges*4), dtype=np.int32, count=nranges).reshape([nranges])
            
            nranges = struct.unpack('I', f.read(4))[0]
            db.range_stops = np.frombuffer(f.read(nranges*4), dtype=np.int32, count=nranges).reshape([nranges])
            
            nframes, ncontacts = struct.unpack('II', f.read(8))
            db.contact_states = np.frombuffer(f.read(nframes*ncontacts), dtype=np.int8, count=nframes*ncontacts).reshape([nframes, ncontacts])

            db.bone_names = MMDatabaseBinaryIO.load_string_list(f)
            
            nbones = struct.unpack('I', f.read(4))[0]
            db.bone_map = np.frombuffer(f.read(nbones*4), dtype=np.int32, count=nbones).reshape([nbones])

            if load_phase:
                nframes, n_phase_dims = struct.unpack('II', f.read(8))
                db.phase_data = np.frombuffer(f.read(nframes*n_phase_dims*4), dtype=np.float32, count=nframes*n_phase_dims).reshape([nframes, n_phase_dims])
                if load_annotation:
                    db.annotation_keys = MMDatabaseBinaryIO.load_string_list(f)
                    db.annotation_values = MMDatabaseBinaryIO.load_string_list(f)
                    nframes, n_annotations = struct.unpack('II', f.read(8))
                    db.annotation_matrix = np.frombuffer(f.read(nframes*n_annotations*4), dtype=np.int32, count=nframes*n_annotations).reshape([nframes, n_annotations])
                    print(db.annotation_keys)
                    print(db.annotation_values)
        print("loaded", nframes, db.bone_names, len(db.bone_names))
        return db
    
    @staticmethod
    def load_string_list(infile):
        str_len = struct.unpack('I',infile.read(4))[0]
        concat_str = MMDatabaseBinaryIO.split_str(infile.read(str_len))# str(infile.read(str_len),"utf-8")
        return concat_str.split(",")

    @staticmethod
    def split_str(concat_bytes):
        print(concat_bytes)
        concat_str = str(concat_bytes,"utf-8")
        return concat_str.split(",")
    
    @staticmethod
    def write(db, filename, concatenate=True):
                
        if concatenate: db.concatenate_data()

        """ Write Database """
        print("Writing Database...")

        with open(filename, 'wb') as f:

            nframes = db.bone_positions.shape[0]
            nbones = db.bone_positions.shape[1]
            nranges = db.range_starts.shape[0]
            ncontacts = db.contact_states.shape[1]

            f.write(struct.pack('II', nframes, nbones) + db.bone_positions.ravel().tobytes())
            f.write(struct.pack('II', nframes, nbones) + db.bone_velocities.ravel().tobytes())
            f.write(struct.pack('II', nframes, nbones) + db.bone_rotations.ravel().tobytes())
            f.write(struct.pack('II', nframes, nbones) + db.bone_angular_velocities.ravel().tobytes())
            f.write(struct.pack('I', nbones) + db.bone_parents.ravel().tobytes())
            
            f.write(struct.pack('I', nranges) + db.range_starts.ravel().tobytes())
            f.write(struct.pack('I', nranges) + db.range_stops.ravel().tobytes())
            
            f.write(struct.pack('II', nframes, ncontacts) + db.contact_states.ravel().tobytes())

            MMDatabaseBinaryIO.save_string_list(db.bone_names, f)
            f.write(struct.pack('I', nbones) + db.bone_map.ravel().tobytes())

            print("save bone map",len(db.bone_map), db.bone_map.dtype)
            if len(db.phase_data) > 0:
                #self.phase_data = np.concatenate(self.phase_data, axis=0).astype(np.float32)
                n_phase_dims = 1
                #if len(self.phase_data.shape) >1:
                #    n_phase_dims = self.phase_data.shape[1]
                f.write(struct.pack('II', nframes, n_phase_dims) + db.phase_data.ravel().tobytes())
            
            if db.annotation_matrix is not None:
                MMDatabaseBinaryIO.save_string_list(db.annotation_keys, f)
                MMDatabaseBinaryIO.save_string_list(db.annotation_values, f)
                n_annotations = db.annotation_matrix.shape[1]
                f.write(struct.pack('II', nframes, n_annotations) + db.annotation_matrix.ravel().tobytes())

    def concat_str_list(string_list):
        concat_str = ""
        for key in string_list: 
            concat_str += key +","
        concat_str = concat_str[:-1]
        return concat_str

    def save_string_list(string_list, outfile):
        concat_str = MMDatabaseBinaryIO.concat_str_list(string_list)
        outfile.write(struct.pack('I', len(concat_str))+str.encode(concat_str, 'utf-8'))


