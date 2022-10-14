
import numpy as np
from . import quat
from transformations import quaternion_matrix, quaternion_from_matrix, quaternion_from_euler


class MMPose:
    def __init__(self, parents, scale=1, use_velocity=True, offset=None)-> None:
        if offset is None:
            offset = np.zeros(3)
        self.offset = offset
        self.parents = parents
        self.n_bones = len(self.parents)
        self.positions = np.zeros((self.n_bones, 3))
        self.rotations = np.zeros((self.n_bones, 4))
        self.global_positions = np.zeros((self.n_bones, 3))
        self.global_rotations = np.zeros((self.n_bones, 4))
        self.sim_position = np.array(self.offset, dtype=np.float32)
        self.sim_rotation = np.array([1,0,0,0], dtype=np.float32)
        self.lin_vel = np.array([0,0,0], dtype=np.float32)
        self.ang_vel = np.array([0,0,0], dtype=np.float32)
        self.use_velocity = use_velocity
        self.scale = scale
        self.ref_position = np.array([0,0,0], dtype=np.float32)
        self.ref_rotation = np.array([1,0,0,0], dtype=np.float32)
        self.frame_idx = 0

    @classmethod
    def from_db(cls, db, frame_idx, scale=1, offset=None):
        pose = MMPose(db.bone_parents, scale, offset=offset)
        pose.set_pose(db, frame_idx)
        return pose

    def set_pose(self, db, frame_idx):
        self.frame_idx = frame_idx
        self.global_positions = np.zeros((self.n_bones, 3))
        self.global_rotations = np.zeros((self.n_bones, 4))
        self.positions = np.array(db.bone_positions[frame_idx])*self.scale
        self.rotations = np.array(db.bone_rotations[frame_idx])
        if self.use_velocity:
            self.positions[0] = self.sim_position
            self.rotations[0] = self.sim_rotation
        self.ref_position = np.array(db.bone_positions[frame_idx,0])*self.scale
        self.ref_rotation = np.array(db.bone_rotations[frame_idx,0])
        inv_root_rot = quat.normalize(quat.inv(self.ref_rotation))
        self.lin_vel = quat.mul_vec(inv_root_rot, db.bone_velocities[frame_idx,0])*self.scale
        self.ang_vel = -quat.mul_vec(inv_root_rot, db.bone_angular_velocities[frame_idx,0])
        self.update_fk_buffer()

    def update_sim(self, dt):
        #delta_pos  = quat.mul_vec(self.sim_rotation, self.lin_vel*dt)
        #self.sim_position += delta_pos
        #global_ang_vel = quat.mul_vec(self.sim_rotation, self.ang_vel*dt)
        #delta_rot = quat.from_euler(global_ang_vel)
        #self.sim_rotation = quat.mul(self.sim_rotation, delta_rot)
        velocity = quat.mul_vec(self.sim_rotation, self.lin_vel)*dt
        av = quat.mul_vec(self.sim_rotation, self.ang_vel)*dt
        delta_rot = quat.from_euler(av, "xyz")
        self.sim_position += velocity
        self.sim_rotation = quat.normalize(quat.mul(delta_rot, self.sim_rotation))

    def update_fk_buffer(self):
        for i in range(self.n_bones):
            self.fk(i)

    def fk(self, bone_idx):
        if self.parents[bone_idx] != -1:
            parent_idx = self.parents[bone_idx]
            #self.fk(parent_idx)
            parent_q = self.global_rotations[parent_idx]
            bone_q = self.rotations[bone_idx]
            global_q = quat.mul(parent_q, bone_q)
            rotated_pos =quat.mul_vec(parent_q, self.positions[bone_idx])
            self.global_positions[bone_idx] = self.global_positions[parent_idx] + rotated_pos
            
            self.global_rotations[bone_idx] = global_q

        else:
            self.global_positions[bone_idx] = self.positions[bone_idx]
            self.global_rotations[bone_idx] = self.rotations[bone_idx]

    def reset(self, frame_idx):
        self.frame_idx = frame_idx#self.start_frame_idx
        self.sim_position = np.array(self.offset, dtype=np.float32)
        self.sim_rotation = np.array([1,0,0,0], dtype=np.float32)
        self.lin_vel = np.array([0,0,0], dtype=np.float32)
        self.ang_vel = np.array([0,0,0], dtype=np.float32)
            
    def get_linear_velocity(self):
        return quat.mul_vec(self.sim_rotation, self.lin_vel)

    def get_ref_linear_velocity(self):
        return quat.mul_vec(self.ref_rotation, self.lin_vel)
