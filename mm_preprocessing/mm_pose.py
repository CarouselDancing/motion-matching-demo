
import numpy as np
import quat
from transformations import quaternion_matrix, quaternion_from_matrix, quaternion_from_euler

class MMPose:
    def __init__(self, parents, scale=10, use_sim=True)-> None:
        
        self.parents = parents
        self.n_bones = len(self.parents)
        self.positions = np.zeros((self.n_bones, 3))
        self.rotations = np.zeros((self.n_bones, 4))
        self.global_positions = np.zeros((self.n_bones, 3))
        self.global_rotations = np.zeros((self.n_bones, 4))
        self.sim_position = np.array([0,0,0], dtype=np.float32)
        self.sim_rotation = np.array([1,0,0,0], dtype=np.float32)
        self.lin_vel = np.array([0,0,0], dtype=np.float32)
        self.ang_vel = np.array([0,0,0], dtype=np.float32)
        self.use_sim = use_sim
        self.scale = scale

    @classmethod
    def from_db(cls, db, frame_idx):
        pose = MMPose(db.bone_parents)
        pose.set_pose(db, frame_idx)
        return pose

    def set_pose(self, db, frame_idx):
        self.global_positions = np.zeros((self.n_bones, 3))
        self.global_rotations = np.zeros((self.n_bones, 4))
        self.positions = db.bone_positions[frame_idx]*self.scale
        self.rotations = db.bone_rotations[frame_idx]
        if self.use_sim:
            self.positions[0] = self.sim_position
            self.rotations[0] = self.sim_rotation
        self.lin_vel = db.bone_velocities[frame_idx,0]*self.scale
        self.ang_vel = db.bone_angular_velocities[frame_idx,0]
        for i in range(self.n_bones):
            self.fk(i)

    def update_sim(self, dt):
        delta_pos  = quat.mul_vec(self.sim_rotation, self.lin_vel*dt)
        self.sim_position += delta_pos
        global_ang_vel = quat.mul_vec(self.sim_rotation, self.ang_vel*dt)
        delta_rot = quat.from_euler(global_ang_vel)
        self.sim_rotation = quat.mul(self.sim_rotation, delta_rot)

    def fk(self, bone_idx):
        if self.parents[bone_idx] != -1:
            parent_idx = self.parents[bone_idx]
            #self.fk(parent_idx)
            parent_m = quaternion_matrix(self.global_rotations[parent_idx])
            bone_m = quaternion_matrix(self.rotations[bone_idx])
            global_rot = np.dot(parent_m, bone_m)
            self.global_positions[bone_idx] = self.global_positions[parent_idx] + np.dot(parent_m[:3, :3], self.positions[bone_idx])
            
            self.global_rotations[bone_idx] = quaternion_from_matrix(global_rot)

        else:
            self.global_positions[bone_idx] = self.positions[bone_idx]
            self.global_rotations[bone_idx] = self.rotations[bone_idx]