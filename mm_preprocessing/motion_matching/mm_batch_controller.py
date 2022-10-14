
import numpy as np

from .mm_controller import MMController
from .mm_pose import MMPose


class MMBatchController(MMController):
    def __init__(self, num_envs, env_offset) -> None:
        self.num_envs = num_envs
        self.target_dir = np.array([0, 0, 1])
        self.max_step_length = 10
        self.search_interval = 1#0.05
        self.force_search_timer =0
        self.mm_database = None
        self.env_poses = []
        self.n_frames = 0
        self.frame_time = 1/30
        self.n_bones = 0
        self.env_offset = env_offset
        self.envs_per_row = 5

    def set_data(self, mm_database, scale=1.0):
        self.mm_database = mm_database
        for i in range(self.num_envs):
            x = i // self.envs_per_row
            z = i %  self.envs_per_row
            offset = np.array([x, 0, z]) * self.env_offset
            pose = MMPose.from_db(mm_database, 0, scale, offset)
            self.env_poses.append(pose)
            
        self.frame_time = 1.0/self.mm_database.fps
        self.n_frames = len(mm_database.bone_positions)
        self.n_bones = len(mm_database.bone_parents)

    @property
    def pose(self):
        return self.env_poses[0]

    def update(self, dt):
        #print(self.force_search_timer)
        for pose in self.env_poses:
            self.find_transition(pose, dt)
            self.step_frame(pose, dt, self.mm_database)
        return self.env_poses[0].frame_idx

    def get_pose(self):
        n_dims = (self.n_bones-1)*4 +3
        frame = np.zeros((self.num_envs, n_dims))
        for i, pose in enumerate(self.env_poses):
            pose.update_fk_buffer()
            frame[i,:3] = pose.global_positions[1]
            frame[i,3:7] = pose.global_rotations[1]
            o = 7
            for j in range(2, pose.n_bones):
                frame[i,o:o+4] = pose.rotations[j]
                o += 4
        return frame[0]

    def get_pose_sim_pos(self):
        """ return sim position """
        n_dims = (self.n_bones-1)*4 +3
        frame = np.zeros((self.num_envs, n_dims))
        for i, pose in enumerate(self.env_poses):
            pose.update_fk_buffer()
            frame[i,:3] = pose.sim_position
            frame[i,3:7] = pose.sim_rotation
            o = 7
            for j in range(2, pose.n_bones):
                frame[i,o:o+4] = pose.rotations[j]
                o += 4
        return frame[0]

    def reset(self, frame_idx):
        for i, pose in enumerate(self.env_poses):
            pose.reset(frame_idx)

    def reset_idx(self, env_indices, frame_idx):
        for i in env_indices:
            self.env_poses[i].reset(frame_idx)

    def get_phase(self):
        phase = np.zeros((self.num_envs))
        for i, pose in enumerate(self.env_poses):
            phase[i] = self.mm_database.phase_data[pose.frame_idx]
        return phase
        
    def rotate_dir_vector(self, angle):
        r = np.radians(angle)
        s = np.sin(r)
        c = np.cos(r)
        self.target_dir = np.array(self.target_dir, float)
        self.target_dir[0] = c * self.target_dir[0] - s * self.target_dir[2]
        self.target_dir[2] = s * self.target_dir[0] + c * self.target_dir[2]
        self.target_dir /= float(np.linalg.norm(self.target_dir))
        print("rotate", self.target_dir)
        

    def get_relative_bone_velocity(self, bone_idx):
        bone_velocity = np.zeros((self.num_envs, 3))
        for i, pose in enumerate(self.env_poses):
            bone_velocity[i] = self.mm_database.get_relative_bone_velocity(pose.frame_idx, bone_idx)
        return bone_velocity

    def get_relative_bone_velocities_frame(self):
        n_dims = (self.n_bones-1)*3
        velicity_frame = np.zeros((self.num_envs, n_dims))
        for i, pose in enumerate(self.env_poses):
            velicity_frame[i] = self.mm_database.get_relative_bone_velocities_frame(pose.frame_idx)
        return velicity_frame

    def get_angular_velocity_frame(self):
        n_dims = self.mm_database.bone_angular_velocities.shape[1]-3#ignore sim bone
        angular_velocity_frame = np.zeros((self.num_envs, n_dims))
        for i, pose in enumerate(self.env_poses):
            angular_velocity_frame[i] = self.mm_database.bone_angular_velocities[pose.frame_idx, 1:].flatten()
        return angular_velocity_frame/self.dt

    def get_root_velocity(self):
        linear_velocity = np.zeros((self.num_envs, 3))
        angular_velocity = np.zeros((self.num_envs, 3))
        for i, pose in enumerate(self.env_poses):
            linear_velocity[i] = self.mm_database.bone_velocities[pose.frame_idx][1]#ignore sim bone
            angular_velocity[i] = self.mm_database.bone_angular_velocities[pose.frame_idx][1]#ignore sim bone
        return linear_velocity/self.dt, angular_velocity/self.dt

    def update_target(self):
        return
