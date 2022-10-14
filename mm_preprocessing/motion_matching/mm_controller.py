
import numpy as np
from .mm_pose import MMPose

class MMController():
    def __init__(self) -> None:
        self.target_dir = np.array([0, 0, 1])
        self.offset = [0,0.1,0]
        self.max_step_length = 10
        self.search_interval = 1#0.05
        self.force_search_timer =0
        self.mm_database = None
        self.pose = None
        self.n_frames = 0

        self.target_skeleton = None
        self.retargeting = None
        self.target_controller = None
        self.frame_time = 1/30
        self.n_frames=0

    def set_data(self, mm_database, scale=1.0):
        self.mm_database = mm_database
        self.pose = MMPose.from_db(mm_database, 0, scale, self.offset)
        self.frame_time = 1.0/mm_database.fps
        self.n_frames = len(mm_database.bone_positions)



    def set_animator_target(self, target_controller, src_skeleton):
        from anim_utils.retargeting.analytical import Retargeting, generate_joint_map

        self.target_controller = target_controller
        self.target_skeleton = target_controller.get_skeleton()
        joint_map = generate_joint_map(src_skeleton.skeleton_model, self.target_skeleton.skeleton_model)
        self.retargeting = Retargeting(src_skeleton, self.target_skeleton, joint_map)
   
    def update(self, dt):
        self.find_transition(self.pose, dt)
        self.step_frame(self.pose, dt, self.mm_database)
        return self.pose.frame_idx
        
    def find_transition(self, pose, dt):
        if pose.force_search_timer > 0:
            pose.force_search_timer -= dt
        if pose.frame_idx >= self.n_frames or pose.force_search_timer <= 0:
            next_frame_idx = min(pose.frame_idx + 1, self.n_frames-1)
            pose.frame_idx = self.mm_database.find_transition(pose, next_frame_idx)
            if pose.frame_idx > self.n_frames:
                pose.frame_idx = 0
            pose.force_search_timer = self.search_interval


    def step_frame(self, pose, dt, mm_database):
        if pose.update_timer >= 0:
            pose.update_timer -= dt
        if pose.update_timer <=0:
            pose.frame_idx +=1
            pose.update_timer = self.frame_time        
        if pose.frame_idx > self.n_frames:
            pose.frame_idx = 0
        pose.set_pose(mm_database, pose.frame_idx)
        pose.update_sim(dt)

    def get_pose(self):
        self.pose.update_fk_buffer()
        n_dims = (self.pose.n_bones-1)*4 +3
        frame = np.zeros(n_dims)
        frame[:3] = self.pose.global_positions[1]
        frame[3:7] = self.pose.global_rotations[1]
        o = 7
        for i in range(2, self.pose.n_bones):
            frame[o:o+4] = self.pose.rotations[i]
            o += 4
        return frame

    def get_pose_sim_pos(self):
        """ return sim position """
        self.pose.update_fk_buffer()
        n_dims = (self.pose.n_bones-1)*4 +3
        frame = np.zeros(n_dims)
        frame[:3] = self.pose.sim_position
        frame[3:7] = self.pose.sim_rotation
        o = 7
        for i in range(2, self.pose.n_bones):
            frame[o:o+4] = self.pose.rotations[i]
            o += 4
        return frame

    def update_target(self):
        if self.retargeting is None or self.target_skeleton is None:
            return
        self.target_frame = self.get_retargeted_pose()
        self.target_controller.replace_current_frame(self.target_frame)

    def get_retargeted_pose(self):
        ref_frame = self.target_skeleton.reference_frame
        src_frame = self.get_pose_sim_pos()
        target_frame = self.retargeting.retarget_frame(src_frame, ref_frame)
        return target_frame

    def reset(self, frame_idx):
        self.pose.reset(frame_idx)

    def get_phase(self):
        return self.mm_database.phase_data[self.pose.frame_idx]
        
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
        return self.mm_database.get_relative_bone_velocity(self.pose.frame_idx, bone_idx)

    def get_relative_bone_velocities_frame(self):
        return self.mm_database.get_relative_bone_velocities_frame(self.pose.frame_idx)

