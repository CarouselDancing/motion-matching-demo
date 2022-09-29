
import numpy as np
from mm_pose import MMPose

class MMController():
    def __init__(self) -> None:
        self.target_dir = np.array([0, 0, 1])
        self.max_step_length = 10
        self.search_interval = 1#0.05
        self.force_search_timer =0
        self.mm_database = None
        self.pose = None
        self.n_frames = 0
        self.frame_idx = 0
        self.frame_time = 1/30

        self.target_skeleton = None
        self.retargeting = None
        self.target_controller = None
        self.update_timer = 0
        self.use_velocity = True

    def set_data(self, mm_database, scale=1.0):
        self.mm_database = mm_database
        self.pose = MMPose.from_db(mm_database, 0, scale)
        self.frame_time = 1.0/self.mm_database.fps
        self.n_frames = len(mm_database.bone_positions)



    def set_animator_target(self, target_controller, src_skeleton):
        from anim_utils.retargeting.analytical import Retargeting, generate_joint_map

        self.target_controller = target_controller
        self.target_skeleton = target_controller.get_skeleton()
        joint_map = generate_joint_map(src_skeleton.skeleton_model, self.target_skeleton.skeleton_model)
        self.retargeting = Retargeting(src_skeleton, self.target_skeleton, joint_map)


    def update(self, dt):
        if self.force_search_timer > 0:
            self.force_search_timer -= dt
        #print(self.force_search_timer)
        if self.frame_idx >= self.n_frames or self.force_search_timer <= 0:
            self.find_transition()
            self.force_search_timer = self.search_interval
        if self.update_timer >= 0:
            self.update_timer -= dt
        if self.update_timer <=0:
            self.frame_idx +=1
        if self.frame_idx >= self.n_frames:
            self.frame_idx =0
        self.pose.set_pose(self.mm_database, self.frame_idx)
        self.pose.update_sim(dt)
        return self.frame_idx


    def find_transition(self):
        next_frame_idx = min(self.frame_idx + 1, self.n_frames-1)
        self.frame_idx = self.mm_database.find_transition(self.pose, next_frame_idx)
        if self.frame_idx > self.n_frames:
            self.frame_idx = 0

    def update_w(self, dt):
        if self.force_search_timer > 0:
            self.force_search_timer -= dt
        self.pose.update_sim(dt)
        if self.frame_idx >= self.n_frames or self.force_search_timer <= 0:
            self.find_transition()
            self.force_search_timer = self.search_interval
        self.animation_time += dt
        self.frame_idx = int(self.animation_time / self.frame_time)
        self.pose.set_pose(self.mm_database, self.frame_idx)
        return self.frame_idx, self.animation_time

    def get_pose(self):
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
        src_frame = self.get_pose()
        target_frame = self.retargeting.retarget_frame(src_frame, ref_frame)
        return target_frame

    def reset(self):
        self.frame_idx = self.start_frame_idx
        self.update_timer = 0.0

    def get_phase(self):
        return self.mm_database.phase_data[self.frame_idx]

    def rotate_dir_vector(self, angle):
        r = np.radians(angle)
        s = np.sin(r)
        c = np.cos(r)
        self.target_dir = np.array(self.target_dir, float)
        self.target_dir[0] = c * self.target_dir[0] - s * self.target_dir[2]
        self.target_dir[2] = s * self.target_dir[0] + c * self.target_dir[2]
        self.target_dir /= float(np.linalg.norm(self.target_dir))
        print("rotate", self.target_dir)