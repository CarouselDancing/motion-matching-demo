
import numpy as np
from motion_database import MotionDatabase
from mm_pose import MMPose


class MMController():
    def __init__(self) -> None:
        self.target_dir = np.array([0, 0, 1])
        self.max_step_length = 10
        self.search_interval = 0.05
        self.force_search_timer = 0.05
        self.mm_database = None
        self.pose = None
        self.n_frames = 0
        self.frame_idx = 0
        self.animation_time = 0
        self.frame_time = 1/30

    def set_data(self, mm_database):
        self.mm_database = mm_database
        self.pose = MMPose.from_db(mm_database, 0)
        self.frame_time = 1.0/self.mm_database.fps
        self.n_frames = len(mm_database.bone_positions)

    def find_transition(self):
        self.frame_idx = self.mm_database.find_transition(self.pose, self.frame_idx)
        print(self.frame_idx)
        if self.frame_idx > self.n_frames:
            self.frame_idx = 0

    def update(self, dt):
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

