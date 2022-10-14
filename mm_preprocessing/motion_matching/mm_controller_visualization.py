
import numpy as np
from PySignal import Signal
from .mm_database import MMDatabase
from .mm_database_binary_io import MMDatabaseBinaryIO
from .mm_controller import MMController
from .mm_batch_controller import MMBatchController
from vis_utils.scene.scene_object_builder import SceneObject, SceneObjectBuilder
from vis_utils.scene.components import ComponentBase
from vis_utils.animation.animation_controller import AnimationController
from vis_utils.graphics.renderer import SphereRenderer
from vis_utils.graphics.renderer.lines import DebugLineRenderer
from vis_utils.graphics.utils import get_translation_matrix
from vis_utils.scene.utils import get_random_color
from transformations import quaternion_matrix
import torch

DEFAULT_COLOR = [0, 0, 1]


class MMControllerVisualization(ComponentBase, AnimationController):
    updated_animation_frame = Signal()
    reached_end_of_animation = Signal()
    def __init__(self, scene_object, color=DEFAULT_COLOR, visualize=True, vis_scale=0.1, use_batch=False):
        ComponentBase.__init__(self, scene_object)
        AnimationController.__init__(self)        
        self.visualize = visualize
        self.skeleton = None
        if visualize:
            self._sphere = SphereRenderer(10, 10, vis_scale, color=color)
            a = [0, 0, 0]
            b = [1, 0, 0]
            self._line = DebugLineRenderer(a, b, [0, 1, 0])
            self._vel_line = DebugLineRenderer(a, b, [1, 0, 0])
            self._ref_vel_line = DebugLineRenderer(a, b, [0, 0,1])
        self.debug_line_len = 5
        if use_batch:
            self.mm_controller = MMBatchController(10, [10,0,10])
        else:
            self.mm_controller = MMController()
        self.use_batch = use_batch
        self.enable_velocity_visualization = True
        self.velocity_scale = 1
        self.figure_c = None

    def toggle_animation_loop(self):
        self.loopAnimation = not self.loopAnimation

    def isLoadedCorrectly(self):
        return self.mm_controller.mm_database is not None

    def set_data(self, mm_database, scale=1.0):
        self.mm_controller.set_data(mm_database, scale)

    def set_animator_target(self, target_controller, src_skeleton):
        self.mm_controller.set_animator_target(target_controller, src_skeleton)

    def update_debug_vis(self):
        pos = np.array(self.mm_controller.pose.sim_position[:])
        pos[1] = 5
        target = pos + self.mm_controller.target_dir * self.debug_line_len
        self._line.set_line(pos, target)


        target = pos + self.mm_controller.pose.get_linear_velocity() * self.debug_line_len
        self._vel_line.set_line(pos, target)

        pos = np.array(self.mm_controller.pose.ref_position[:])
        target = pos + self.mm_controller.pose.get_ref_linear_velocity() * self.debug_line_len
        self._ref_vel_line.set_line(pos, target)
        return

    def update(self, dt):
        self.velocity_scale  = dt
        self.update_debug_vis()
        if not self.isLoadedCorrectly() or not self.playAnimation:
            return
        self.currentFrameNumber = self.mm_controller.update(dt* self.animationSpeed)
        
        self.animationTime += dt* self.animationSpeed
        self.mm_controller.update_target()
        # update gui
        if self.currentFrameNumber > self.getNumberOfFrames():
            self.resetAnimationTime()
        else:
            self.updated_animation_frame.emit(self.currentFrameNumber)

    def draw(self, model, view, projection, lights):
        if self._sphere is None:
                return
        if self.use_batch:
            for pose in self.mm_controller.env_poses:
                self.draw_pose(pose, model, view, projection, lights)
        else:
            self.draw_pose(self.mm_controller.pose,  model, view, projection, lights)
        self._line.draw(model, view, projection)
        self._vel_line.draw(model, view, projection)
        self._ref_vel_line.draw(model, view, projection)
        if self.enable_velocity_visualization:
            self.draw_velocity(model, view, projection)
            if self.figure_c is not None:
                self.draw_figure_velocity(model, view, projection)

    def draw_pose(self, pose, m, v, p, lights):#
        for position in pose.global_positions:
            tm = get_translation_matrix(position[:3])
            self._sphere.draw(np.dot(tm, m), v, p, lights)
        return

    def draw_velocity(self, m, v, p):
        root_m = quaternion_matrix(self.mm_controller.pose.sim_rotation)[:3,:3]
        frame = self.mm_controller.mm_database.get_relative_bone_velocities_frame(self.currentFrameNumber)
        for i, position in  enumerate(self.mm_controller.pose.global_positions):
            lv = frame[i]
            lv = np.dot(root_m, lv)
            #print(self.mm_controller.mm_database.bone_names[i], lv)
            self._ref_vel_line.set_line(position, position-lv*self.velocity_scale)
            self._ref_vel_line.draw(m, v, p)

    def draw_figure_velocity(self, m, v, p):
        for b in self.figure_c.target_figure.bodies:
            position = np.array(self.figure_c.target_figure.bodies[b].get_position())
            lv = np.array(self.figure_c.target_figure.bodies[b].get_linear_velocity())
            self._vel_line.set_line(position, position-lv*self.velocity_scale)
            self._vel_line.draw(m, v, p)


    def getNumberOfFrames(self):
        return self.mm_controller.n_frames

    def getFrameTime(self):
        return self.mm_controller.frame_time

    def updateTransformation(self, frame_number=None):
        if frame_number is not None:
            self.currentFrameNumber = frame_number
        self.animationTime = self.getFrameTime() * self.currentFrameNumber

    def setCurrentFrameNumber(self, frame_number=None):
        if frame_number is not None:
            self.currentFrameNumber = frame_number
        self.animationTime = self.getFrameTime() * self.currentFrameNumber


    def setColor(self, color):
        self._sphere.technique.material.diffuse_color = color
        self._sphere.technique.material.ambient_color = color * 0.1

    def getColor(self):
        return self._sphere.technique.material.diffuse_color


    def get_current_frame(self):
        return self.pose.fk_positions

    def get_skeleton(self):
        return self.skeleton

    def get_frame_time(self):
        return self.getFrameTime()

    def get_label_color_map(self):
        return dict()

    def set_frame_time(self, frame_time):
        self.frameTime = frame_time
    def get_semantic_annotation(self):
        return dict()

    def rotate_dir_vector(self, angle):
        self.mm_controller.rotate_dir_vector(angle)

   
    def handle_keyboard_input(self, key):
        print("handle", key)
        if key == b"a":
            self.rotate_dir_vector(-10)
        elif key == b"d":
            self.rotate_dir_vector(10)
        elif key == b"w":
            self.debug_line_len += 1
            self.debug_line_len = min(self.debug_line_len, self.mm_controller.max_step_length)
        elif key == b"s":
            self.debug_line_len -= 1
            self.debug_line_len = max(self.debug_line_len, 0)
        elif key == b"v":
            self.toggle_velocity()

    def get_pose(self):
        return self.mm_controller.get_pose()

    def toggle_velocity(self):
        self.mm_controller.pose.use_velocity = not self.mm_controller.pose.use_velocity
        print("use_velocity",self.mm_controller.use_velocity)


def load_mm_database(builder, filename, scale=1, use_batch=False):
    name = filename.split("/")[-1]
    scene_object = SceneObject()
    scene_object.name = name
    db = MMDatabaseBinaryIO.load(filename)
    animation_controller = MMControllerVisualization(scene_object, color=get_random_color(), use_batch=use_batch)
    animation_controller.set_data(db, scale)
    scene_object.add_component("animation_controller", animation_controller)
    controller = scene_object._components["animation_controller"]
    builder._scene.addAnimationController(scene_object, "animation_controller")
    return scene_object

def load_mm_database_numpy(builder, filename, scale=1, use_batch=False):
    name = filename.split("/")[-1]
    scene_object = SceneObject()
    scene_object.name = name
    db = MMDatabase()
    db.load_from_numpy(filename)
    animation_controller = MMControllerVisualization(scene_object, color=get_random_color(), use_batch=use_batch)
    animation_controller.set_data(db, scale)
    scene_object.add_component("animation_controller", animation_controller)
    controller = scene_object._components["animation_controller"]
    builder._scene.addAnimationController(scene_object, "animation_controller")
    return scene_object
    

    
SceneObjectBuilder.register_file_handler("bin.txt", load_mm_database)
SceneObjectBuilder.register_file_handler("npz.txt", load_mm_database_numpy)
