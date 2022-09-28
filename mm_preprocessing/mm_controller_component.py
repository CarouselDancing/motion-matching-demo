
import numpy as np
from PySignal import Signal
from motion_database import MotionDatabase
from mm_controller import MMController
from vis_utils.scene.scene_object_builder import SceneObject, SceneObjectBuilder
from vis_utils.scene.components import ComponentBase
from vis_utils.animation.animation_controller import AnimationController
from vis_utils.graphics.renderer import SphereRenderer
from vis_utils.graphics.renderer.lines import DebugLineRenderer
from vis_utils.graphics.utils import get_translation_matrix
from vis_utils.scene.utils import get_random_color
DEFAULT_COLOR = [0, 0, 1]


class MMControllerComponent(ComponentBase, AnimationController):
    updated_animation_frame = Signal()
    reached_end_of_animation = Signal()
    def __init__(self, scene_object, color=DEFAULT_COLOR, visualize=True):
        ComponentBase.__init__(self, scene_object)
        AnimationController.__init__(self)        
        self.visualize = visualize
        self.skeleton = None
        if visualize:
            self._sphere = SphereRenderer(10, 10, 1, color=color)
            a = [0, 0, 0]
            b = [1, 0, 0]
            self._line = DebugLineRenderer(a, b, [0, 1, 0])
        self.debug_line_len = 1
        self.mm_controller = MMController()

    def toggle_animation_loop(self):
        self.loopAnimation = not self.loopAnimation

    def isLoadedCorrectly(self):
        return self.mm_controller.mm_database is not None

    def set_data(self, mm_database):
        self.mm_controller.set_data(mm_database)

    def set_animator_target(self, target_controller, src_skeleton):
        self.mm_controller.set_animator_target(target_controller, src_skeleton)

    def update(self, dt):
        pos = np.array(self.mm_controller.pose.sim_position[:])
        pos[1] = 5
        target = pos + self.mm_controller.target_dir * self.debug_line_len
        self._line.set_line(pos, target)
        if not self.isLoadedCorrectly() or not self.playAnimation:
            return
        self.currentFrameNumber, self.animationTime = self.mm_controller.update(dt* self.animationSpeed)
        self.mm_controller.update_target()
        # update gui
        if self.currentFrameNumber > self.getNumberOfFrames():
            self.resetAnimationTime()
        else:
            self.updated_animation_frame.emit(self.currentFrameNumber)

    def draw(self, modelMatrix, viewMatrix, projectionMatrix, lightSources):
        if self._sphere is None:
                return
        if self.currentFrameNumber < 0 or self.currentFrameNumber >= self.getNumberOfFrames():
            return

        for position in self.mm_controller.pose.global_positions:
            m = get_translation_matrix(position[:3])
            m = np.dot(m, modelMatrix)
            self._sphere.draw(m, viewMatrix, projectionMatrix, lightSources)
        self._line.draw(modelMatrix, viewMatrix, projectionMatrix)


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
        r = np.radians(angle)
        s = np.sin(r)
        c = np.cos(r)
        self.target_dir = np.array(self.target_dir, float)
        self.target_dir[0] = c * self.target_dir[0] - s * self.target_dir[2]
        self.target_dir[2] = s * self.target_dir[0] + c * self.target_dir[2]
        self.target_dir /= float(np.linalg.norm(self.target_dir))
        print("rotate", self.target_dir)

   

    def handle_keyboard_input(self, key):
        print("handle", key)
        if key == b"a":
            self.rotate_dir_vector(-10)
        elif key == b"d":
            self.rotate_dir_vector(10)
        elif key == b"w":
            self.debug_line_len += 1
            self.debug_line_len = min(self.debug_line_len, self.max_step_length)
        elif key == b"s":
            self.debug_line_len -= 1
            self.debug_line_len = max(self.debug_line_len, 0)
    def get_pose(self):
        return self.mm_controller.get_pose()


def load_mm_database(builder, filename):
    name = filename.split("/")[-1]
    scene_object = SceneObject()
    scene_object.name = name
    db = MotionDatabase()
    db.load(filename)
    animation_controller = MMControllerComponent(scene_object, color=get_random_color())
    animation_controller.set_data(db)
    scene_object.add_component("animation_controller", animation_controller)
    controller = scene_object._components["animation_controller"]
    builder._scene.addAnimationController(scene_object, "animation_controller")
    return scene_object

def load_mm_database_numpy(builder, filename):
    name = filename.split("/")[-1]
    scene_object = SceneObject()
    scene_object.name = name
    db = MotionDatabase()
    db.load_from_numpy(filename)
    animation_controller = MMControllerComponent(scene_object, color=get_random_color())
    animation_controller.set_data(db)
    scene_object.add_component("animation_controller", animation_controller)
    controller = scene_object._components["animation_controller"]
    builder._scene.addAnimationController(scene_object, "animation_controller")
    return scene_object
    

SceneObjectBuilder.register_file_handler("bin.txt", load_mm_database)
SceneObjectBuilder.register_file_handler("npz.txt", load_mm_database_numpy)
