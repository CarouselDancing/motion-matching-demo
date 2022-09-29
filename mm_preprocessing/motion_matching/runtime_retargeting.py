
import numpy as np
from vis_utils.scene.scene_object_builder import SceneObjectBuilder
from vis_utils.scene.components import ComponentBase
from anim_utils.retargeting.analytical import Retargeting, generate_joint_map


class RuntimeRetargeting(ComponentBase):
    def __init__(self, scene_object, target_skeleton, target_controller):
        ComponentBase.__init__(self, scene_object)
        self.anim_src = None
        self.src_skeleton = None
        self.target_skeleton = target_skeleton
        self.target_controller = target_controller

    def set_animation_src(self, src_skeleton, anim_src):
        self.src_skeleton = src_skeleton
        self.anim_src = anim_src
        joint_map = generate_joint_map(src_skeleton.skeleton_model, self.target_skeleton.skeleton_model)
        self.retargeting = Retargeting(src_skeleton, self.target_skeleton, joint_map)

    def update(self, dt):
        if self.retargeting is None or self.anim_src is None:
            return
        ref_frame = self.target_skeleton.reference_frame
        src_frame = self.anim_src.get_pose()
        self.target_frame = self.retargeting.retarget_frame(src_frame, ref_frame)
        self.target_controller.replace_current_frame(self.target_frame)


def create_runtime_retargeting(builder, scene_object, target_skeleton, target_controller):
    retargeting= RuntimeRetargeting(scene_object, target_skeleton, target_controller)
    scene_object.add_component("runtime_retargeting", retargeting)
    return retargeting


SceneObjectBuilder.register_component("runtime_retargeting", create_runtime_retargeting)
