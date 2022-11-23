import sys
import os
sys.path.append(os.sep.join([".."])+os.sep)
import numpy as np
from vis_utils.glut_app import GLUTApp
from vis_utils.scene.task_manager import Task
import motion_matching.mm_controller_visualization
from motion_matching.utils import load_skeleton, load_skeleton_model


def print_global_vars(dt, app):
    scene = app.scene
    lines = []
    for key in scene.global_vars:
        value = str(scene.global_vars[key])
        lines.append(key+": "+value)
    app.set_console_lines(lines)


    
def control_func(key, params):
    app, controller, mesh = params
    if key == str.encode(" "):
        controller.toggleAnimation()
    if key == str.encode("m") and mesh is not None:
        mesh.visible = not mesh.visible
    controller.handle_keyboard_input(key)

    
class RetargetingConfig:
    src_type : str
    dst_type :str


default_config = RetargetingConfig()
default_config.dst_type = "mh_cmu"
default_config.src_type = "raw"


def main(mm_filename, src_skeleton_filename=None, mesh_filename=None, retargeting_config=default_config, use_batch=False):
    c_pose = dict()
    c_pose["zoom"] = -50
    c_pose["position"] = [0, 0, -5]
    c_pose["angles"] = (45, 200)
    app = GLUTApp(800, 600, title="mm player", camera_pose=c_pose)
    scale = 1
    if mesh_filename is not None:
        scale = 5

    o = app.scene.object_builder.create_object_from_file("npz.txt", mm_filename, scale, use_batch=use_batch)
    c = o._components["animation_controller"]
    mesh = None
    if mesh_filename is not None:
        mo = app.scene.object_builder.create_object_from_file("fbx_model",mesh_filename, scale=0.05)#0.5)
        target_controller = mo._components["animation_controller"]
        target_skeleton = target_controller._visualization.skeleton
        target_skeleton.skeleton_model = load_skeleton_model(retargeting_config.dst_type)
        #target_skeleton.skeleton_model["cos_map"]["Hips"]["x"] = -np.array(target_skeleton.skeleton_model["cos_map"]["Hips"]["x"])
        
        
        src_skeleton = load_skeleton(src_skeleton_filename, retargeting_config.src_type)
        #src_skeleton.skeleton_model["cos_map"]["Hips"]["x"] = -np.array(src_skeleton.skeleton_model["cos_map"]["Hips"]["x"])
        c.set_animator_target(target_controller, src_skeleton)
        mesh = mo._components["animated_mesh"]
        

    app.keyboard_handler["control"] = (control_func, (app, c, mesh))
    
    c.toggleAnimation()
    app.scene.draw_task_manager.add("print", Task("print", print_global_vars, app))
    app.run()

if __name__ == "__main__":
    out_path = "D:\Research\Carousel\workspace\motion_matching_demo\mm_demo\Assets\Resources"
    mm_filename = out_path + os.sep + "database_merengue_raw20.npz.txt"
    mm_filename = r"D:\Research\physics\workspace\mi_custom_envs\custom_envs\data\imitation\reference_motion\database_merengue_raw20_gl.npz.txt"
    
    out_path = "..\data"
    mm_filename = out_path + os.sep + "database_merengue_fig_bodies_and_joints.npz"
    mm_filename = out_path + os.sep + "database_merengue_amp_humanoid.npz"
    mm_filename = out_path + os.sep + "database_merengue_raw20_gl.npz.txt"
    src_skeleton=r"..\data\raw.bvh"
    mesh_filename = r"..\data\model3_cmu.fbx"
    mesh_filename = r"..\data\model7.fbx"
    mesh_filename = None
    default_config.dst_type = "mh_cmu"
    default_config.src_type = "raw"
    main(mm_filename, src_skeleton, mesh_filename, use_batch=False)
