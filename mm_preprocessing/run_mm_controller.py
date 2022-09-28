import os
import json
from vis_utils.glut_app import GLUTApp
from vis_utils.scene.task_manager import Task
from anim_utils.animation_data import BVHReader, SkeletonBuilder
import mm_controller_component

MODEL_DATA_PATH = "data"+os.sep+"models"

def print_global_vars(dt, app):
    scene = app.scene
    lines = []
    for key in scene.global_vars:
        value = str(scene.global_vars[key])
        lines.append(key+": "+value)
    app.set_console_lines(lines)


    
def control_func(key, params):
    app, controller = params
    if key == str.encode(" "):
        controller.toggleAnimation()
    controller.handle_keyboard_input(key)

    
def load_json_file(path):
    with open(path, "rt") as in_file:
        return json.load(in_file)

def load_skeleton_model(skeleton_type):
    skeleton_model = dict()
    path = MODEL_DATA_PATH + os.sep+skeleton_type+".json"
    if os.path.isfile(path):
        data = load_json_file(path)
        skeleton_model = data["model"]
    else:
        print("Error: model unknown", path)
    return skeleton_model 



def load_skeleton(path, skeleton_type=None):
    bvh = BVHReader(path)   
    skeleton = SkeletonBuilder().load_from_bvh(bvh)
    skeleton.skeleton_model = load_skeleton_model(skeleton_type)
    return skeleton
    
class RetargetingConfig:
    src_type : str
    dst_type :str


default_config = RetargetingConfig()
default_config.dst_type = "mh_cmu"
default_config.src_type = "raw"


def main(mm_filename, src_skeleton_filename=None, mesh_filename=None, retargeting_config=default_config):
    c_pose = dict()
    c_pose["zoom"] = -50
    c_pose["position"] = [0, 0, -5]
    c_pose["angles"] = (45, 200)
    app = GLUTApp(800, 600, title="mm player", camera_pose=c_pose)
    o = app.scene.object_builder.create_object_from_file("npz.txt", mm_filename)
    c = o._components["animation_controller"]

    if mesh_filename is not None:
        mo = app.scene.object_builder.create_object_from_file("fbx_model",mesh_filename, scale=10)
        target_controller = mo._components["animation_controller"]
        target_skeleton = target_controller._visualization.skeleton
        target_skeleton.skeleton_model = load_skeleton_model(retargeting_config.dst_type)
        src_skeleton = load_skeleton(src_skeleton_filename, retargeting_config.src_type)
        c.set_animator_target(target_controller, src_skeleton)

    app.keyboard_handler["control"] = (control_func, (app, c))
    
    app.scene.draw_task_manager.add("print", Task("print", print_global_vars, app))
    app.run()

if __name__ == "__main__":
    DATA_DIR = r"D:\Research\Carousel\data"
    motion_path = DATA_DIR + os.sep + r"m11\retarget\raw\combined"
    out_path = "D:\Research\Carousel\workspace\motion_matching_demo\mm_demo\Assets\Resources"
    mm_filename = out_path + os.sep + "database_merengue_raw20.npz.txt"
    src_skeleton=r"data\raw.bvh"
    mesh_filename = r"data\model3_cmu.fbx"
    default_config.dst_type = "mh_cmu"
    default_config.src_type = "raw"
    main(mm_filename, src_skeleton, mesh_filename)
