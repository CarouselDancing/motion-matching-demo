import os
from vis_utils.glut_app import GLUTApp
from vis_utils.scene.task_manager import Task
import mm_controller


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


def main(src_filename):
    c_pose = dict()
    c_pose["zoom"] = -50
    c_pose["position"] = [0, 0, -5]
    c_pose["angles"] = (45, 200)
    app = GLUTApp(800, 600, title="mm player", camera_pose=c_pose)
    o = app.scene.object_builder.create_object_from_file("npz.txt", src_filename)
    c = o._components["animation_controller"]
    app.keyboard_handler["control"] = (control_func, (app, c))
    
    app.scene.draw_task_manager.add("print", Task("print", print_global_vars, app))
    app.run()

if __name__ == "__main__":
    DATA_DIR = r"D:\Research\Carousel\data"
    motion_path = DATA_DIR + os.sep + r"m11\retarget\raw\combined"
    out_path = "D:\Research\Carousel\workspace\motion_matching_demo\mm_demo\Assets\Resources"
    src_filename = out_path + os.sep + "database_merengue_raw20.npz.txt"
    main(src_filename)
