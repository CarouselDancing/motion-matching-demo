""" wrapper for poselib used by ASE
"""
import numpy as np
import xml.etree.ElementTree as ET
from motion_matching import quat
from transformations import quaternion_from_matrix
from anim_utils.animation_data import MotionVector
from anim_utils.animation_data.skeleton import Skeleton
from anim_utils.animation_data.skeleton_builder import SkeletonBuilder, SkeletonRootNode, SkeletonJointNode, SkeletonEndSiteNode
import scipy.ndimage.filters as filters


def load_skeleton_from_mjcf(file_path, frame_time=1.0/30, convert_cs=False): 
    skeleton = Skeleton()
    skeleton.frame_time = frame_time
    fk_joint_order = []
    t_m = np.array([[1,0,0],[0,0,1],[0,1,0]])
    def add_joint_for_body(body, parent=None):
        b_name = body.attrib["name"]
        offset = np.fromstring(body.attrib.get("pos"), dtype=float, sep=" ")
        if convert_cs:
            offset= np.dot(t_m, offset)
        if parent is None:
            level = 0
            channels = ["Xposition","Yposition","Zposition", "Xrotation","Yrotation","Zrotation"]
            node = SkeletonRootNode(b_name, channels, parent, level)
            skeleton.root = b_name
        else:
            
            level = parent.level + 1
            channels = ["Xrotation","Yrotation","Zrotation"]
            node = SkeletonJointNode(b_name, channels, parent, level)
            parent.children.append(node)
        
        node.rotation = np.array([1.0,0.0,0.0,0.0])
        node.offset = offset
        node.fixed = False
        node.index = len(fk_joint_order)
        node.quaternion_frame_index = node.index
        skeleton.nodes[b_name] = node
        fk_joint_order.append(b_name)
        children = body.findall('body')
        if len(children) > 0:
            for cbody_node in children:
                add_joint_for_body(cbody_node, node)
        else:
            end_site_name = b_name +"EndSite"
            channels = []
            end_site_node = SkeletonEndSiteNode(end_site_name, channels, node, level+1)
            end_site_node.quaternion_frame_index = -1
            end_site_node.index = -1
            end_site_node.fixed = True
            skeleton.nodes[b_name].children.append(end_site_node)
            skeleton.nodes[end_site_name] = end_site_node

    tree = ET.parse(file_path)
    root = tree.getroot()
    worldbody = root.findall('worldbody')[0]
    body_node = worldbody.findall('body')[0]
    add_joint_for_body(body_node, None)
    skeleton.animated_joints = fk_joint_order
    SkeletonBuilder.set_meta_info(skeleton)
    return skeleton



def skeleton_from_poselib(data, scale=1, convert_cs=False, frame_time=1.0/30):
    skeleton = Skeleton()
    skeleton.frame_time = frame_time
    t_m = np.array([[1,0,0],[0,0,1],[0,1,0]])
    fk_joint_order = data["node_names"]
    parent_indices = data["parent_indices"]["arr"]
    local_translation = data["local_translation"]["arr"]

    def add_joint_for_body(node_idx):
        name = fk_joint_order[node_idx]
        offset = local_translation[node_idx] * scale
        parent_idx = parent_indices[node_idx]
        if convert_cs:
            offset= np.dot(t_m, offset)
        if parent_idx == -1:
            level = 0
            channels = ["Xposition","Yposition","Zposition", "Xrotation","Yrotation","Zrotation"]
            node = SkeletonRootNode(name, channels, None, level)
            skeleton.root = name
        else:
            parent_name = fk_joint_order[parent_idx]
            parent = skeleton.nodes[parent_name]
            level = parent.level + 1
            channels = ["Xrotation","Yrotation","Zrotation"]
            node = SkeletonJointNode(name, channels, parent, level)
            parent.children.append(node)
        
        node.rotation = np.array([1.0,0.0,0.0,0.0])
        node.offset = offset
        node.fixed = False
        node.index = node_idx
        node.quaternion_frame_index = node_idx
        skeleton.nodes[name] = node
        if node_idx not in parent_indices:
            end_site_name = name +"EndSite"
            channels = []
            end_site_node = SkeletonEndSiteNode(end_site_name, channels, node, level+1)
            end_site_node.quaternion_frame_index = -1
            end_site_node.index = -1
            end_site_node.fixed = True
            skeleton.nodes[name].children.append(end_site_node)
            skeleton.nodes[end_site_name] = end_site_node
    for i in range(len(fk_joint_order)):
        add_joint_for_body(i)
    skeleton.animated_joints = fk_joint_order
    SkeletonBuilder.set_meta_info(skeleton)
    return skeleton

def mv_from_poselib_file(filename, scale=1, convert_cs=False, frame_time=1/30):
    data = np.load(filename, allow_pickle=True).item()
    mv = MotionVector()
    rt = data["root_translation"]["arr"]*scale
    rot = data["rotation"]["arr"]
    if len(rot.shape)> 2: # motion file
        n_frames = rot.shape[0]
        n_bones = rot.shape[1]
        n_dims = n_bones*rot.shape[2]
        rot = rot.reshape(n_frames, n_bones, rot.shape[2])
    else: # pose file
        n_frames = 1
        n_bones = rot.shape[0]
        n_dims = n_bones*rot.shape[1]
        rt = rt.reshape(n_frames, 3)
        rot = rot.reshape(n_frames, n_bones, rot.shape[1])
    rot_copy = np.array(rot)
    if convert_cs: #switch y and z axis and order of quaternion values
        rot[:,:,0]= rot_copy[:,:,3]
        rot[:,:,1]= -rot_copy[:,:,0]
        rot[:,:,2]= -rot_copy[:,:,2]
        rot[:,:,3]= -rot_copy[:,:,1]
        t_m = np.array([[1,0,0],[0,0,1],[0,1,0]])
        rt= [t_m.dot(t) for t in rt]
    else:
        #change order of quaternion values
        rot[:,:,0] = rot_copy[:,:,3]
        rot[:,:,1:] = rot_copy[:,:,:3]
    rot = rot.reshape(n_frames, n_dims)
    mv.frames = np.concatenate([rt, rot], axis=1)
    mv.n_frames = len(mv.frames)
    mv.frame_time = frame_time
    mv.skeleton = skeleton_from_poselib(data["skeleton_tree"], scale, convert_cs, frame_time)
    return mv


def add_array(data, name, array, dtype):
    data[name] = dict()
    data[name]["arr"] = np.array(array, dtype=dtype)
    data[name]["context"]= dict()
    data[name]["context"]["dtype"] = dtype
    return data


def mv_to_poselib(mv, scale=1.0, convert_cs=False, dtype=np.float64):
    data = dict()
    data["root_translation"] = dict()
    data["rotation"] = dict()
    n_bones = int((mv.frames.shape[1]-3) / 4)
    rt =  np.array(mv.frames[:,:3], dtype=dtype)
    rot = np.array(mv.frames[:,3:].reshape(mv.n_frames, n_bones, 4), dtype=dtype)
    rot_copy = np.array(rot)
    if convert_cs: #switch y and z axis and order of quaternion values
        rot[:,:,3]= rot_copy[:,:,0]
        rot[:,:,0]= -rot_copy[:,:,1]
        rot[:,:,2]= -rot_copy[:,:,2]
        rot[:,:,1]= -rot_copy[:,:,3]
        t_m = np.array([[1,0,0],[0,0,1],[0,1,0]])
        rt= np.array([t_m.dot(t) for t in rt],dtype=dtype)
    else:#change order of quaternion values
        rot[:,:,3] = rot_copy[:,:,0]
        rot[:,:,:3] = rot_copy[:,:,1:]
    data = add_array(data, "root_translation", rt*scale, dtype)
    data = add_array(data, "rotation", rot, dtype)
    global_positions, global_rotations = compute_global_transforms(mv)
    global_velocity, global_angular_velocity = compute_velocities_poselib(global_positions, global_rotations, mv.frame_time)
    global_velocity = global_velocity.reshape(mv.n_frames, n_bones, 3)
    global_angular_velocity = global_angular_velocity.reshape(mv.n_frames, n_bones, 3)
    data = add_array(data, "global_velocity", global_velocity*scale, dtype)
    data = add_array(data, "global_angular_velocity", global_angular_velocity, dtype)

    data["is_local"] = True
    data["fps"] = int(1.0/mv.frame_time)
    data["skeleton_tree"] = skeleton_to_poselib(mv.skeleton, convert_cs, scale, dtype)
    return data


def skeleton_to_poselib(skeleton, convert_cs=False, scale=1.0, dtype=np.float64):
    parent_indicies = [skeleton.nodes[j].parent.index if skeleton.nodes[j].parent is not None else -1 for j in skeleton.animated_joints]
    local_translation = np.array([skeleton.nodes[j].offset for j in skeleton.animated_joints], dtype=dtype)
    if convert_cs:
        t_m = np.array([[1,0,0],[0,0,1],[0,1,0]])
        local_translation = np.array([t_m.dot(t) for t in local_translation],dtype=dtype)
    data = dict()
    data["node_names"] = skeleton.animated_joints
    data = add_array(data, "parent_indices", parent_indicies,  np.int32)
    data = add_array(data, "local_translation", local_translation*scale, dtype)
    return data


def compute_global_transforms(mv):
    skeleton = mv.skeleton
    n_joints = len(skeleton.animated_joints)
    n_frames = len(mv.frames)
    global_positions = np.zeros((n_frames, n_joints, 3))
    global_rotations = np.zeros((n_frames, n_joints, 4))
    skeleton = mv.skeleton
    global_matrix_cache = dict()
    for i in range(n_frames):
        for j, joint_name in enumerate(skeleton.animated_joints):
            m = skeleton.nodes[joint_name].get_local_matrix(mv.frames[i])
            if skeleton.nodes[joint_name].parent is not None:
                parent_idx = skeleton.nodes[joint_name].parent.index
                m = np.dot(global_matrix_cache[parent_idx], m)
            global_positions[i][j] = m[:3, 3]
            global_rotations[i][j] = quaternion_from_matrix(m)
            global_matrix_cache[j] = m
    return global_positions, global_rotations


def compute_velocities(positions, rotations, fps):
    """ Compute velocities by Holden"""
    # Compute velocities via central difference
    velocities = np.empty_like(positions)
    velocities[1:-1] = (
        0.5 * (positions[2:  ] - positions[1:-1]) * fps +
        0.5 * (positions[1:-1] - positions[ :-2]) * fps)
    velocities[ 0] = velocities[ 1] - (velocities[ 3] - velocities[ 2])
    velocities[-1] = velocities[-2] + (velocities[-2] - velocities[-3])

    # Same for angular velocities
    angular_velocities = np.zeros_like(positions)
    angular_velocities[1:-1] = (
        0.5 * quat.to_scaled_angle_axis(quat.abs(quat.mul_inv(rotations[2:  ], rotations[1:-1]))) * fps +
        0.5 * quat.to_scaled_angle_axis(quat.abs(quat.mul_inv(rotations[1:-1], rotations[ :-2]))) * fps)
    angular_velocities[ 0] = angular_velocities[ 1] - (angular_velocities[ 3] - angular_velocities[ 2])
    angular_velocities[-1] = angular_velocities[-2] + (angular_velocities[-2] - angular_velocities[-3])
    return velocities, angular_velocities

def compute_velocities_poselib(positions, rotations, time_delta):
    """ Compute velocities from PoseLib"""
    velocities = np.gradient(positions, axis=-3)
    velocities = filters.gaussian_filter1d(velocities, 2, axis=-3, mode="nearest") / time_delta

    angular_velocities = np.zeros_like(positions)
    quat_delta = quat.mul_inv(rotations[ 1:], rotations[ :-1])
    angular_velocities[:-1] = quat.to_scaled_angle_axis(quat_delta)
    angular_velocities = filters.gaussian_filter1d(angular_velocities, 2, axis=-3, mode="nearest") / time_delta
    return velocities, angular_velocities

