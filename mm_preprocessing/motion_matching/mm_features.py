from enum import IntEnum
import numpy as np
from .utils import UHumanBodyBones



class MMFeatureType(IntEnum):
    Position = 0
    Velocity = 1
    TrajectoryPositions = 2
    TrajectoryDirections = 3
    Phase = 4

class MMFeature:
    bone : UHumanBodyBones
    ftype : MMFeatureType
    weight : float
    bone_idx: int
    def __init__(self,bone, ftype, weight) -> None:
        self.bone = bone
        self.ftype = ftype
        self.weight = weight



def get_n_features(feature_descs):
    n_features = 0
    for d in feature_descs:
        if d.ftype == MMFeatureType.Position or d.ftype ==  MMFeatureType.Velocity:
            n_features += 3
        elif d.ftype == MMFeatureType.TrajectoryPositions or d.ftype ==  MMFeatureType.TrajectoryDirections:
            n_features += 6
        elif d.ftype == MMFeatureType.Phase:
            n_features += 1
    return n_features


def calculate_features(motion_db, feature_descs):
    n_features = get_n_features(feature_descs)
    n_frames = len(motion_db.bone_positions)
    features = np.zeros((n_frames, n_features), np.float32)
    offset = 0
    for d in feature_descs:
        if d.ftype == MMFeatureType.Phase:
            features[:, offset] = motion_db.phase_data
            offset+=1
        if d.ftype == MMFeatureType.Position:
            print("pos", d.bone, d.bone_idx)
            features[:, offset: offset+3] = motion_db.get_relative_bone_positions(d.bone_idx)
            offset+=3
        if d.ftype == MMFeatureType.Velocity:
            print("vel", d.bone, d.bone_idx)
            features[:, offset: offset+3] = motion_db.get_relative_bone_velocities(d.bone_idx)
            offset+=3
        if d.ftype == MMFeatureType.TrajectoryPositions:
            print("t pos", d.bone, d.bone_idx)
            features[:, offset: offset+6] = motion_db.get_position_trajectories(d.bone_idx)
            offset+=6
        if d.ftype == MMFeatureType.TrajectoryDirections:
            print("t dir", d.bone, d.bone_idx)
            features[:, offset: offset+6] = motion_db.get_direction_trajectories(d.bone_idx)
            offset+=6
    return features

def calculate_normalized_features(motion_db, feature_descs):
    
    features = calculate_features(motion_db, feature_descs)
    feature_weight_vector = get_feature_weigth_vector(feature_descs)
    features_mean, features_scale = calculate_feature_mean_and_scale(features, feature_weight_vector)
    features = (features-features_mean) / features_scale
    return features, features_mean, features_scale

def get_feature_weigth_vector(feature_descs):
    n_features = get_n_features(feature_descs)
    weights = np.zeros(n_features)
    offset = 0
    for d in feature_descs:
        if d.ftype == MMFeatureType.Phase:
            weights[offset] = d.weight
            offset+=1
        if d.ftype == MMFeatureType.Position:
            weights[offset: offset+3] = d.weight
            offset+=3
        if d.ftype == MMFeatureType.Velocity:
            weights[offset: offset+3] = d.weight
            offset+=3
        if d.ftype == MMFeatureType.TrajectoryPositions:
            weights[offset: offset+6] = d.weight
            offset+=6
        if d.ftype == MMFeatureType.TrajectoryDirections:
            weights[offset: offset+6] = d.weight
            offset+=6
    return weights

def calculate_feature_mean_and_scale(features, feature_weight_vector):
    n_frames = features.shape[0]
    n_features = features.shape[1]
    #features_mean = np.zeros(n_features)
    #for i in range(n_frames):
    #    features_mean += features[i]/n_frames
    #features_vars = np.zeros(n_features)
    #for i in range(n_frames):
    #    temp = features[i] - features_mean
    #    features_vars += (temp*temp)/n_frames
    features_mean = np.mean(features, axis=0)
    feature_deltas = n_features - features_mean
    features_vars = (feature_deltas * feature_deltas)/2
    std = np.sum([np.sqrt(var)/n_features for var in features_vars])
    features_scale = std/feature_weight_vector
    return features_mean, features_scale