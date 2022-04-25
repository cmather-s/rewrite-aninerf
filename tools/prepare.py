import os
import json
import numpy as np
import cv2
import open3d as o3d
from psbody.mesh import Mesh
import pickle
import trimesh
import tqdm

def batch_rodrigues(poses):
    batch_size = poses.shape[0]
    angle = np.linalg.norm(poses + 1e-8, axis=1, keepdims=True)
    rot_dir = poses / angle

    cos = np.cos(angle)[:, None]
    sin = np.sin(angle)[:, None]

    rx, ry, rz = np.split(rot_dir, 3, axis=1)
    zeros = np.zeros([batch_size, 1])
    K = np.concatenate([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros],axis=1)
    K = K.reshape([batch_size, 3, 3])

    ident = np.eye(3)[None]
    rot_mat = ident + sin * K + (1 - cos) * np.matmul(K, K)

    return rot_mat

def get_rigid_transformation(poses, joints, parents):
    rot_mats = batch_rodrigues(poses)
    rel_joints = joints.copy()
    rel_joints[1:] -= joints[parents[1:]]

    transforms_mat = np.concatenate([rot_mats, rel_joints[..., None]], axis=2)
    padding = np.zeros([24, 1, 4])
    padding[..., 3] = 1
    transforms_mat = np.concatenate([transforms_mat, padding], axis=1)

def get_transform_params(smpl, params):
    v_template = np.array(smpl['v_template'])
    shapedirs = np.array(smpl['shapedirs'])
    betas = params['shapes']
    v_shaped = v_template + np.sum(shapedirs * betas[None], axis=2)

    poses = params['poses'].reshape(-1, 3)
    rot_mats = batch_rodrigues(poses)

    joints = smpl['J_regressor'].dot(v_shaped)
    parents = smpl['kintree_table'][0]
    A = get_rigid_transformation(rot_mats, joints, parents)
    R = cv2.Rodrigues(params['Rh'][0])[0]
    Th = params['Th']
    return A, R, Th, joints

def get_grid_points(xyz):
    min_xyz = np.min(xyz, axis=0)
    max_xyz = np.max(xyz, axis=0)
    min_xyz -= 0.05
    max_xyz += 0.05
    bounds = np.stack([min_xyz, max_xyz], axis=0)
    vsize = 0.025
    voxel_size = [vsize, vsize, vsize]
    x = np.arange(bounds[0, 0], bounds[1, 0] + voxel_size[0], voxel_size[0])
    y = np.arange(bounds[0, 1], bounds[1, 1] + voxel_size[1], voxel_size[1])
    z = np.arange(bounds[0, 2], bounds[1, 2] + voxel_size[2], voxel_size[2])
    pts = np.stack(np.meshgrid(x, y, z, indexing='ij'), axis=-1)
    return pts

def get_tpose_blend_weights(param_path,vertices_path,smpl_path):
    param_path = os.path.join(param_path, '{}.npy'.format(i))
    vertices_path = os.path.join(vertices_path, '{}.npy'.format(i))

    params = np.load(param_path, allow_pickle=True).item()
    vertices = np.load(vertices_path)

    sf = open(smpl_path,'rb')
    su = pickle._Unpickler(sf)
    su.encoding = 'latin1'
    smpl = su.load()
    faces = smpl['f']
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.compute_vertex_normals()
    i = 1
    parent_path = os.path.join(lbs_root, 'parents.npy')
    np.save(parent_path, smpl['kintree_table'][0])
    joint_path = os.path.join(lbs_root, 'joints.npy')
    np.save(joint_path, joints)

    A, R, Th, joints = get_transform_params(smpl, params)
    pxyz = np.dot(vertices - Th, R)
    smpl_mesh = Mesh(pxyz, faces)

    bweights = smpl['weights']
    A = np.dot(bweights, A.reshape(24, -1)).reshape(-1, 4, 4)
    can_pts = pxyz - A[:, :3, 3]
    R_inv = np.linalg.inv(A[:, :3, :3])
    pxyz = np.sum(R_inv * can_pts[:, None], axis=2)

    tvertices_path = os.path.join(lbs_root, 'tvertices.npy')
    np.save(tvertices_path, pxyz)

    smpl_mesh = Mesh(pxyz, faces)

    pts = get_grid_points(pxyz)
    sh = pts.shape
    pts = pts.reshape(-1, 3)

    closest_face, closest_points = smpl_mesh.closest_faces_and_points(pts)
    vert_ids, bary_coords = smpl_mesh.barycentric_coordinates_for_points(
        closest_points, closest_face.astype('int32'))

    t = smpl['weights'][vert_ids] * bary_coords[..., np.newaxis]
    bweights = t.sum(axis=1)

    norm = np.linalg.norm(pts - closest_points, axis=1)

    bweights = np.concatenate((bweights, norm[:, None]), axis=1)
    bweights = bweights.reshape(*sh[:3], 25).astype(np.float32)
    bweight_path = os.path.join(lbs_root, 'tbw.npy')
    np.save(bweight_path, bweights)

    return bweights

def get_bweights(param_path, vertices_path):
    params = np.load(param_path, allow_pickle=True).item()
    vertices = np.load(vertices_path)
    sf = open(smpl_path,'rb')
    su = pickle._Unpickler(sf)
    su.encoding = 'latin1'
    smpl = su.load()
    faces = smpl['f']
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.compute_vertex_normals()
    A, R, Th, joints = get_transform_params(smpl, params)

    pxyz = np.dot(vertices - Th, R)
    smpl_mesh = Mesh(pxyz, faces)

    pts = get_grid_points(pxyz)
    sh = pts.shape
    pts = pts.reshape(-1, 3)

    vert_ids, norm = smpl_mesh.closest_vertices(pts, use_cgal=True)
    bweights = smpl['weights'][vert_ids]

    A = np.dot(bweights, A.reshape(24, -1)).reshape(-1, 4, 4)
    can_pts = pts - A[:, :3, 3]
    R_inv = np.linalg.inv(A[:, :3, :3])
    can_pts = np.sum(R_inv * can_pts[:, None], axis=2)

    bweights = np.concatenate((bweights, norm[:, None]), axis=1)
    bweights = bweights.reshape(*sh[:3], 25).astype(np.float32)
    return bweights

def prepare_blend_weights(param_path,vertices_path,begin_frame, end_frame, frame_interval):
    annot_path = os.path.join(data_root, human, 'annots.npy')
    annot = np.load(annot_path, allow_pickle=True).item()
    bweight_dir = os.path.join(lbs_root, 'bweights')
    os.system('mkdir -p {}'.format(bweight_dir))

    end_frame = len(annot['ims']) if end_frame < 0 else end_frame
    for i in range(begin_frame, end_frame, frame_interval):
        param_save_path = os.path.join(param_path, '{}.npy'.format(i))
        vertices_path = os.path.join(vertices_path, '{}.npy'.format(i))
        bweights = get_bweights(param_save_path, vertices_path)
        bweight_path = os.path.join(bweight_dir, '{}.npy'.format(i))
        np.save(bweight_path, bweights)

if __name__ == "__main__":
    data_root = 'data/zju_mocap'
    human = 'CoreView_313'
    begin_frame = 1
    num_frames = 1400
    frame_interval = 5
    lbs_root = os.path.join(data_root, human, 'lbs')
    os.system('mkdir -p {}'.format(lbs_root))

    param_path = os.path.join(data_root, human, 'new_params')
    vertices_path = os.path.join(data_root, human, 'new_vertices')
    smpl_path = os.path.join(data_root, 'smplx/smpl/SMPL_NEUTRAL.pkl')

    end_frame = begin_frame + num_frames

    get_tpose_blend_weights(param_path,vertices_path,smpl_path)
    prepare_blend_weights(param_path,vertices_path,begin_frame, end_frame, frame_interval)
