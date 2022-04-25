import numpy as np
import cv2
import trimesh

def project(xyz, K, RT):
    xyz = np.dot(xyz, RT[:, :3].T) + RT[:, 3:].T
    xyz = np.dot(xyz, K.T)
    xy = xyz[:, :2] / xyz[:, 2:]
    return xy

def get_bound_corners(bounds):
    min_x, min_y, min_z = bounds[0]
    max_x, max_y, max_z = bounds[1]
    corners_3d = np.array([
        [min_x, min_y, min_z],
        [min_x, min_y, max_z],
        [min_x, max_y, min_z],
        [min_x, max_y, max_z],
        [max_x, min_y, min_z],
        [max_x, min_y, max_z],
        [max_x, max_y, min_z],
        [max_x, max_y, max_z],
    ])
    return corners_3d

def get_bound_2d_mask(bounds, K, pose, H, W):
    corners_3d = get_bound_corners(bounds)
    corners_2d = project(corners_3d, K, pose)
    corners_2d = np.round(corners_2d).astype(int)
    mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(mask, [corners_2d[[0, 1, 3, 2, 0]]], 1)
    cv2.fillPoly(mask, [corners_2d[[4, 5, 7, 6, 5]]], 1)
    cv2.fillPoly(mask, [corners_2d[[0, 1, 5, 4, 0]]], 1)
    cv2.fillPoly(mask, [corners_2d[[2, 3, 7, 6, 2]]], 1)
    cv2.fillPoly(mask, [corners_2d[[0, 2, 6, 4, 0]]], 1)
    cv2.fillPoly(mask, [corners_2d[[1, 3, 7, 5, 1]]], 1)
    return mask

def get_rays(H, W, K, R, T):
    rays_o = -np.dot(R.T, T).ravel()

    i, j = np.meshgrid(np.arange(W, dtype=np.float32),
                       np.arange(H, dtype=np.float32),
                       indexing='xy')
    xy1 = np.stack([i, j, np.ones_like(i)], axis=2)
    pixel_camera = np.dot(xy1, np.linalg.inv(K).T)
    pixel_world = np.dot(pixel_camera - T.ravel(), R)

    rays_d = pixel_world - rays_o[None, None]
    rays_d = rays_d / np.linalg.norm(rays_d, axis=2, keepdims=True)
    rays_o = np.broadcast_to(rays_o, rays_d.shape)
    return rays_o, rays_d

def get_near_far(bounds, ray_o, ray_d):
    bounds = bounds + np.array([-0.01, 0.01])[:, None]
    nominator = bounds[None] - ray_o[:, None]

    d_intersect = (nominator / ray_d[:, None]).reshape(-1, 6)
    p_intersect = d_intersect[..., None] * ray_d[:, None] + ray_o[:, None]

    min_x, min_y, min_z, max_x, max_y, max_z = bounds.ravel()
    eps = 1e-6
    p_mask_at_box = (p_intersect[..., 0] >= (min_x - eps)) * \
                    (p_intersect[..., 0] <= (max_x + eps)) * \
                    (p_intersect[..., 1] >= (min_y - eps)) * \
                    (p_intersect[..., 1] <= (max_y + eps)) * \
                    (p_intersect[..., 2] >= (min_z - eps)) * \
                    (p_intersect[..., 2] <= (max_z + eps))

    mask_at_box = p_mask_at_box.sum(-1) == 2
    p_intervals = p_intersect[mask_at_box][p_mask_at_box[mask_at_box]].reshape(
        -1, 2, 3)

    ray_o = ray_o[mask_at_box]
    ray_d = ray_d[mask_at_box]
    norm_ray = np.linalg.norm(ray_d, axis=1)
    d0 = np.linalg.norm(p_intervals[:, 0] - ray_o, axis=1) / norm_ray
    d1 = np.linalg.norm(p_intervals[:, 1] - ray_o, axis=1) / norm_ray
    near = np.minimum(d0, d1)
    far = np.maximum(d0, d1)

    return near, far, mask_at_box

def sample_ray_h36m(img, msk, K, R, T, bounds, nrays, split):
    H, W = img.shape[:2]
    ray_o, ray_d = get_rays(H, W, K, R, T)

    pose = np.concatenate([R, T], axis=1)
    bound_mask = get_bound_2d_mask(bounds, K, pose, H, W)

    msk = msk * bound_mask
    bound_mask[msk == 100] = 0
    nsampled_rays = 0
    face_sample_ratio = 0
    body_sample_ratio = 0.5
    ray_o_list = []
    ray_d_list = []
    rgb_list = []
    near_list = []
    far_list = []
    coord_list = []
    mask_at_box_list = []
    while nsampled_rays < nrays:
        n_body = int((nrays - nsampled_rays) * body_sample_ratio)
        n_face = int((nrays - nsampled_rays) * face_sample_ratio)
        n_rand = (nrays - nsampled_rays) - n_body - n_face

        coord_body = np.argwhere(msk == 1)
        coord_body = coord_body[np.random.randint(0, len(coord_body),
                                                    n_body)]

        coord_face = np.argwhere(msk == 13)
        if len(coord_face) > 0:
            coord_face = coord_face[np.random.randint(
                0, len(coord_face), n_face)]
                
        coord = np.argwhere(bound_mask == 1)
        coord = coord[np.random.randint(0, len(coord), n_rand)]

        if len(coord_face) > 0:
            coord = np.concatenate([coord_body, coord_face, coord], axis=0)
        else:
            coord = np.concatenate([coord_body, coord], axis=0)

        ray_o_ = ray_o[coord[:, 0], coord[:, 1]]
        ray_d_ = ray_d[coord[:, 0], coord[:, 1]]
        rgb_ = img[coord[:, 0], coord[:, 1]]

        near_, far_, mask_at_box = get_near_far(bounds, ray_o_, ray_d_)

        ray_o_list.append(ray_o_[mask_at_box])
        ray_d_list.append(ray_d_[mask_at_box])
        rgb_list.append(rgb_[mask_at_box])
        near_list.append(near_)
        far_list.append(far_)
        coord_list.append(coord[mask_at_box])
        mask_at_box_list.append(mask_at_box[mask_at_box])
        nsampled_rays += len(near_)

        ray_o = np.concatenate(ray_o_list).astype(np.float32)
        ray_d = np.concatenate(ray_d_list).astype(np.float32)
        rgb = np.concatenate(rgb_list).astype(np.float32)
        near = np.concatenate(near_list).astype(np.float32)
        far = np.concatenate(far_list).astype(np.float32)
        coord = np.concatenate(coord_list)
        mask_at_box = np.concatenate(mask_at_box_list)
    else:
        rgb = img.reshape(-1, 3).astype(np.float32)
        ray_o = ray_o.reshape(-1, 3).astype(np.float32)
        ray_d = ray_d.reshape(-1, 3).astype(np.float32)
        near, far, mask_at_box = get_near_far(bounds, ray_o, ray_d)
        near = near.astype(np.float32)
        far = far.astype(np.float32)
        rgb = rgb[mask_at_box]
        ray_o = ray_o[mask_at_box]
        ray_d = ray_d[mask_at_box]
        coord = np.argwhere(mask_at_box.reshape(H, W) == 1)

    return rgb, ray_o, ray_d, near, far, coord, mask_at_box


