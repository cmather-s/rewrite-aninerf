import os
import torch.utils.data as data
import numpy as np
from PIL import Image
import imageio
import cv2
import sample_rays

class Dataset(data.DataSet):
    def __init__(self,data_root):
        self.data_root = data_root
        ann_file  = os.path.join(data_root,'lbs/annots.npy')
        annots = np.load('', allow_pickle=True).item()
        self.cams = annots['cams']
        begin_frame = 1
        frame_interval = 6
        total_frame = 260
        view = [0,1,2]
        self.ims = np.array([
            np.array(ims_data['ims'])[view]
            for ims_data in annots['ims'][begin_frame:begin_frame+frame_interval*total_frame][::frame_interval]
        ]).ravel()
        self.indxes =  np.array([
            np.array(len(ims_data['ims']))[view]
            for ims_data in annots['ims'][begin_frame:begin_frame+frame_interval*total_frame][::frame_interval]
        ]).ravel()
        joints = np.load(os.path.join(self.data_root, 'lbs/joints.npy'))
        self.joints = joints.astype(np.float32)
        self.parents = np.load(os.path.join(self.data_root, 'lbs/parents.npy'))
        self.nrays = 1024

    def get_mask(self,index):
        msk_path = os.path.join(self.data_root, 'mask_cihp',
                                self.ims[index])[:-4] + '.png'
        if not os.path.exists(msk_path):
            msk_path = os.path.join(self.data_root, 'mask',
                                    self.ims[index])[:-4] + '.png'
        if not os.path.exists(msk_path):
            msk_path = os.path.join(self.data_root, self.ims[index].replace(
                'images', 'mask'))[:-4] + '.png'
        msk_cihp = imageio.imread(msk_path)
        if len(msk_cihp.shape) == 3:
            msk_cihp = msk_cihp[..., 0]
        msk_cihp = (msk_cihp != 0).astype(np.uint8)
        msk = msk_cihp
        orig_msk = msk.copy()

        border = 5
        kernel = np.ones((border, border), np.uint8)
        msk_erode = cv2.erode(msk.copy(), kernel)
        msk_dilate = cv2.dilate(msk.copy(), kernel)
        msk[(msk_dilate - msk_erode) == 1] = 100

        return msk, orig_msk

    def get_input(self,index):
        vertices_path = os.path.join(self.data_root, 'vertices'
                                     '{}.npy'.format(index))
        wxyz = np.load(vertices_path).astype(np.float32)

        params_path = os.path.join(self.data_root, 'params',
                                   '{}.npy'.format(i))
        params = np.load(params_path, allow_pickle=True).item()
        Rh = params['Rh'].astype(np.float32)
        Th = params['Th'].astype(np.float32)

        R = cv2.Rodrigues(Rh)[0].astype(np.float32)

        pxyz = np.dot(wxyz - Th, R).astype(np.float32)

        poses = params['poses'].reshape(-1, 3)
        joints = self.joints
        parents = self.parents

        A = sample_rays.get_rigid_transformation(poses, joints, parents)

        pbw = np.load(os.path.join(self.data_root, 'lbs/bweights/{}.npy'.format(index)))
        pbw = pbw.astype(np.float32)

        return wxyz, pxyz, A, pbw, Rh, Th

    def __getitem__(self, index):
        img_path = os.path.join(self.data_root, self.ims[index])
        img = imageio.imread(img_path).astype(np.float32) / 255.
        msk, orig_msk = self.get_mask(index)

        H, W = img.shape[:2]
        msk = cv2.resize(msk, (W, H), interpolation=cv2.INTER_NEAREST)
        orig_msk = cv2.resize(orig_msk, (W, H),
                              interpolation=cv2.INTER_NEAREST)

        cam_ind = self.cam_inds[index]
        K = np.array(self.cams['K'][cam_ind])
        D = np.array(self.cams['D'][cam_ind])
        img = cv2.undistort(img, K, D)
        msk = cv2.undistort(msk, K, D)
        orig_msk = cv2.undistort(orig_msk, K, D)

        R = np.array(self.cams['R'][cam_ind])
        T = np.array(self.cams['T'][cam_ind]) / 1000.

        H, W = int(img.shape[0]), int(img.shape[1])
        img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        msk = cv2.resize(msk, (W, H), interpolation=cv2.INTER_NEAREST)
        orig_msk = cv2.resize(orig_msk, (W, H),
                              interpolation=cv2.INTER_NEAREST)

        i = int(os.path.basename(img_path).split('_')[4])
        frame_index = i - 1

        vertices_path = os.path.join(self.lbs_root, 'tvertices.npy')
        tpose = np.load(vertices_path).astype(np.float32)
        tbounds = get_bounds(tpose)
        tbw = np.load(os.path.join(self.lbs_root, 'tbw.npy'))
        tbw = tbw.astype(np.float32)

        wpts, ppts, A, pbw, Rh, Th = self.prepare_input(i)

        pbounds = get_bounds(ppts)
        wbounds = get_bounds(wpts)

        rgb, ray_o, ray_d, near, far, coord, mask_at_box = if_nerf_dutils.sample_ray_h36m(
            img, msk, K, R, T, wbounds, self.nrays, self.split)

        occupancy = orig_msk[coord[:, 0], coord[:, 1]]

        # nerf
        ret = {
            'rgb': rgb,
            'occupancy': occupancy,
            'ray_o': ray_o,
            'ray_d': ray_d,
            'near': near,
            'far': far,
            'mask_at_box': mask_at_box
        }

        # blend weight
        meta = {
            'A': A,
            'pbw': pbw,
            'tbw': tbw,
            'pbounds': pbounds,
            'wbounds': wbounds,
            'tbounds': tbounds
        }
        ret.update(meta)

        # transformation
        R = cv2.Rodrigues(Rh)[0].astype(np.float32)
        meta = {'R': R, 'Th': Th, 'H': H, 'W': W}
        ret.update(meta)

        latent_index = index // self.num_cams
        bw_latent_index = index // self.num_cams
        if cfg.test_novel_pose:
            latent_index = cfg.num_train_frame - 1
        meta = {
            'latent_index': latent_index,
            'bw_latent_index': bw_latent_index,
            'frame_index': frame_index,
            'cam_ind': cam_ind
        }
        ret.update(meta)

        return ret

    def __len__(self):
        return len(self.ims)