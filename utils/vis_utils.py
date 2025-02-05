import pickle
import numpy as np
import torch
import math
import sys
import os
import cv2 as cv
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.manolayer import ManoLayer
from utils.config import get_cfg_defaults
from utils.utils import projection_batch, get_mano_path, get_dense_color_path


# Data structures and functions for rendering
from pytorch3d.structures import Meshes
from pytorch3d.vis.plotly_vis import AxisArgs, plot_batch_individually, plot_scene
from pytorch3d.vis.texture_vis import texturesuv_image_matplotlib
from pytorch3d.renderer import (
    look_at_view_transform,
    PerspectiveCameras,
    OrthographicCameras,
    PointLights,
    DirectionalLights,
    Materials,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    HardPhongShader,
    TexturesUV,
    TexturesVertex,
    HardFlatShader,
    HardGouraudShader,
    AmbientLights,
    SoftSilhouetteShader
)
import open3d as o3d
import pdb


class Renderer():
    def __init__(self, img_size, device='cpu'):
        self.img_size = img_size
        self.raster_settings = RasterizationSettings(
            image_size=img_size,
            blur_radius=0.0,
            faces_per_pixel=1
        )

        self.amblights = AmbientLights(device=device)
        self.point_lights = PointLights(location=[[0, 0, -1.0]], device=device)

        self.renderer_rgb = MeshRenderer(
            rasterizer=MeshRasterizer(raster_settings=self.raster_settings),
            shader=HardFlatShader(device=device)
        )
        self.device = device

    def build_camera(self, cameras=None,
                     scale=None, trans2d=None):
        if scale is not None and trans2d is not None:
            bs = scale.shape[0]
            R = torch.tensor([[-1, 0, 0], [0, -1, 0], [0, 0, 1]]).repeat(bs, 1, 1).to(scale.dtype)
            T = torch.tensor([0, 0, 10]).repeat(bs, 1).to(scale.dtype)
            return OrthographicCameras(focal_length=2 * scale.to(self.device),
                                       principal_point=-trans2d.to(self.device),
                                       R=R.to(self.device),
                                       T=T.to(self.device),
                                       in_ndc=True,
                                       device=self.device)
        if cameras is not None:
            # cameras: bs x 3 x 3
            fs = -torch.stack((cameras[:, 0, 0], cameras[:, 1, 1]), dim=-1) * 2 / self.img_size
            pps = -cameras[:, :2, -1] * 2 / self.img_size + 1
            return PerspectiveCameras(focal_length=fs.to(self.device),
                                      principal_point=pps.to(self.device),
                                      in_ndc=True,
                                      device=self.device
                                      )

    def build_texture(self, uv_verts=None, uv_faces=None, texture=None,
                      v_color=None):
        if uv_verts is not None and uv_faces is not None and texture is not None:
            return TexturesUV(texture.to(self.device), uv_faces.to(self.device), uv_verts.to(self.device))
        if v_color is not None:
            return TexturesVertex(verts_features=v_color.to(self.device))

    def render(self, verts, faces, cameras, textures, amblights=False,
               lights=None):
        if lights is None:
            if amblights:
                lights = self.amblights
            else:
                lights = self.point_lights
        mesh = Meshes(verts=verts.to(self.device), faces=faces.to(self.device), textures=textures)
        output = self.renderer_rgb(mesh, cameras=cameras, lights=lights)
        alpha = output[..., 3]
        img = output[..., :3] / 255
        return img, alpha


class mano_renderer(Renderer):
    def __init__(self, mano_path=None, dense_path=None, img_size=224, device='cpu'):
        super(mano_renderer, self).__init__(img_size, device)
        if mano_path is None:
            mano_path = get_mano_path()
        if dense_path is None:
            dense_path = get_dense_color_path()

        self.mano = ManoLayer(mano_path, center_idx=9, use_pca=True)
        self.mano.to(self.device)
        self.faces_np = self.mano.get_faces().astype(np.int64)
        self.faces = torch.from_numpy(self.faces_np).to(self.device).unsqueeze(0)

        with open(dense_path, 'rb') as file:
            dense_coor = pickle.load(file)
        self.dense_coor = torch.from_numpy(dense_coor) * 255

    def render_rgb(self, cameras=None, scale=None, trans2d=None,
                   R=None, pose=None, shape=None, trans=None,
                   v3d=None,
                   uv_verts=None, uv_faces=None, texture=None, v_color=(255, 255, 255),
                   amblights=False):
        if v3d is None:
            v3d, _ = self.mano(R, pose, shape, trans=trans)
        bs = v3d.shape[0]
        vNum = v3d.shape[1]

        if not isinstance(v_color, torch.Tensor):
            v_color = torch.tensor(v_color)
        v_color = v_color.expand(bs, vNum, 3).to(v3d)

        return self.render(v3d, self.faces.repeat(bs, 1, 1),
                           self.build_camera(cameras, scale, trans2d),
                           self.build_texture(uv_verts, uv_faces, texture, v_color),
                           amblights)

    def render_densepose(self, cameras=None, scale=None, trans2d=None,
                         R=None, pose=None, shape=None, trans=None,
                         v3d=None):
        if v3d is None:
            v3d, _ = self.mano(R, pose, shape, trans=trans)
        bs = v3d.shape[0]
        vNum = v3d.shape[1]

        return self.render(v3d, self.faces.repeat(bs, 1, 1),
                           self.build_camera(cameras, scale, trans2d),
                           self.build_texture(v_color=self.dense_coor.expand(bs, vNum, 3).to(v3d)),
                           True)

class ours_two_hand_renderer(Renderer):
    def __init__(self, mano_path=None, dense_path=None, img_size=224, device='cpu'):
        super(ours_two_hand_renderer, self).__init__(img_size, device)
        if mano_path is None:
            mano_path = get_mano_path()
        if dense_path is None:
            dense_path = get_dense_color_path()

        self.mano = {'right': ManoLayer(mano_path['right'], center_idx=None),
                     'left': ManoLayer(mano_path['left'], center_idx=None)}
        self.mano['left'].to(self.device)
        self.mano['right'].to(self.device)

        left_faces = torch.from_numpy(self.mano['left'].get_faces().astype(np.int64)).to(self.device).unsqueeze(0)
        right_faces = torch.from_numpy(self.mano['right'].get_faces().astype(np.int64)).to(self.device).unsqueeze(0)
        left_faces = right_faces[..., [1, 0, 2]]

        with open(dense_path, 'rb') as file:
            dense_coor = pickle.load(file)
        self.dense_coor = torch.from_numpy(dense_coor) * 255
    
    def render_other_view_ours(self, img, obj_paths, cam_param, theta=60, v_color=None):
        left_mesh = o3d.io.read_triangle_mesh(obj_paths['left'])
        left_verts = torch.Tensor(np.asarray(left_mesh.vertices)).cuda().unsqueeze(0)
        left_faces = torch.Tensor(np.asarray(left_mesh.triangles)).cuda().unsqueeze(0)

        right_mesh = o3d.io.read_triangle_mesh(obj_paths['right'])
        right_verts = torch.Tensor(np.asarray(right_mesh.vertices)).cuda().unsqueeze(0)
        right_faces = torch.Tensor(np.asarray(right_mesh.triangles)).cuda().unsqueeze(0)

        c = (torch.mean(left_verts, axis=1) + torch.mean(right_verts, axis=1)).unsqueeze(1) / 2
        left_verts = left_verts - c
        right_verts = right_verts - c

        theta = 3.14159 / 180 * theta

        R = [[math.cos(theta), 0, math.sin(theta)],
                [0, 1, 0],
                [-math.sin(theta), 0, math.cos(theta)]]
        R = torch.tensor(R).float().cuda()

        left_verts = torch.matmul(left_verts, R) + c
        right_verts = torch.matmul(right_verts, R) + c


        bs = left_verts.shape[0]
        left_vNum = left_verts.shape[1]
        right_vNum = right_verts.shape[1]
        if v_color is None:
            v_color = torch.zeros((left_vNum+right_vNum, 3))
            v_color[:left_vNum, 0] = 204
            v_color[:left_vNum, 1] = 153
            v_color[:left_vNum, 2] = 0
            v_color[left_vNum:, 0] = 102
            v_color[left_vNum:, 1] = 102
            v_color[left_vNum:, 2] = 255

        if not isinstance(v_color, torch.Tensor):
            v_color = torch.tensor(v_color)
        v_color = v_color.expand(bs, left_vNum+right_vNum, 3).float().to(self.device)

        v3d = torch.cat((left_verts, right_verts), dim=1)

        faces = torch.cat((left_faces, right_faces + left_vNum), dim=1)

        R = cam_param['R']
        T = cam_param['t']
        cameras = torch.Tensor(cam_param['camera']).unsqueeze(0)

        # fs = -torch.tensor(cam_param['focal']).unsqueeze(0) * 2 / self.img_size
        # pps = -torch.tensor(cam_param['princpt']).unsqueeze(0) * 2 / self.img_size + 1
        
        
        # scale=torch.ones((1,)).float().cuda() * 3
        # trans2d=torch.zeros((1, 2)).float().cuda()
        
        # bs = scale.shape[0]
        # R = torch.tensor([[-1, 0, 0], [0, -1, 0], [0, 0, 1]]).repeat(bs, 1, 1).to(scale.dtype)
        # T = torch.tensor([0, 0, 10]).repeat(bs, 1).to(scale.dtype)
        # cameras = OrthographicCameras(focal_length=1.9 * scale.to(self.device),
        #                             principal_point=-trans2d.to(self.device),
        #                             R=R.to(self.device),
        #                             T=T.to(self.device),
        #                             in_ndc=True,
        #                             device=self.device)
        R = cam_param['R']
        T = cam_param['t']
        cameras = torch.Tensor(cam_param['camera']).unsqueeze(0)

        # fs = -torch.tensor(cam_param['focal']).unsqueeze(0) * 2 / self.img_size
        # pps = -torch.tensor(cam_param['princpt']).unsqueeze(0) * 2 / self.img_size + 1


        fs = -torch.stack((cameras[:, 0, 0], cameras[:, 1, 1]), dim=-1) * 2 / 256
        pps = -cameras[:, :2, -1] * 2 / 256 + 1
        cameras = PerspectiveCameras(focal_length=fs.to(self.device),
                                    principal_point=pps.to(self.device),
                                    in_ndc=True,
                                    device=self.device
                                    )

        # fs = -torch.stack((cameras[:, 0, 0], cameras[:, 1, 1]), dim=-1) * 2 / self.img_size
        # pps = -cameras[:, :2, -1] * 2 / self.img_size + 1
        # cameras = PerspectiveCameras(focal_length=fs.to(self.device),
        #                             principal_point=pps.to(self.device),
        #                             in_ndc=True,
        #                             device=self.device
        #                             )

        lights = self.point_lights
        textures= self.build_texture(None, None, None, v_color)
        
        mesh = Meshes(verts=v3d.to(self.device), faces=faces.to(self.device), textures=textures)
        output = self.renderer_rgb(mesh, cameras=cameras, lights=lights)
        alpha = output[..., 3]
        img_out = output[..., :3] / 255

        img_out = img_out[0].detach().cpu().numpy() * 255
        white_bg_img = np.ones_like(img_out) * 255
        alpha = alpha[0].detach().cpu().numpy()[..., np.newaxis]
        img_out = img_out * alpha + white_bg_img * (1 - alpha)
        img_out = img_out.astype(np.uint8)

        return img_out

    def render(self, img, obj_paths, cam_param, v_color=None):
        left_mesh = o3d.io.read_triangle_mesh(obj_paths['left'])
        left_verts = torch.Tensor(np.asarray(left_mesh.vertices)).cuda().unsqueeze(0)
        left_faces = torch.Tensor(np.asarray(left_mesh.triangles)).cuda().unsqueeze(0)

        right_mesh = o3d.io.read_triangle_mesh(obj_paths['right'])
        right_verts = torch.Tensor(np.asarray(right_mesh.vertices)).cuda().unsqueeze(0)
        right_faces = torch.Tensor(np.asarray(right_mesh.triangles)).cuda().unsqueeze(0)
        
        bs = left_verts.shape[0]
        left_vNum = left_verts.shape[1]
        right_vNum = right_verts.shape[1]
        if v_color is None:
            v_color = torch.zeros((left_vNum+right_vNum, 3))
            v_color[:left_vNum, 0] = 204
            v_color[:left_vNum, 1] = 153
            v_color[:left_vNum, 2] = 0
            v_color[left_vNum:, 0] = 102
            v_color[left_vNum:, 1] = 102
            v_color[left_vNum:, 2] = 255

        if not isinstance(v_color, torch.Tensor):
            v_color = torch.tensor(v_color)
        v_color = v_color.expand(bs, left_vNum+right_vNum, 3).float().to(self.device)

        v3d = torch.cat((left_verts, right_verts), dim=1)

        faces = torch.cat((left_faces, right_faces + left_vNum), dim=1)

        R = cam_param['R']
        T = cam_param['t']
        cameras = torch.Tensor(cam_param['camera']).unsqueeze(0)

        # fs = -torch.tensor(cam_param['focal']).unsqueeze(0) * 2 / self.img_size
        # pps = -torch.tensor(cam_param['princpt']).unsqueeze(0) * 2 / self.img_size + 1


        fs = -torch.stack((cameras[:, 0, 0], cameras[:, 1, 1]), dim=-1) * 2 / 256
        pps = -cameras[:, :2, -1] * 2 / 256 + 1
        cameras = PerspectiveCameras(focal_length=fs.to(self.device),
                                    principal_point=pps.to(self.device),
                                    in_ndc=True,
                                    device=self.device
                                    )


        # cameras = PerspectiveCameras(focal_length=fs.to(self.device),
        #                               principal_point=pps.to(self.device),
        #                               in_ndc=True,
        #                               device=self.device
        #                               )
        #import pdb; pdb.set_trace()
    

        # cameras = PerspectiveCameras(focal_length=torch.tensor(cam_param['focal']).unsqueeze(0).to(self.device),
        #                              principal_point=torch.tensor(cam_param['princpt']).unsqueeze(0).to(self.device),
        #                              in_ndc=True,
        #                              device=self.device
        #                             )

        # R = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]]).repeat(bs, 1, 1)
        # T = torch.tensor([0, 0, 10]).repeat(bs, 1)

        # cameras = OrthographicCameras(focal_length=fs.to(self.device),
        #                                principal_point=pps.to(self.device),
        #                                R=R.to(self.device),
        #                                T=T.to(self.device),
        #                                in_ndc=True,
        #                                device=self.device)
       
        lights = self.point_lights
        textures= self.build_texture(None, None, None, v_color)
        
        mesh = Meshes(verts=v3d.to(self.device), faces=faces.to(self.device), textures=textures)
        output = self.renderer_rgb(mesh, cameras=cameras, lights=lights)
        alpha = output[..., 3]
        img_out = output[..., :3] / 255

        img_out = img_out[0].detach().cpu().numpy() * 255
        alpha = alpha[0].detach().cpu().numpy()[..., np.newaxis]
        bg_img = cv.resize(img, (self.img_size, self.img_size))
        img_out = img_out * alpha + bg_img * (1 - alpha)
        img_out = img_out.astype(np.uint8)

        return img_out

        # return self.render(v3d,
        #                    self.faces.repeat(bs, 1, 1),
        #                    self.build_camera(cameras, scale, trans2d),
        #                    self.build_texture(uv_verts, uv_faces, texture, v_color),
        #                    amblights,
        #                    lights)
    def render_rgb2hands(self, img, obj_paths, cam_param, v_color=None):
        left_mesh = o3d.io.read_triangle_mesh(obj_paths['left'])
        left_verts = torch.Tensor(np.asarray(left_mesh.vertices)).cuda().unsqueeze(0)
        left_faces = torch.Tensor(np.asarray(left_mesh.triangles)).cuda().unsqueeze(0)

        right_mesh = o3d.io.read_triangle_mesh(obj_paths['right'])
        right_verts = torch.Tensor(np.asarray(right_mesh.vertices)).cuda().unsqueeze(0)
        right_faces = torch.Tensor(np.asarray(right_mesh.triangles)).cuda().unsqueeze(0)
        
        bs = left_verts.shape[0]
        left_vNum = left_verts.shape[1]
        right_vNum = right_verts.shape[1]
        if v_color is None:
            v_color = torch.zeros((left_vNum+right_vNum, 3))
            v_color[:left_vNum, 0] = 204
            v_color[:left_vNum, 1] = 153
            v_color[:left_vNum, 2] = 0
            v_color[left_vNum:, 0] = 102
            v_color[left_vNum:, 1] = 102
            v_color[left_vNum:, 2] = 255

        if not isinstance(v_color, torch.Tensor):
            v_color = torch.tensor(v_color)
        v_color = v_color.expand(bs, left_vNum+right_vNum, 3).float().to(self.device)

        v3d = torch.cat((left_verts, right_verts), dim=1)

        faces = torch.cat((left_faces, right_faces + left_vNum), dim=1)

        # fx = focal length axis-x
        # fy = focal length axis-y
        # dx = principal point axis-x
        # dy = principal point axis-y
        # fs = -cam_param[:2] * 2 / 256
        # pps = -cam_param[2:] * 2 / 256 + 1
        # cameras = PerspectiveCameras(focal_length=fs.unsqueeze(0).to(self.device),
        #                             principal_point=pps.unsqueeze(0).to(self.device),
        #                             in_ndc=True,
        #                             device=self.device
        #                             )

        

        scale=torch.ones((1,)).float().cuda() * 4
        trans2d=torch.zeros((1, 2)).float().cuda()
        trans2d[0,0] = -1.0
        trans2d[0,1] = -1.0

        bs = scale.shape[0]
        R = torch.tensor([[-1, 0, 0], [0, -1, 0], [0, 0, 1]]).repeat(bs, 1, 1).to(scale.dtype)
        T = torch.tensor([0, 0, 10]).repeat(bs, 1).to(scale.dtype)
        cameras = OrthographicCameras(focal_length=2 * scale.to(self.device),
                                    principal_point=-trans2d.to(self.device),
                                    R=R.to(self.device),
                                    T=T.to(self.device),
                                    in_ndc=True,
                                    device=self.device)

       
        lights = self.point_lights
        textures= self.build_texture(None, None, None, v_color)
        
        mesh = Meshes(verts=v3d.to(self.device), faces=faces.to(self.device), textures=textures)
        output = self.renderer_rgb(mesh, cameras=cameras, lights=lights)
        alpha = output[..., 3]
        img_out = output[..., :3] / 255

        img_out = img_out[0].detach().cpu().numpy() * 255
        alpha = alpha[0].detach().cpu().numpy()[..., np.newaxis]
        bg_img = cv.resize(img, (self.img_size, self.img_size))
        img_out = img_out * alpha + bg_img * (1 - alpha)
        img_out = img_out.astype(np.uint8)

        return img_out

        
class mano_two_hands_renderer(Renderer):
    def __init__(self, mano_path=None, dense_path=None, img_size=224, device='cpu'):
        super(mano_two_hands_renderer, self).__init__(img_size, device)
        if mano_path is None:
            mano_path = get_mano_path()
        if dense_path is None:
            dense_path = get_dense_color_path()

        self.mano = {'right': ManoLayer(mano_path['right'], center_idx=None),
                     'left': ManoLayer(mano_path['left'], center_idx=None)}
        self.mano['left'].to(self.device)
        self.mano['right'].to(self.device)

        left_faces = torch.from_numpy(self.mano['left'].get_faces().astype(np.int64)).to(self.device).unsqueeze(0)
        right_faces = torch.from_numpy(self.mano['right'].get_faces().astype(np.int64)).to(self.device).unsqueeze(0)
        left_faces = right_faces[..., [1, 0, 2]]

        self.faces = torch.cat((left_faces, right_faces + 778), dim=1)

        with open(dense_path, 'rb') as file:
            dense_coor = pickle.load(file)
        self.dense_coor = torch.from_numpy(dense_coor) * 255

    def render_rgb(self, cameras=None, scale=None, trans2d=None,
                   v3d_left=None, v3d_right=None,
                   uv_verts=None, uv_faces=None, texture=None, v_color=None,
                   amblights=False,
                   lights=None):
        bs = v3d_left.shape[0]
        vNum = v3d_left.shape[1]

        if v_color is None:
            v_color = torch.zeros((778 * 2, 3))
            v_color[:778, 0] = 204
            v_color[:778, 1] = 153
            v_color[:778, 2] = 0
            v_color[778:, 0] = 102
            v_color[778:, 1] = 102
            v_color[778:, 2] = 255

        if not isinstance(v_color, torch.Tensor):
            v_color = torch.tensor(v_color)
        v_color = v_color.expand(bs, 2 * vNum, 3).float().to(self.device)

        v3d = torch.cat((v3d_left, v3d_right), dim=1)

        return self.render(v3d,
                           self.faces.repeat(bs, 1, 1),
                           self.build_camera(cameras, scale, trans2d),
                           self.build_texture(uv_verts, uv_faces, texture, v_color),
                           amblights,
                           lights)

    def render_rgb_orth(self, scale_left=None, trans2d_left=None,
                        scale_right=None, trans2d_right=None,
                        v3d_left=None, v3d_right=None,
                        uv_verts=None, uv_faces=None, texture=None, v_color=None,
                        amblights=False):
        scale = scale_left
        trans2d = trans2d_left

        s = scale_right / scale_left
        d = -(trans2d_left - trans2d_right) / 2 / scale_left.unsqueeze(-1)

        s = s.unsqueeze(-1).unsqueeze(-1)
        d = d.unsqueeze(1)
        v3d_right = s * v3d_right
        v3d_right[..., :2] = v3d_right[..., :2] + d

        # scale = (scale_left + scale_right) / 2
        # trans2d = (trans2d_left + trans2d_right) / 2

        return self.render_rgb(self, scale=scale, trans2d=trans2d,
                               v3d_left=v3d_left, v3d_right=v3d_right,
                               uv_verts=uv_verts, uv_faces=uv_faces, texture=texture, v_color=v_color,
                               amblights=amblights)

    def render_mask(self, cameras=None, scale=None, trans2d=None,
                    v3d_left=None, v3d_right=None):
        v_color = torch.zeros((778 * 2, 3))
        v_color[:778, 2] = 255
        v_color[778:, 1] = 255
        rgb, mask = self.render_rgb(cameras, scale, trans2d,
                                    v3d_left, v3d_right,
                                    v_color=v_color,
                                    amblights=True)
        return rgb

    def render_densepose(self, cameras=None, scale=None, trans2d=None,
                         v3d_left=None, v3d_right=None,):
        bs = v3d_left.shape[0]
        vNum = v3d_left.shape[1]

        v3d = torch.cat((v3d_left, v3d_right), dim=1)

        v_color = torch.cat((self.dense_coor, self.dense_coor), dim=0)

        return self.render(v3d,
                           self.faces.repeat(bs, 1, 1),
                           self.build_camera(cameras, scale, trans2d),
                           self.build_texture(v_color=v_color.expand(bs, 2 * vNum, 3).to(v3d_left)),
                           True)
