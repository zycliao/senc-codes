import torch
import numpy as np
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    PointLights,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    HardPhongShader,
    TexturesVertex,
    BlendParams
)


def color_mapping(verts, color):
    assert color in ['skin', 'cloth']
    color_dict = {
        'skin': [250, 237, 205],
        'cloth': [42, 157, 143]
    }
    color = np.array(color_dict[color]) / 255.
    color = np.tile(np.expand_dims(color, 0), [verts.shape[0], 1])
    return color


def merge_mesh(vs, fs, vcs):
    v_num = 0
    new_fs = [fs[0]]
    new_vcs = []
    for i in range(len(vs)):
        if i >= 1:
            v_num += vs[i-1].shape[0]
            new_fs.append(fs[i]+v_num)
        if vcs is not None:
            if isinstance(vcs[i], str):
                new_vcs.append(color_mapping(vs[i], vcs[i]))
            else:
                if vcs[i].ndim == 1:
                    new_vcs.append(np.tile(np.expand_dims(vcs[i], 0), [vs[i].shape[0], 1]))
                else:
                    new_vcs.append(vcs[i])
    vs = np.concatenate(vs, 0)
    new_fs = np.concatenate(new_fs, 0)
    if vcs is not None:
        vcs = np.concatenate(new_vcs, 0)

    return vs, new_fs, vcs


class Renderer(object):
    def __init__(self, img_size, device=torch.device("cuda:0"), max_size=1.5, bg_color=(0.2, 0.2, 0.2), img_num=1):
        # max_size: when the object has such a size, it can fill the whole image
        self.img_size = img_size
        self.device = device
        self.max_size = max_size
        self.cameras = None
        self.renderer = None
        self.bg_color = bg_color
        self.phi_ = - np.pi
        self.theta_ = np.pi / 2
        self.r_ = 10
        self.up = np.array([0, 1, 0])
        self.rotate_speed = 0.1
        self.img_num = img_num

        self.raster_settings = RasterizationSettings(
            image_size=img_size,
            blur_radius=0.0,
            faces_per_pixel=1,
        )
        self.verts, self.faces = None, None
        self.verts_rgb = None
        self.center = None

    @property
    def fov(self):
        return 2 * np.arctan(self.max_size / 2 / self.r) * 180 / np.pi

    @property
    def _phi(self):
        # range (0, 2 * pi)
        self.phi_ = self.phi_ % (2 * np.pi)
        return self.phi_

    @_phi.setter
    def _phi(self, value):
        self.phi_ = value

    @property
    def _theta(self):
        # range (0, pi)
        self.theta_ = self.theta_ % (2 * np.pi)
        if self.theta_ > np.pi:
            self.theta_ = 2 * np.pi - self.theta_
            self._phi += np.pi
        return self.theta_

    @_theta.setter
    def _theta(self, value):
        self.theta_ = np.clip(value, 1e-8, np.pi - 1e-8)

    @property
    def r(self):
        # range (0.1, inf)
        self.r_ = max(0.1, self.r_)
        return self.r_

    @r.setter
    def r(self, value):
        self.r_ = value

    def reset_cam(self):
        self._phi = - np.pi
        self._theta = np.pi / 2
        self.r = 10

    @property
    def cam_location(self):
        # phi is the angle between the y-axis and the projection of the vector on the x-z plane
        # theta is the angle between the vector and the y-axis
        return np.array([self.r * np.sin(self._theta) * np.sin(self._phi),
                            self.r * np.cos(self._theta),
                            self.r * np.sin(self._theta) * np.cos(self._phi)])

    def rotate_horizon(self, left=True):
        if left:
            self._phi -= self.rotate_speed
        else:
            self._phi += self.rotate_speed

    def rotate_vertical(self, up=True):
        if up:
            self._theta -= self.rotate_speed
        else:
            self._theta += self.rotate_speed

    def keyboard_control(self, k):
        if k == 81:
            self.rotate_horizon(True)
        elif k == 83:
            self.rotate_horizon(False)
        elif k == 82:
            self.rotate_vertical(True)
        elif k == 84:
            self.rotate_vertical(False)
        elif k == 43 or k == 61:
            self.max_size -= 0.1
            self.max_size = np.maximum(self.max_size, 0.1)
        elif k == 45 or k == 95:
            self.max_size += 0.1


    def set_mesh(self, verts, faces, verts_rgb=None, center=True, set_center=False):
        if isinstance(verts, list):
            assert isinstance(faces, list)
            assert isinstance(verts[0], np.ndarray)
            verts, faces, verts_rgb = merge_mesh(verts, faces, verts_rgb)
        else:
            if isinstance(verts_rgb, str):
                verts_rgb = color_mapping(verts, verts_rgb)
        if not torch.is_tensor(verts):
            verts = torch.from_numpy(verts).float().to(self.device)
        if not torch.is_tensor(faces):
            faces = torch.from_numpy(faces).to(self.device)
        if verts_rgb is not None and not torch.is_tensor(verts_rgb):
            self.verts_rgb = torch.from_numpy(verts_rgb).float().to(self.device)[None]

        if set_center:
            self.center = (verts.max(0)[0] + verts.min(0)[0]) / 2
        if center:
            if self.center is None:
                verts = verts - (verts.max(0)[0] + verts.min(0)[0]) / 2
            else:
                verts = verts - self.center

        self.verts = verts
        self.faces = faces
        if verts_rgb is None:
            self.verts_rgb = torch.ones_like(verts)[None]

    def render(self):
        assert self.verts is not None, 'Please set the mesh first!'
        textures = TexturesVertex(verts_features=self.verts_rgb)
        mesh = Meshes(verts=[self.verts], faces=[self.faces], textures=textures)
        R, T = look_at_view_transform(eye=[self.cam_location], at=[[0.0, 0.0, 0.0]], up=[self.up])
        cameras = FoVPerspectiveCameras(device=self.device, R=R, T=T, fov=self.fov)

        lights = PointLights(device=self.device, location=[self.cam_location])
        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=cameras,
                raster_settings=self.raster_settings),
            shader=HardPhongShader(device=self.device, cameras=cameras, lights=lights,
                                   blend_params=BlendParams(background_color=self.bg_color))
        )

        # render the mesh
        images = renderer(mesh)
        return images[0, ..., :3].cpu().numpy()
