import numpy as np
import torch
from pytorch3d.renderer.utils import (TensorProperties,
                                      convert_to_tensors_and_broadcast)


class SphericalHarmonicsLights(TensorProperties):
    def __init__(
        self,
        sh_params=None,
        device="cpu",
    ):

        super().__init__(
            device=device,
            ambient_color=((0.0, 0.0, 0.0),),
            sh_params=sh_params,
        )

        if self.sh_params.shape[-2:] != (9, 3):
            msg = "Expected sh_params to have shape (N, 9, 3); got %r"
            raise ValueError(msg % repr(self.sh_params.shape))

        pi = np.pi
        sqrt = np.sqrt
        att = pi * np.array([1., 2./3., 1./4.])
        sh_coeff = torch.FloatTensor([
            att[0] * (1/2) * (1/sqrt(pi)),          # 1
            att[1] * (sqrt(3)/2) * (1/sqrt(pi)),    # Y
            att[1] * (sqrt(3)/2) * (1/sqrt(pi)),    # Z
            att[1] * (sqrt(3)/2) * (1/sqrt(pi)),    # X
            att[2] * (sqrt(15)/2) * (1/sqrt(pi)),   # YX
            att[2] * (sqrt(15)/2) * (1/sqrt(pi)),   # YZ
            att[2] * (sqrt(5)/4) * (1/sqrt(pi)),    # 3Z^2 - 1
            att[2] * (sqrt(15)/2) * (1/sqrt(pi)),   # XZ
            att[2] * (sqrt(15)/4) * (1/sqrt(pi)),   # X^2 - Y^2
        ])
        self.register_buffer("sh_coeff", sh_coeff[None, None, :])

    def clone(self):
        other = self.__class__(device=self.device)
        return super().clone(other)

    def diffuse(self, normals, points) -> torch.Tensor:
        # normals: (B, ..., 3)
        input_shape = normals.shape
        B = input_shape[0]
        normals = normals.view(B, -1, 3)
        # normals: (B, K, 3)

        sh = torch.stack([
            torch.ones_like(normals[..., 0]),               # 1
            normals[..., 1],                                # Y
            normals[..., 2],                                # Z
            normals[..., 0],                                # X
            normals[..., 1] * normals[..., 0],              # YX
            normals[..., 1] * normals[..., 2],              # YZ
            3 * (normals[..., 2] ** 2) - 1,                 # 3Z^2 - 1
            normals[..., 0] * normals[..., 2],              # XZ
            normals[..., 0] ** 2 - normals[..., 1] ** 2,    # X^2 - Y^2
        ], dim=-1)
        # sh: (B, K, 9)

        sh, sh_coeff, sh_params = convert_to_tensors_and_broadcast(sh, self.sh_coeff, self.sh_params, device=normals.device)

        sh = sh * sh_coeff
        # sh_params: (B, 9, 3)
        color = torch.einsum("ijk,ikl->ijl", sh, sh_params)
        color = color.view(B, *input_shape[1:-1], 3)
        return color

    def specular(self, normals, points, camera_position, shininess) -> torch.Tensor:
        return torch.zeros_like(points)
