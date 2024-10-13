import torch
import numpy as np
import open3d as o3d


# def get_nearest_face(v0, v, f):
#     tri_verts = v[f.reshape(-1)].reshape([-1, 3, 3])
#     tri_centroid = np.mean(tri_verts, 1)
#     dist = np.sum(np.square(v0[:, None] - tri_centroid[None]), 2)
#     return np.argmin(dist, 1)


# reimplement get_nearest_face() using KDTree in scipy
from scipy.spatial import KDTree
def get_nearest_face(v0, v, f):
    tri_verts = v[f.reshape(-1)].reshape([-1, 3, 3])
    tri_centroid = np.mean(tri_verts, 1)
    tree = KDTree(tri_centroid)
    dist, idx = tree.query(v0)
    return dist, idx


def barycentric(p, a, b, c):
    # p (N, 3)
    # a, b, c (N, 3), three vertices of the triangle
    # return (N, 3), the barycentric coordinates
    v0, v1, v2 = b-a, c-a, p-a
    d00 = np.sum(v0*v0, 1)
    d01 = np.sum(v0*v1, 1)
    d11 = np.sum(v1*v1, 1)
    d20 = np.sum(v2*v0, 1)
    d21 = np.sum(v2*v1, 1)
    denom = d00 * d11 - d01 * d01 + 1e-8
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    v = np.clip(v, 0, 1)
    w = np.clip(w, 0, 1)
    u = np.clip(1.0 - v - w, 0, 1)
    v = 1.0 - u - w
    return np.stack((u, v, w), 1)


class MeshResMapper(object):
    def __init__(self, v=None, f=None, orig_v=None, mapper_path=None, dtype=np.float32, dist_thresh=7e-3):
        """
        Given two aligned meshes (m and orig_m) with different triangulations, this class finds their mapping.
        Usually orig_m is the original mesh and m is a remeshed version of orig_m.
        When m deforms, the corresponding orig_m can be obtained using upsample().
        :param v, f: The vertex and face of m. Shape: (N, 3), (M, 3)
        :param orig_v: The vertex of orig_m. Shape: (N2, 3)
        :param mapper_path: a npz path that contains all the mapping parameters.
                            when it is provided, other arguments can be None. Such npz file can be saved using save().
        """
        if mapper_path is None:
            assert v is not None and orig_v is not None and f is not None
            if isinstance(v, torch.Tensor):
                v = v.detach().cpu().numpy()
            if isinstance(orig_v, torch.Tensor):
                orig_v = orig_v.detach().cpu().numpy()
            if isinstance(f, torch.Tensor):
                f = f.detach().cpu().numpy()
            orig_v = orig_v.astype(dtype)
            v = v.astype(dtype)
            f = f.astype(np.int32)

            nearest_dist, self.nearest_face = get_nearest_face(orig_v, v, f)
            self.inside = nearest_dist < dist_thresh

            import trimesh
            mesh = trimesh.Trimesh(vertices=orig_v[self.inside])
            mesh.export('test.ply')

            self.bary = barycentric(orig_v, v[f[:, 0][self.nearest_face]],
                                    v[f[:, 1][self.nearest_face]],
                                    v[f[:, 2][self.nearest_face]])
            # dist = np.sum(np.square(orig_v[:, None] - v[None]), 2)
            # self.nearest_s2v = np.argmin(dist, 0)
            # self.nearest_v2s = np.argmin(dist, 1)

            tree = KDTree(orig_v)
            self.nearest_s2v = tree.query(v)[1]
            tree = KDTree(v)
            self.nearest_v2s = tree.query(orig_v)[1]

            self.f = f
        else:
            c = np.load(mapper_path)
            self.nearest_face = c['nearest_face']
            self.bary = c['bary']
            self.nearest_s2v = c['s2v']
            self.nearest_v2s = c['v2s']
            self.f = c['f']
            self.inside = c['inside']

        self.f_torch = torch.from_numpy(self.f).long()
        self.nearest_face_torch = torch.from_numpy(self.nearest_face).long()
        self.bary_torch = torch.from_numpy(self.bary).float()

    def save(self, path):
        np.savez(path, nearest_face=self.nearest_face,
                 s2v=self.nearest_s2v, v2s=self.nearest_v2s,
                 bary=self.bary, f=self.f, inside=self.inside)

    def upsample(self, v):
        if isinstance(v, torch.Tensor):
            return self.upsample_torch(v)
        assert v.shape[0] == self.nearest_s2v.shape[0]
        rec_v = v[self.f[:, 0][self.nearest_face]] * self.bary[:, 0:1] + \
                v[self.f[:, 1][self.nearest_face]] * self.bary[:, 1:2] + \
                v[self.f[:, 2][self.nearest_face]] * self.bary[:, 2:3]
        return rec_v

    def upsample_arap(self, v, orig_f, orig_canonical_v):
        rec_v = self.upsample(v)
        # convert rec_v and f into an open3d mesh
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(orig_canonical_v)
        mesh.triangles = o3d.utility.Vector3iVector(orig_f)
        constraint_ids = np.where(self.inside)[0].tolist()
        constraint_ids = o3d.utility.IntVector(constraint_ids)
        constraint_pos = rec_v[constraint_ids]
        constraint_pos = o3d.utility.Vector3dVector(constraint_pos)

        deformed = mesh.deform_as_rigid_as_possible(constraint_ids, constraint_pos, max_iter=50)
        return np.asarray(deformed.vertices)

    def upsample_torch(self, v):
        assert v.shape[0] == self.nearest_s2v.shape[0]
        rec_v = v[self.f_torch[:, 0][self.nearest_face_torch]] * self.bary_torch[:, 0:1] + \
                v[self.f_torch[:, 1][self.nearest_face_torch]] * self.bary_torch[:, 1:2] + \
                v[self.f_torch[:, 2][self.nearest_face_torch]] * self.bary_torch[:, 2:3]
        return rec_v

    def to(self, device):
        self.f_torch = self.f_torch.to(device)
        self.nearest_face_torch = self.nearest_face_torch.to(device)
        self.bary_torch = self.bary_torch.to(device)