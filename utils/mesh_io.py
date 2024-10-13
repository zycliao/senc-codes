import os
import struct
from struct import pack, unpack

import torch
import numpy as np
import trimesh
from pytorch3d.structures import Meshes

from utils.common import pickle_load


def trimesh2pytorch3d(mesh, device=None):
    verts = mesh.vertices
    faces = mesh.faces
    verts = torch.from_numpy(verts).unsqueeze(0).float()
    faces = torch.from_numpy(faces).unsqueeze(0).long()
    if device is not None:
        verts = verts.to(device)
        faces = faces.to(device)
    mesh = Meshes(verts, faces)
    return mesh

def normalize_v3(arr):
    ''' Normalize a numpy array of 3 component vectors shape=(n,3) '''
    lens = np.sqrt(arr[:, 0] ** 2 + arr[:, 1] ** 2 + arr[:, 2] ** 2)
    eps = 0.00000001
    lens[lens < eps] = eps
    arr[:, 0] /= lens
    arr[:, 1] /= lens
    arr[:, 2] /= lens
    return arr


def compute_normal(vertices, faces):
    # Create a zeroed array with the same type and shape as our vertices i.e., per vertex normal
    norm = np.zeros(vertices.shape, dtype=vertices.dtype)
    # Create an indexed view into the vertex array using the array of three indices for triangles
    tris = vertices[faces]
    # Calculate the normal for all the triangles, by taking the cross product of the vectors v1-v0, and v2-v0 in each triangle
    n = np.cross(tris[::, 1] - tris[::, 0], tris[::, 2] - tris[::, 0])
    # n is now an array of normals per triangle. The length of each normal is dependent the vertices,
    # we need to normalize these, so that our next step weights each normal equally.
    normalize_v3(n)
    # now we have a normalized array of normals, one per triangle, i.e., per triangle normals.
    # But instead of one per triangle (i.e., flat shading), we add to each vertex in that triangle,
    # the triangles' normal. Multiple triangles would then contribute to every vertex, so we need to normalize again afterwards.
    # The cool part, we can actually add the normals through an indexed view of our (zeroed) per vertex normal array
    norm[faces[:, 0]] += n
    norm[faces[:, 1]] += n
    norm[faces[:, 2]] += n
    normalize_v3(norm)

    return norm

def save_obj_mesh(mesh_path, verts, faces):
    file = open(mesh_path, 'w')
    for v in verts:
        file.write('v %.4f %.4f %.4f\n' % (v[0], v[1], v[2]))
    for f in faces:
        f_plus = f + 1
        file.write('f %d %d %d\n' % (f_plus[0], f_plus[1], f_plus[2]))
    file.close()

def load_obj_mesh(mesh_file, with_normal=False, with_texture=False):
    vertex_data = []
    norm_data = []
    uv_data = []

    face_data = []
    face_norm_data = []
    face_uv_data = []

    if isinstance(mesh_file, str):
        f = open(mesh_file, "r")
    else:
        f = mesh_file
    for line in f:
        if isinstance(line, bytes):
            line = line.decode("utf-8")
        if line.startswith('#'):
            continue
        values = line.split()
        if not values:
            continue

        if values[0] == 'v':
            v = list(map(float, values[1:4]))
            vertex_data.append(v)
        elif values[0] == 'vn':
            vn = list(map(float, values[1:4]))
            norm_data.append(vn)
        elif values[0] == 'vt':
            vt = list(map(float, values[1:3]))
            uv_data.append(vt)

        elif values[0] == 'f':
            # quad mesh
            if len(values) > 4:
                f = list(map(lambda x: int(x.split('/')[0]), values[1:4]))
                face_data.append(f)
                f = list(map(lambda x: int(x.split('/')[0]), [values[3], values[4], values[1]]))
                face_data.append(f)
            # tri mesh
            else:
                f = list(map(lambda x: int(x.split('/')[0]), values[1:4]))
                face_data.append(f)

            # deal with texture
            if len(values[1].split('/')) >= 2:
                # quad mesh
                if len(values) > 4:
                    f = list(map(lambda x: int(x.split('/')[1]), values[1:4]))
                    face_uv_data.append(f)
                    f = list(map(lambda x: int(x.split('/')[1]), [values[3], values[4], values[1]]))
                    face_uv_data.append(f)
                # tri mesh
                elif len(values[1].split('/')[1]) != 0:
                    f = list(map(lambda x: int(x.split('/')[1]), values[1:4]))
                    face_uv_data.append(f)
            # deal with normal
            if len(values[1].split('/')) == 3:
                # quad mesh
                if len(values) > 4:
                    f = list(map(lambda x: int(x.split('/')[2]), values[1:4]))
                    face_norm_data.append(f)
                    f = list(map(lambda x: int(x.split('/')[2]), [values[3], values[4], values[1]]))
                    face_norm_data.append(f)
                # tri mesh
                elif len(values[1].split('/')[2]) != 0:
                    f = list(map(lambda x: int(x.split('/')[2]), values[1:4]))
                    face_norm_data.append(f)

    vertices = np.array(vertex_data)
    faces = np.array(face_data) - 1

    if with_texture and with_normal:
        uvs = np.array(uv_data)
        face_uvs = np.array(face_uv_data) - 1
        norms = np.array(norm_data)
        if norms.shape[0] == 0:
            norms = compute_normal(vertices, faces)
            face_normals = faces
        else:
            norms = normalize_v3(norms)
            face_normals = np.array(face_norm_data) - 1
        return vertices, faces, norms, face_normals, uvs, face_uvs

    if with_texture:
        uvs = np.array(uv_data)
        face_uvs = np.array(face_uv_data) - 1
        return vertices, faces, uvs, face_uvs

    if with_normal:
        norms = np.array(norm_data)
        norms = normalize_v3(norms)
        face_normals = np.array(face_norm_data) - 1
        return vertices, faces, norms, face_normals

    return vertices, faces


def writePC2(file, V, float16=False):
    file = str(file)
    assert (
        file.endswith(".pc2") and not float16 or file.endswith(".pc16") and float16
    ), "File format not consistent with specified input format"
    if float16:
        V = V.astype(np.float16)
    else:
        V = V.astype(np.float32)
    with open(file, "wb") as f:
        # Create the header
        headerFormat = "<12siiffi"
        headerStr = pack(
            headerFormat, b"POINTCACHE2\0", 1, V.shape[1], 0, 1, V.shape[0]
        )
        f.write(headerStr)
        # Write vertices
        f.write(V.tobytes())


def writePC2Frames(file, V, float16=False):
    assert (
        file.endswith(".pc2") and not float16 or file.endswith(".pc16") and float16
    ), "File format not consistent with specified input format"
    # Read file metadata (dimensions)
    if os.path.isfile(file):
        if float16:
            V = V.astype(np.float16)
        else:
            V = V.astype(np.float32)
        with open(file, "rb+") as f:
            # Num points
            f.seek(16)
            nPoints = unpack("<i", f.read(4))[0]
            assert len(V.shape) == 3 and V.shape[1] == nPoints, (
                "Inconsistent dimensions: "
                + str(V.shape)
                + " and should be (-1,"
                + str(nPoints)
                + ",3)"
            )
            # Read n. of samples
            f.seek(28)
            nSamples = unpack("<i", f.read(4))[0]
            # Update n. of samples
            nSamples += V.shape[0]
            f.seek(28)
            f.write(pack("i", nSamples))
            # Append new frame/s
            f.seek(0, 2)
            f.write(V.tobytes())
    else:
        writePC2(file, V, float16)


def read_pc2(path):
    with open(path, 'rb') as f:
        head_fmt = '<12siiffi'
        data_fmt = '<fff'
        head_unpack = struct.Struct(head_fmt).unpack_from
        data_unpack = struct.Struct(data_fmt).unpack_from
        data_size = struct.calcsize(data_fmt)
        headerStr = f.read(struct.calcsize(head_fmt))
        head = head_unpack(headerStr)
        nverts, nframes = head[2], head[5]
        data = []
        for i in range(nverts*nframes):
            data_line = f.read(data_size)
            if len(data_line) != data_size:
                return None
            data.append(list(data_unpack(data_line)))
        data = np.array(data).reshape([nframes, nverts, 3])
    return data


def save_as_pc2(sequence_path, save_dir, prefix='', save_mesh=False, garment_name='cloth'):
    traj_dict = pickle_load(sequence_path)
    cloth_pos = traj_dict['pred']
    cloth_faces = traj_dict['cloth_faces']
    body_pos = traj_dict['obstacle']
    body_faces = traj_dict['obstacle_faces']
    if prefix != '':
        prefix += '_'
    writePC2(os.path.join(save_dir, prefix + f'{garment_name}.pc2'), cloth_pos)
    writePC2(os.path.join(save_dir, prefix + 'body.pc2'), body_pos)

    if save_mesh:
        trimesh.Trimesh(vertices=cloth_pos[0], faces=cloth_faces, process=False).export(os.path.join(save_dir, prefix + f'{garment_name}.obj'))
        trimesh.Trimesh(vertices=body_pos[0], faces=body_faces, process=False).export(os.path.join(save_dir, prefix + 'body.obj'))
