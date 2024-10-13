import numpy as np
from scipy.sparse import coo_matrix
import time
from collections import defaultdict


def triangulate(faces):
    triangles = np.int32(
        [triangle for polygon in faces for triangle in _triangulate_recursive(polygon)]
    )
    return triangles


def _triangulate_recursive(face):
    if len(face) == 3:
        return [face]
    else:
        return [face[:3]] + _triangulate_recursive([face[0], *face[2:]])


def faces_to_edges_and_adjacency(faces):
    vertex_edge_dict = dict()
    edges = dict()
    adjacency_edge_idx = dict()
    face_adjacency_dict = dict()
    for fidx, face in enumerate(faces):
        for i, v in enumerate(face):
            nv = face[(i + 1) % len(face)]
            edge = tuple(sorted([v, nv]))
            if not edge in edges:
                edges[edge] = []
                vertex_edge_dict.setdefault(v, []).append(nv)
                vertex_edge_dict.setdefault(nv, []).append(v)
            edges[edge] += [fidx]
    face_adjacency = []
    face_adjacency_edges = []
    boundary_edges_dict = dict()
    for edge, face_list in edges.items():
        if len(face_list) == 1:
            boundary_edges_dict.setdefault(edge[0], []).append(edge[1])
            boundary_edges_dict.setdefault(edge[1], []).append(edge[0])
        for i in range(len(face_list) - 1):
            for j in range(i + 1, len(face_list)):
                face_adjacency += [[face_list[i], face_list[j]]]
                face_adjacency_dict.setdefault(face_list[i], []).append(face_list[j])
                face_adjacency_dict.setdefault(face_list[j], []).append(face_list[i])
                face_adjacency_edges += [edge]
                adjacency_edge_idx.setdefault(edge, []).append(len(face_adjacency) - 1)
    edges_np = np.array([list(edge) for edge in edges.keys()], np.int32)
    face_adjacency = np.array(face_adjacency, np.int32)
    face_adjacency_edges = np.array(face_adjacency_edges, np.int32)
    return (
        edges_np,
        face_adjacency,
        face_adjacency_edges,
        vertex_edge_dict,
        adjacency_edge_idx,
        boundary_edges_dict,
        face_adjacency_dict
    )


def faces_to_edges_and_adjacency_in_progress(faces):

    face_adjencency_dict = dict()
    face_adjacency_to_edge_dict = dict()   
    vertex_edge_dict = dict()
    edge_to_face_adjencency = dict()
    edges = dict()

    for fidx, face in enumerate(faces):
        for i, v in enumerate(face):
            nv = face[(i + 1) % len(face)]
            edge = tuple(sorted([v, nv]))
            if not edge in edges:
                edges[edge] = []
                vertex_edge_dict.setdefault(v, []).append(nv)
                vertex_edge_dict.setdefault(nv, []).append(v)
            edges[edge] += [fidx]

    for edge, face_list in edges.items():
        for i in range(len(face_list) - 1):
            for j in range(i + 1, len(face_list)):
                edge_to_face_adjencency.setdefault(edge, []).append((face_list[i], face_list[j]))
                face_adjencency_dict.setdefault(face_list[i], []).append(face_list[j])
                face_adjencency_dict.setdefault(face_list[j], []).append(face_list[i])
                face_adjacency_to_edge_dict[(face_list[i], face_list[j])] = edge
    # edges_np = np.array([list(edge) for edge in edges.keys()], np.int32)
    edges_list = [edge for edge in edges.keys()]
    return (
        edges_list,
        face_adjencency_dict,
        face_adjacency_to_edge_dict,
        vertex_edge_dict,
        edge_to_face_adjencency
    )
def edge_in_edges(edge, edges):
    for e in edges:
        if e == edge:
            return True
        elif e == (-1, -1):
            return False
def faces_to_edges_and_adjacency_in_progress_array(faces, vertices_num):

    face_adjacency_dict = [[] for i in range(faces.shape[0])]
    vertex_edge_dict = [[] for i in range(vertices_num)]
    face_adjacency_to_edge_dict = dict()
    edge_to_face_adjencency = dict()
    edges = dict()
    for fidx, face in enumerate(faces):
        for i, v in enumerate(face):
            nv = face[(i + 1) % len(face)]
            edge = tuple(sorted([v, nv]))
            if not edge in edges:
                edges[edge] = []
                vertex_edge_dict[v].append(nv)
                vertex_edge_dict[nv].append(v)
            edges[edge] += [fidx]

    for edge, face_list in edges.items():
        for i in range(len(face_list) - 1):
            for j in range(i + 1, len(face_list)):
                edge_to_face_adjencency.setdefault(edge, []).append((face_list[i], face_list[j]))
                # face_adjencency_dict.setdefault(face_list[i], []).append(face_list[j])
                # face_adjencency_dict.setdefault(face_list[j], []).append(face_list[i])
                face_adjacency_dict[face_list[i]].append(face_list[j])
                face_adjacency_dict[face_list[j]].append(face_list[i])
                face_adjacency_to_edge_dict[(face_list[i], face_list[j])] = edge
    # edges_np = np.array([list(edge) for edge in edges.keys()], np.int32)

    edges_list = [edge for edge in edges.keys()]
    return (
        edges_list,
        face_adjacency_dict,
        face_adjacency_to_edge_dict,
        vertex_edge_dict,
        edge_to_face_adjencency
    )

def get_vertex_edge_dict_and_face_adj_dict(faces):
   
    vertex_edge_dict = dict()
    face_adjencency_dict = dict()
    edges = dict()

    for fidx, face in enumerate(faces):
        for i, v in enumerate(face):
            nv = face[(i + 1) % len(face)]
            edge = tuple(sorted([v, nv]))
            if not edge in edges:
                edges[edge] = []
                vertex_edge_dict.setdefault(v, []).append(nv)
                vertex_edge_dict.setdefault(nv, []).append(v)
            edges[edge] += [fidx]
    for edge, face_list in edges.items():
        for i in range(len(face_list) - 1):
            for j in range(i + 1, len(face_list)):
                face_adjencency_dict.setdefault(face_list[i], []).append(face_list[j])
                face_adjencency_dict.setdefault(face_list[j], []).append(face_list[i])

    return vertex_edge_dict, face_adjencency_dict


def compute_boundary_paths(boundary_edges_dict):
    boundary_paths = []
    while boundary_edges_dict:
        for key in boundary_edges_dict:
            curr_p = key
            break
        boundary_path = []
        while curr_p in boundary_edges_dict:
            boundary_path.append(curr_p)
            next_p = boundary_edges_dict[curr_p][0]
            boundary_edges_dict[curr_p].remove(next_p)
            boundary_edges_dict[next_p].remove(curr_p)
            if not boundary_edges_dict[curr_p]:
                del boundary_edges_dict[curr_p]
            if not boundary_edges_dict[next_p]:
                del boundary_edges_dict[next_p]
            curr_p = next_p
        boundary_paths.append(boundary_path)
    return boundary_paths


def get_closed_garment(boundary_paths, vertices, faces):
    for boundary in boundary_paths:
        new_v = np.mean(vertices[boundary], axis=0)
        vertices = np.vstack((vertices, new_v))
        v_idx = len(vertices) - 1
        for i, v in enumerate(boundary):
            nv = boundary[(i + 1) % len(boundary)]
            faces = np.vstack((faces, np.array([nv, v, v_idx])))
    return vertices, faces

def find_closed_four_points(vertex_edge_dict):
    rank = [0, 0, 0, 0]
    rank_v = [-1, -1, -1, -1]
    for v in vertex_edge_dict:
        num_neighbors = len(vertex_edge_dict[v])
        if num_neighbors > rank[0]:
            rank[3] = rank[2]
            rank[2] = rank[1]
            rank[1] = rank[0]
            rank[0] = num_neighbors

            rank_v[3] = rank_v[2]
            rank_v[2] = rank_v[1]
            rank_v[1] = rank_v[0]
            rank_v[0] = v
        elif num_neighbors > rank[1]:
            rank[3] = rank[2]
            rank[2] = rank[1]
            rank[1] = num_neighbors

            rank_v[3] = rank_v[2]
            rank_v[2] = rank_v[1]
            rank_v[1] = v
        elif num_neighbors > rank[2]:
            rank[3] = rank[2]
            rank[2] = num_neighbors

            rank_v[3] = rank_v[2]
            rank_v[2] = v
        elif num_neighbors > rank[3]:
            rank[3] = num_neighbors

            rank_v[3] = v
    return rank, rank_v

def find_faces_to_delete(closed_v, closed_faces):
    faces_to_delete = []
    for i in range(closed_faces.shape[0]):
        for j in range(len(closed_v)):
            if closed_v[j] in closed_faces[i]:
                faces_to_delete.append(i)
                break
    return faces_to_delete

def v_to_faces(faces):
    v_to_faces = dict()
    for fidx, face in enumerate(faces):
        if (face[0] not in v_to_faces) or (fidx not in v_to_faces[face[0]]):
            v_to_faces.setdefault(face[0], []).append(fidx)
        if (face[1] not in v_to_faces) or (fidx not in v_to_faces[face[1]]):
            v_to_faces.setdefault(face[1], []).append(fidx)
        if (face[2] not in v_to_faces) or (fidx not in v_to_faces[face[2]]):
            v_to_faces.setdefault(face[2], []).append(fidx)
    return v_to_faces


def laplacian_matrix(faces):
    G = {}
    for face in faces:
        for i, v in enumerate(face):
            nv = face[(i + 1) % len(face)]
            if v not in G:
                G[v] = {}
            if nv not in G:
                G[nv] = {}
            G[v][nv] = 1
            G[nv][v] = 1
    return graph_laplacian(G)


def graph_laplacian(graph):
    row, col, data = [], [], []
    for v in graph:
        n = len(graph[v])
        row += [v] * n
        col += [u for u in graph[v]]
        data += [1.0 / n] * n
    print(len(row), len(col), len(data), len(graph))
    return coo_matrix((data, (row, col)), shape=[len(graph)] * 2)


def edge_lengths(vertices, edges):
    return np.linalg.norm(vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=-1)


def dihedral_angle_adjacent_faces(normals, adjacency):
    normals0 = normals[adjacency[:, 0]]
    normals1 = normals[adjacency[:, 1]]
    cos = np.einsum("ab,ab->a", normals0, normals1)
    sin = np.linalg.norm(np.cross(normals0, normals1), axis=-1)
    return np.arctan2(sin, cos)


def vertex_area(vertices, faces):
    v01 = vertices[faces[:, 1]] - vertices[faces[:, 0]]
    v12 = vertices[faces[:, 2]] - vertices[faces[:, 1]]
    face_areas = np.linalg.norm(np.cross(v01, v12), axis=-1)
    vertex_areas = np.zeros((vertices.shape[0],), np.float32)
    for i, face in enumerate(faces):
        vertex_areas[face] += face_areas[i]
    vertex_areas *= 1 / 6
    total_area = vertex_areas.sum()
    return vertex_areas, face_areas, total_area

