from utils_self_inter.remesh_crsp import remesh_crsp
from utils_self_inter.check_left_newv import check_left_newv
from utils_self_inter.reassemble_path_newv import reassemble_path_newv
from utils_self_inter.find_inner_f_one_batch import find_inner_f_one_batch_path_seperate
from utils_self_inter.remesh_crsp import find_no_cross_edges
import igl.copyleft.cgal
import numpy as np
import cmake_example

from utils_self_inter.reassemble_all_v_one_batch import reassemble_all_v_one_batch
def find_penetration_triangles_one_batch(v, f, old_face_adjacency):
    (
        vv,
        ff,
        inter_f_pairs,
        birth_tri,
        unique_vv,
    ) = igl.copyleft.cgal.remesh_self_intersections(v, f)
    ff = ff.astype(np.int32)
    inter_f_pairs = inter_f_pairs.astype(np.int32)
    birth_tri = birth_tri.astype(np.int32)
    unique_vv = unique_vv.astype(np.int32)

    for i in range(len(ff)):
        for j in range(3):
            ff[i][j] = unique_vv[ff[i][j]]

    (
        new_edges,
        face_adjacency_dict,
        face_adjacency_to_edge_dict,
        vertex_edge_dict,
        edge_to_face_adjacency,
    ) = cmake_example.faces_to_edges_and_adjacency_in_progress_array(ff, vv.shape[0])
    
    (
        assembled_v_one_batch,
        bary_dict_one_batch,
        crsp_inter_vv,
    ) = remesh_crsp(
        v,
        f,
        vv,
        ff,
        inter_f_pairs,
        birth_tri,
        unique_vv,
        vertex_edge_dict
    )
    vs_left = check_left_newv(
        unique_vv, len(v), assembled_v_one_batch
    )
    assembled_v_one_batch = cmake_example.reassemble_path_newv(
        assembled_v_one_batch, vv, vs_left, vertex_edge_dict, new_edges
    )
    no_cross_edges_one_batch = find_no_cross_edges(
        assembled_v_one_batch
    )
    inner_f_one_batch, wrong_paths_idx, write_wrong = find_inner_f_one_batch_path_seperate(
        v,
        f,
        no_cross_edges_one_batch,
        ff,
        edge_to_face_adjacency,
        birth_tri,
        face_adjacency_dict,
        face_adjacency_to_edge_dict,
        old_face_adjacency,
    )
    return (
        vv,
        bary_dict_one_batch,
        crsp_inter_vv,
        ff,
        vertex_edge_dict,
        edge_to_face_adjacency,
        birth_tri,
        inner_f_one_batch,
        vs_left,
    )

def compute_self_intersection_one_batch(vertices, faces, old_face_adjacency):
    (
        vv,
        bary_dict_one_batch,
        crsp_inter_vv,
        ff,
        vertex_edge_dict,
        edge_to_face_adjacency,
        birth_tri,
        inner_f_one_batch,
        vs_left,
    ) = find_penetration_triangles_one_batch(
        vertices, faces, old_face_adjacency
    )

    (
        new_v_indices,
        new_v_bary,
    ) = reassemble_all_v_one_batch(
        vertices,
        vv,
        bary_dict_one_batch,
        crsp_inter_vv,
        faces,
        vertex_edge_dict,
        edge_to_face_adjacency,
        birth_tri,
        vs_left,
    )
    return (
        ff,
        new_v_indices,
        new_v_bary,
        inner_f_one_batch,
    )

