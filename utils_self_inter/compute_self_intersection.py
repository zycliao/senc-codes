from utils_self_inter.reassemble_all_v_one_batch import reassemble_all_v_one_batch
from utils_self_inter.remesh_crsp import remesh_crsp
from utils_self_inter.mesh import faces_to_edges_and_adjacency_in_progress, faces_to_edges_and_adjacency_in_progress_array
from utils_self_inter.check_left_newv import check_left_newv
from utils_self_inter.reassemble_path_newv import reassemble_path_newv
from utils_self_inter.find_inner_f_one_batch import find_inner_f_one_batch, find_inner_f_one_batch_path_seperate
from utils_self_inter.remesh_crsp import find_no_cross_edges
import numpy as np
import igl.copyleft.cgal
from utils_self_inter.remesh_crsp import on_line
import igl
import time
import cmake_example
# import ray
def find_penetration_triangles_one_batch(v, garment):
    f = garment.closed_faces
    old_face_adjacency = garment.closed_face_adjacency_dict
    t0 = time.time()
    (
        vv,
        ff,
        inter_f_pairs,
        birth_tri,
        unique_vv,
    ) = igl.copyleft.cgal.remesh_self_intersections(v, f)
    t1 = time.time()
    print("look here", t1 - t0)
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
    t2 = time.time()

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
    t3 = time.time()
    # print(bary_dict_one_batch)

    vs_left = check_left_newv(
        unique_vv, len(v), assembled_v_one_batch
    )
    t4 = time.time()
    print(vv.dtype)
    reassembled_v_one_batch = cmake_example.reassemble_path_newv(
        assembled_v_one_batch, vv, vs_left, vertex_edge_dict, new_edges
    )
    # print(reassembled_v_one_batch[1])
    # print(vs_left)
    # print(new_edges[0])
    t5 = time.time()
    no_cross_edges_one_batch = find_no_cross_edges(
        reassembled_v_one_batch
    )
    t6 = time.time()
    """ for i in range(len(assembled_v_one_batch)):
        print(str(i))
        print("i: ", assembled_i_one_batch[i])
        print("v: ", assembled_v_one_batch[i])
        print("re v: ", reassembled_v_one_batch[i])
        print() """
        # [77, 4505, 4424, 4491, 4490, 4492, 4501]
    # print(vv[4315], vv[4317])
    # print(vertex_edge_dict[5334])
    # print(5643 in vertex_edge_dict[5086], 5641 in vertex_edge_dict[5086])
    inner_f_one_batch_path_seperate, wrong_paths_idx, write_wrong = find_inner_f_one_batch_path_seperate(
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
    t7 = time.time()
    print("remesh time",t1-t0)
    print("faces_to_edges_and_adjacency_in_progress", t2-t1)
    print("remesh_crsp", t3-t2)
    print("check_left_newv", t4-t3)
    print("reassemble_path_newv", t5 - t4)
    print("find_no_cross_edges", t6 - t5)
    print("find_inner_f_one_batch_path_seperate", t7 - t6)
    return (
        # v,
        vv,
        bary_dict_one_batch,
        crsp_inter_vv,
        # f,
        ff,
        vertex_edge_dict,
        edge_to_face_adjacency,
        birth_tri,
        inner_f_one_batch_path_seperate,
        # list(original_faces_one_batch),
        vs_left,
        # unique_vv
    )


def compute_self_intersection_test(vertices, garment):
    ff_all_batches = []
    new_indices_all_batches = []
    new_v_bary_all_batches = []
    inner_f_all_batches = []
    original_faces_all_batches = []
    original_v_all_batches = []

    for i in range(len(vertices)):
        # find_penetraion_triangles_one_batch is time consuming1, reasseble is very fast!
        t0 = time.time()
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
        ) = find_penetration_triangles_one_batch(vertices[i], garment)
        t2 = time.time()
        (
            new_v_indices,
            new_v_bary,
        ) = reassemble_all_v_one_batch(
            vertices[i],
            vv,
            bary_dict_one_batch,
            crsp_inter_vv,
            garment.closed_faces,
            vertex_edge_dict,
            edge_to_face_adjacency,
            birth_tri,
            vs_left,
        )
        t1 = time.time()
        print("find_penetration_traingles_one_batch:", t2-t0)
        print("compute_self_collision total time",t1 - t0)
        original_v_all_batches.append(vertices[i].astype(np.float32))
        new_indices_all_batches.append(new_v_indices)
        new_v_bary_all_batches.append(new_v_bary)
        ff_all_batches.append(ff)
        inner_f_all_batches.append(inner_f_one_batch)
        # original_faces_all_batches.append(original_faces_one_batch)
    return (
        ff_all_batches,
        new_indices_all_batches,
        new_v_bary_all_batches,
        inner_f_all_batches,
        vv,
        original_v_all_batches
    )
# @ray.remote
def compute_self_intersection_one_batch(vertices, garment):
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
    ) = find_penetration_triangles_one_batch(vertices, garment)
    (
        new_v_indices,
        new_v_bary,
    ) = reassemble_all_v_one_batch(
        vertices,
        vv,
        bary_dict_one_batch,
        crsp_inter_vv,
        garment.closed_faces,
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

