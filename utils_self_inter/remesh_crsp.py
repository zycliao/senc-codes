from utils_self_inter.path_assembly_one_batch import path_assembly_one_batch_igl
import numpy as np
from collections import defaultdict

def find_crsp(crsp_inter_vv, split_triangles, original_vertices, vv, ff, inter_p, pos):
    for split_tri in split_triangles:
        for i in range(3):
            if ff[split_tri][i] not in original_vertices:
                test_pos = vv[ff[split_tri][i]]
                if np.linalg.norm(pos - test_pos) <= 2e-11:
                    crsp_inter_vv[inter_p] = ff[split_tri][i]
                    return


def remesh_crsp(
    v,
    f,
    vv,
    ff,
    inter_f_pairs,
    birth_tri,
    unique_vv,
    vertex_edge_dict
):

    # look for triangels that the original one split into
    split_tris = defaultdict(list)
    for i in range(len(birth_tri)):
        split_tris[birth_tri[i]].append(i)

    (
        assembled_i_one_batch,
        bary_dict_one_batch,
        # loop_vertices_one_batch,
        # normal_paths_one_batch
    ) = path_assembly_one_batch_igl(v, f, inter_f_pairs)

    crsp_inter_vv = dict()
    for inter_p in bary_dict_one_batch:
        birth_f_of_p = inter_p[2]
        a, b, c = bary_dict_one_batch[inter_p]
        pos = (
            a * v[f[birth_f_of_p][0]]
            + b * v[f[birth_f_of_p][1]]
            + c * v[f[birth_f_of_p][2]]
        )
        find_crsp(
            crsp_inter_vv,
            split_tris[birth_f_of_p],
            f[birth_f_of_p],
            vv,
            ff,
            inter_p,
            pos,
        )

    assembled_v_one_batch = []
    for path in assembled_i_one_batch:
        assembled_v_one_path = []
        for p in path:
            if isinstance(p, tuple):
                assembled_v_one_path.append(crsp_inter_vv[p])
            else:
                assembled_v_one_path.append(p)
        assembled_v_one_batch.append(assembled_v_one_path)
    # connect paths if needed
    remove_list = []
    for i in range(len(assembled_v_one_batch)):
        for j in range(i + 1, len(assembled_v_one_batch)):
            if assembled_v_one_batch[i][0] == assembled_v_one_batch[j][-1]:
                assembled_v_one_batch[j].pop(-1)
                assembled_v_one_batch[j] += assembled_v_one_batch[i]
                remove_list.append(i)
            elif assembled_v_one_batch[i][-1] == assembled_v_one_batch[j][0]:
                assembled_v_one_batch[i].pop(-1)
                assembled_v_one_batch[i] += assembled_v_one_batch[j]
                remove_list.append(j)
            elif assembled_v_one_batch[i][0] == assembled_v_one_batch[j][0]:
                assembled_v_one_batch[j].pop(0)
                assembled_v_one_batch[i] = (assembled_v_one_batch[i])[::-1] + assembled_v_one_batch[j]
                remove_list.append(j)
            elif assembled_v_one_batch[i][-1] == assembled_v_one_batch[j][-1]:
                assembled_v_one_batch[j].pop(-1)
                assembled_v_one_batch[i] = assembled_v_one_batch[j] + (assembled_v_one_batch[i])[::-1]
                remove_list.append(j)
                
    new_assembled_v_one_batch = []
    for i in range(len(assembled_v_one_batch)):
        if i not in remove_list:
            new_assembled_v_one_batch.append(assembled_v_one_batch[i])

    return (
        # assembled_i_one_batch,
        new_assembled_v_one_batch,
        bary_dict_one_batch,
        # loop_vertices_one_batch,
        # normal_paths_one_batch,
        crsp_inter_vv,
    )


def find_no_cross_edges(assembled_v_one_batch):
    no_cross_edges_one_batch = []
    for path in assembled_v_one_batch:
        no_cross_edges_one_path = []
        for i in range(len(path) - 1):
            no_cross_edges_one_path.append(tuple(sorted([path[i], path[i + 1]])))
        no_cross_edges_one_batch.append(no_cross_edges_one_path)
    return no_cross_edges_one_batch


def on_line(v0, v1, v2):
    vec0 = v1 - v0
    vec1 = v2 - v1
    if np.linalg.norm(np.cross(vec0, vec1)) < 1e-16 and np.dot(vec0, vec1) > 0:
        return True
    else:
        return False
