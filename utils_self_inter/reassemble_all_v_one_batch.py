import numpy as np
from utils_self_inter.tri_tri_intersect_bary import compute_barycentric


def reassemble_all_v_one_batch(
    v,
    vv,
    bary_dict_one_batch,
    crsp_inter_vv,
    f,
    vertex_edge_dict,
    edge_to_face_adjacency,
    birth_tri,
    vs_left
):
    num_old_vertices = len(v)
    new_v_indices = np.zeros((len(vv) - len(v), 3), dtype=np.int32)
    new_v_bary = np.zeros((len(vv) - len(v), 3), dtype=np.float32)
    for key_crsp_vv in bary_dict_one_batch:
        new_v_indices[crsp_inter_vv[key_crsp_vv] - num_old_vertices] = f[key_crsp_vv[2]]
        new_v_bary[crsp_inter_vv[key_crsp_vv] - num_old_vertices] = bary_dict_one_batch[
            key_crsp_vv
        ]
    # find those new vertices with three tri intersection's bary
    for v_left in vs_left:
        neighbor_v_left = vertex_edge_dict[v_left][0]
        face_containing_v = edge_to_face_adjacency[
            tuple(sorted([neighbor_v_left, v_left]))
        ][0][0]
        one_of_birth_tri = birth_tri[face_containing_v]
        new_v_indices[v_left - num_old_vertices] = f[one_of_birth_tri]
        new_v_bary[v_left - num_old_vertices] = compute_barycentric(
            vv[v_left],
            v[f[one_of_birth_tri][0]],
            v[f[one_of_birth_tri][1]],
            v[f[one_of_birth_tri][2]],
        )
    return new_v_indices, new_v_bary