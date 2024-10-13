from utils_self_inter.find_shared_vertex import find_shared_vertex
from utils_self_inter.remove import remove
from utils_self_inter.tri_tri_intersect_bary import *
from collections import defaultdict

def path_assembly_one_batch_igl(v, f, inter_f_pairs):
    graph_dict = defaultdict(list)
    bary_dict = dict()
    for tri_u, tri_v in inter_f_pairs:
        ids_u = f[tri_u]
        ids_v = f[tri_v]
        tri_u0 = v[ids_u[0]]
        tri_u1 = v[ids_u[1]]
        tri_u2 = v[ids_u[2]]
        if (len(set([ids_u[0], ids_u[1], ids_u[2], ids_v[0], ids_v[1], ids_v[2]])) == 6
        ):
            tri_v0 = v[ids_v[0]]
            tri_v1 = v[ids_v[1]]
            tri_v2 = v[ids_v[2]]
            intersection_points = find_intersection_point_bary(
                tri_u0, tri_u1, tri_u2,
                tri_v0, tri_v1, tri_v2,
                ids_u, ids_v,
                tri_u, tri_v,
                bary_dict
            )
            if len(intersection_points) == 2:
                graph_dict[intersection_points[0]].append(
                    intersection_points[1]
                )
                graph_dict[intersection_points[1]].append(
                    intersection_points[0]
                )
        # loop vertex case
        elif (
            len(set([ids_u[0], ids_u[1], ids_u[2], ids_v[0], ids_v[1], ids_v[2]])) == 5
        ):
            ids_u_s, ids_v_s, shared_v_idx = find_shared_vertex(
                ids_u, ids_v
            )
            tri_u0_s = v[ids_u_s[0]]
            tri_u1_s = v[ids_u_s[1]]
            tri_v0_s = v[ids_v_s[0]]
            tri_v1_s = v[ids_v_s[1]]
            shared_v_pos = v[shared_v_idx]
            intersection_points = find_intersection_point_share_one_v_bary(
                tri_u0_s, tri_u1_s, shared_v_pos,
                tri_v0_s, tri_v1_s,
                ids_u_s, ids_v_s,
                tri_u, tri_v,
                bary_dict,
                ids_u, ids_v
            )
            # Only consider the case when two triangles have 2 different intersection points
            if len(intersection_points) == 1:
                graph_dict[shared_v_idx].append(
                    intersection_points[0]
                )
                graph_dict[intersection_points[0]].append(
                    shared_v_idx
                )

    assembled_i_one_batch = []
    while graph_dict:
        for key in graph_dict:
            curr_intersect_point = key
            if len(graph_dict[curr_intersect_point]) == 1:
                break
        assembled_i_one_path = []
        assembled_i_one_path.append(curr_intersect_point)
        while curr_intersect_point in graph_dict:
            next_intersect_point = graph_dict[curr_intersect_point][0]
            assembled_i_one_path.append(next_intersect_point)
            remove(graph_dict[curr_intersect_point], next_intersect_point)
            remove(graph_dict[next_intersect_point], curr_intersect_point)
            if not graph_dict[curr_intersect_point]:
                del graph_dict[curr_intersect_point]
            if not graph_dict[next_intersect_point]:
                del graph_dict[next_intersect_point]
            curr_intersect_point = next_intersect_point
        assembled_i_one_batch.append(assembled_i_one_path)

    return (
        assembled_i_one_batch,
        bary_dict
    )