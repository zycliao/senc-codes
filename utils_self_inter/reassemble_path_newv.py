from utils_self_inter.remesh_crsp import on_line

# assembled_v_one_batch, vs_left = check_left_newv(unique_vv, v, assembled_v_one_batch, vertex_edge_dict)
# assembled_v_one_batch: a vector of vector, every inner vector is a path
# vs_left: those vertices still left, example: {4496, 5054, 4805, 4934}
# new_edges: all the new edges produced: example: [(4530,, 4951), (2830, 3401), ...]
def reassemble_path_newv(assembled_v_one_batch, vv, vs_left, new_vertex_edge_dict, new_edges):
    new_assembled_v_one_batch = []
    for inter_path in assembled_v_one_batch:
        new_assembled_v_one_path = []
        for i in range(len(inter_path) - 1):
            new_assembled_v_one_path.append(inter_path[i])
            test_edge = tuple(sorted([inter_path[i + 1], inter_path[i]]))
            if test_edge not in new_edges:
                insert_v_candidate = []
                for v_left in vs_left:
                    if on_line(vv[inter_path[i + 1]], vv[v_left], vv[inter_path[i]]):
                        insert_v_candidate.append(v_left)
                while insert_v_candidate:
                    for j in range(len(insert_v_candidate)):
                        if insert_v_candidate[j] in new_vertex_edge_dict[new_assembled_v_one_path[-1]]:
                            new_assembled_v_one_path.append(insert_v_candidate[j])
                            insert_v_candidate.pop(j)
                            break
        new_assembled_v_one_path.append(inter_path[-1])
        new_assembled_v_one_batch.append(new_assembled_v_one_path)
    return new_assembled_v_one_batch