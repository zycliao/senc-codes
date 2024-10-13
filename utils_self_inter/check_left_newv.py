def check_left_newv(unique_vv, num_v, assembled_v_one_batch):
    set_unique_vv = set(unique_vv)
    all_new_vs = set_unique_vv - set([i for i in range(num_v)])
    v_on_paths = set()
    for path_v in assembled_v_one_batch:
        v_on_paths = v_on_paths.union(set(path_v))

    vs_left = all_new_vs - v_on_paths
    return vs_left
    # len(set_unique_vv) is the total number of vertices
