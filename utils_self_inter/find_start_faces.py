import warnings

def find_start_faces(
    v,
    f,
    f_colors,
    no_cross_edges_one_path,
    no_start_edge,
    edge_to_face_adjacency,
    birth_tri,
):
    for no_cross_edge in no_cross_edges_one_path:
        if no_cross_edge not in no_start_edge:
            if no_cross_edge in edge_to_face_adjacency:
                for adjacency_2f in edge_to_face_adjacency[no_cross_edge]:
                    stf0 = adjacency_2f[0]
                    stf1 = adjacency_2f[1]
                    # print(stf0)
                    # print(stf1)
                    if birth_tri[stf0] == birth_tri[stf1] and (
                        f_colors[stf0] == 0
                        and f_colors[stf1] == 0
                        # or f_colors[stf0] == 2
                        # and f_colors[stf1] == 0
                        # or f_colors[stf0] == 0
                        # and f_colors[stf1] == 2
                    ):
                        # exit(0)
                        return (stf0, stf1), no_cross_edge
            else:
                warnings.warn(f"intersection path segment {no_cross_edge} is not an edge of the remeshed garment")
    return False
