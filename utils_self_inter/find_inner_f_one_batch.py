from utils_self_inter.find_start_faces import find_start_faces
from utils_self_inter.parallel_flood_fill import parallel_flood_fill_face


def find_inner_f_one_batch(
    v,
    f,
    no_cross_edges_one_batch,
    ff,
    edge_to_face_adjacency,
    birth_tri,
    face_adjacency_dict,
    face_adjacency_to_edge_dict,
    old_face_adjacency,
):
    inner_f_one_batch = []
    write_wrong = False
    wrong_paths_idx = []
    for i in range(len(no_cross_edges_one_batch)):
        no_cross_edges_one_path = no_cross_edges_one_batch[i]
        f_colors = [0 for i in range(ff.shape[0])]
        no_start_edge = []
        stfs_and_start_edge = find_start_faces(
            v,
            f,
            f_colors,
            no_cross_edges_one_path,
            no_start_edge,
            edge_to_face_adjacency,
            birth_tri,
        )
        while stfs_and_start_edge:
            stf0, stf1 = stfs_and_start_edge[0]
            parallel_flood_fill_result = parallel_flood_fill_face(
                f_colors,
                stf0,
                stf1,
                face_adjacency_dict,
                face_adjacency_to_edge_dict,
                no_cross_edges_one_path,
                birth_tri,
                old_face_adjacency,
            )
            if parallel_flood_fill_result:
                inner_f_one_batch += parallel_flood_fill_result
            else:
                no_start_edge.append(stfs_and_start_edge[1])
                write_wrong = True
                wrong_paths_idx.append(i)
                print(str(i) + " th path has problem !!!")
                return inner_f_one_batch, wrong_paths_idx, write_wrong
            # print("length of inner f", len(set(inner_f_one_batch)))
            stfs_and_start_edge = find_start_faces(
                v,
                f,
                f_colors,
                no_cross_edges_one_path,
                no_start_edge,
                edge_to_face_adjacency,
                birth_tri,
            )
    return inner_f_one_batch, wrong_paths_idx, write_wrong


def find_inner_f_one_batch_path_seperate(
    v,
    f,
    no_cross_edges_one_batch,
    ff,
    edge_to_face_adjacency,
    birth_tri,
    face_adjacency_dict,
    face_adjacency_to_edge_dict,
    old_face_adjacency,
):
    inner_f_one_batch = []
    write_wrong = False
    wrong_paths_idx = []
    for i in range(len(no_cross_edges_one_batch)):
        inner_f_one_path = []
        no_cross_edges_one_path = no_cross_edges_one_batch[i]
        f_colors = [0 for i in range(ff.shape[0])]
        no_start_edge = []
        stfs_and_start_edge = find_start_faces(
            v,
            f,
            f_colors,
            no_cross_edges_one_path,
            no_start_edge,
            edge_to_face_adjacency,
            birth_tri,
        )
        while stfs_and_start_edge:
            stf0, stf1 = stfs_and_start_edge[0]
            parallel_flood_fill_result = parallel_flood_fill_face(
                f_colors,
                stf0,
                stf1,
                face_adjacency_dict,
                face_adjacency_to_edge_dict,
                no_cross_edges_one_path,
                birth_tri,
                old_face_adjacency,
            )
            if parallel_flood_fill_result:
                inner_f_one_path += parallel_flood_fill_result
            else:
                no_start_edge.append(stfs_and_start_edge[1])
                write_wrong = True
                wrong_paths_idx.append(i)
                print(str(i) + " th path has problem !!!")
                return inner_f_one_batch, wrong_paths_idx, write_wrong
            # print("length of inner f", len(set(inner_f_one_batch)))
            stfs_and_start_edge = find_start_faces(
                v,
                f,
                f_colors,
                no_cross_edges_one_path,
                no_start_edge,
                edge_to_face_adjacency,
                birth_tri,
            )
        if list(set(inner_f_one_path)):
            inner_f_one_batch.append(list(set(inner_f_one_path)))
    return inner_f_one_batch, wrong_paths_idx, write_wrong