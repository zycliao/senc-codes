from utils_self_inter.parallel_flood_fill import *
from utils_self_inter.find_faces import *
from utils_self_inter.split_vertices import split_vertices

def equal(a, b):
    if type(a) != type(b):
        return False
    else:
        return a==b


def mesh_coloring_one_batch(
    no_cross_edges_one_batch,
    no_reach_vertices_one_batch,
    assembled_i_one_batch,
    garment_vertex_edge_dict,
    garment_vertices,
    garment_v_to_faces,
    garment_faces,
):
    num_paths = 0
    v_and_f_in_intersection_path_one_batch = []
    for n_th_path in range(len(no_cross_edges_one_batch)):
        vertices_side_1 = []
        vertices_side_2 = []

        path = no_cross_edges_one_batch[n_th_path]
        vert = no_reach_vertices_one_batch[n_th_path]
        assembled_i_one_path = assembled_i_one_batch[n_th_path]
        #print("path len:", len(path))
        #print("vert :", vert)
        #print(len(assembled_i_one_path))


        # boundary loop
        #if (len(vert) == 0) or (len(vert)==2 and equal(assembled_i_one_path[0],vert[0]) and equal(assembled_i_one_path[-1],vert[1])):
        if not vert:
            no_cross_edges_side_1 = set(path)
        # start from a edge that only intersects exactly once with tri
        v_colors = [0 for i in range(garment_vertices.shape[0])]
        for edge in path:
            first_start_edge = edge
            if (
                path.count(first_start_edge) == 1
                and (first_start_edge[0] not in vert)
                and (first_start_edge[1] not in vert)
            ):
                break
        # print("start edge:", start_edge)
        vertices_side_1, visited_no_cross_edges = parallel_flood_fill(
            v_colors,
            first_start_edge[0],
            first_start_edge[1],
            garment_vertex_edge_dict,
            path,
            vert,
        )
        # print("vertices side 1 number:", len(vertices_side_1))
        # print("visited no cross edges:", visited_no_cross_edges)
        for visited_edge in visited_no_cross_edges:
            for i in range(path.count(visited_edge)):
                path.remove(visited_edge)
        if not vert:
            no_cross_edges_side_2 = set(path)
            no_cross_edges_side_1 = no_cross_edges_side_1 - no_cross_edges_side_2
        second_start_edge = False
        if len(path) > 0 and not vert:
            for edge in path:
                if path.count(edge) == 1:
                    second_start_edge = edge
                    break
            if second_start_edge:
                num_paths += 1
                v_colors = [0 for i in range(garment_vertices.shape[0])]
                vertices_side_2, visited_no_cross_edges = parallel_flood_fill(
                    v_colors,
                    second_start_edge[0],
                    second_start_edge[1],
                    garment_vertex_edge_dict,
                    path,
                    vert,
                )
        if vert:
            # print("loop vertices:", vert)
            vertices_side_1, vertices_side_2, partition_path = split_vertices(
                vertices_side_1,
                vert[0],
                vert[1],
                garment_vertices,
                garment_vertex_edge_dict,
            )

        # get faces on two sides
        faces_side_1 = []
        faces_side_2 = []
        if len(vertices_side_1) != 0:
            if vert:
                faces_side_2 = find_one_side_faces_loop(
                    vertices_side_2,
                    vertices_side_1,
                    partition_path,
                    vert,
                    garment_v_to_faces,
                    garment_faces,
                )
            else:
                if vertices_side_2:
                    faces_side_2 = find_one_side_faces(
                        vertices_side_2, garment_v_to_faces, garment_faces
                    )
                else:
                    faces_side_2 = find_faces_from_intersect_edges(
                        no_cross_edges_side_1, assembled_i_one_path
                    )

        if len(vertices_side_2) != 0:
            if vert:
                faces_side_1 = find_one_side_faces_loop(
                    vertices_side_1,
                    vertices_side_2,
                    partition_path,
                    vert,
                    garment_v_to_faces,
                    garment_faces,
                )
            else:
                if vertices_side_1:
                    faces_side_1 = find_one_side_faces(
                        vertices_side_1, garment_v_to_faces, garment_faces
                    )
                else:
                    faces_side_1 = find_faces_from_intersect_edges(
                        no_cross_edges_side_2, assembled_i_one_path
                    )
        v_and_f_in_intersection_path_one_batch.append(
            [vertices_side_1, faces_side_1, vertices_side_2, faces_side_2]
        )
        num_paths += 1
        #print(v_and_f_in_intersection_path_one_batch)
        # print(len(vertices_side_1))
        # print(len(vertices_side_2))
        #else:
            #v_and_f_in_intersection_path_one_batch.append([[],[],[],[]])
    return v_and_f_in_intersection_path_one_batch


# boundary loop start/end
"""         if len(vert) == 1 and (equal(assembled_i_one_path[0],vert[0]) or equal(assembled_i_one_path[-1],vert[0])):
            vert_and_its_neighbors = []
            for neighbor_vert in garment_vertex_edge_dict[vert[0]]:
                vert_and_its_neighbors.append(neighbor_vert)
            vert_and_its_neighbors.append(vert[0])
            v_colors = [0 for i in range(garment_vertices.shape[0])]
            for edge in path:
                first_start_edge = edge
                if (
                    path.count(first_start_edge) == 1
                    and (((first_start_edge[0] in vert_and_its_neighbors)
                    and (first_start_edge[1] not in vert_and_its_neighbors))
                    or  ((first_start_edge[0] not in vert_and_its_neighbors)
                    and (first_start_edge[1]  in vert_and_its_neighbors)))
                ):
                    break
            parallel_flood_fill_result = parallel_flood_fill_one_loop_end(
                v_colors,
                first_start_edge[0],
                first_start_edge[1],
                garment_vertex_edge_dict,
                path,
                vert_and_its_neighbors,
            )
            while not isinstance(parallel_flood_fill_result, tuple):
                #if len(path)>0:
                    #print(path)
                v_colors = [0 for i in range(garment_vertices.shape[0])]
                path.remove(first_start_edge)

                for edge in path:
                    first_start_edge = edge
                    if (
                        path.count(first_start_edge) == 1
                        and (((first_start_edge[0] in vert_and_its_neighbors)
                        and (first_start_edge[1] not in vert_and_its_neighbors))
                        or  ((first_start_edge[0] not in vert_and_its_neighbors)
                        and (first_start_edge[1]  in vert_and_its_neighbors)))
                    ):
                        break
                #exit(0)
                parallel_flood_fill_result = parallel_flood_fill_one_loop_end(
                    v_colors,
                    first_start_edge[0],
                    first_start_edge[1],
                    garment_vertex_edge_dict,
                    path,
                    vert_and_its_neighbors,
                )       
            vertices_side_1, visited_no_cross_edges = parallel_flood_fill_result 
            if equal(assembled_i_one_path[0],vert[0]):
                faces_side_2 = find_faces_from_intersect_edges(
                    visited_no_cross_edges, assembled_i_one_path[1:]
                )
            else:
                faces_side_2 = find_faces_from_intersect_edges(
                    visited_no_cross_edges, assembled_i_one_path[:-1]
                )               
            vertices_side_2 = []
            faces_side_1 = []
            v_and_f_in_intersection_path_one_batch.append(
                [vertices_side_1, faces_side_1, vertices_side_2, faces_side_2]
            )
            num_paths += 1         

        # a pair of closed paths
        # loop loop path """
