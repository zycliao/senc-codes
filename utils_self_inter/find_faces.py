def find_one_side_faces_loop(inside_v, outside_v, partition_path, loop_vertices, garment_v_to_faces, garment_faces):
    this_side_v = inside_v + partition_path
    faces = set()
    if len(inside_v)>0:
        for v in this_side_v:
            v_faces_list = garment_v_to_faces[v]
            for f in v_faces_list:
                v_in_f = garment_faces[f]
                if (
                    v_in_f[0] in this_side_v 
                    and v_in_f[1] in this_side_v
                    and v_in_f[2] in this_side_v
                ) and not (
                    (v_in_f[0] in outside_v)
                    or (v_in_f[1] in outside_v)
                    or (v_in_f[2] in outside_v)
                ):
                    faces.add(f)
    else:            
        for v in this_side_v:
            v_faces_list = garment_v_to_faces[v]
            for f in v_faces_list:
                v_in_f = garment_faces[f]
                # if at least two vertices of the face are in this side of vertices
                # add it to faces
                if (
                    (v_in_f[0] in this_side_v and v_in_f[1] in this_side_v)
                    or (v_in_f[0] in this_side_v and v_in_f[2] in this_side_v)
                    or (v_in_f[1] in this_side_v and v_in_f[2] in this_side_v)
                ) and not (
                    (v_in_f[0] in outside_v)
                    or (v_in_f[1] in outside_v)
                    or (v_in_f[2] in outside_v)
                ):
                    faces.add(f)
    return list(faces)


def find_one_side_faces(this_side_v, garment_v_to_faces, garment_faces):
    faces = set()
    if len(this_side_v)>2:
        for v in this_side_v:
            v_faces_list = garment_v_to_faces[v]
            for f in v_faces_list:
                v_in_f = garment_faces[f]
                # if at least two vertices of the face are in this side of vertices
                # add it to faces
                if (
                    (v_in_f[0] in this_side_v)
                    and (v_in_f[1] in this_side_v)
                    and (v_in_f[2] in this_side_v)
                ):
                    faces.add(f)

    elif len(this_side_v)>1:
        for v in this_side_v:
            v_faces_list = garment_v_to_faces[v]
            for f in v_faces_list:
                v_in_f = garment_faces[f]
                if (
                    (v_in_f[0] in this_side_v and v_in_f[1] in this_side_v)
                    or (v_in_f[0] in this_side_v and v_in_f[2] in this_side_v)
                    or (v_in_f[1] in this_side_v and v_in_f[2] in this_side_v)
                ):
                    faces.add(f)

    else:
        for v in this_side_v:
            v_faces_list = garment_v_to_faces[v]
            for f in v_faces_list:
                v_in_f = garment_faces[f]
                # if at least two vertices of the face are in this side of vertices
                # add it to faces
                if (
                    (v_in_f[0] in this_side_v)
                    or (v_in_f[1] in this_side_v)
                    or (v_in_f[2] in this_side_v)
                ):
                    faces.add(f)

    return list(faces)


def find_faces_from_intersect_edges(intersect_edges, assembled_i_one_path):
    faces = set()
    for intersect_edge in intersect_edges:
        for intersect_p in assembled_i_one_path:
            if (intersect_edge[0], intersect_edge[1]) == (intersect_p[0], intersect_p[1]):
                faces.add(intersect_p[2])
    return list(faces)