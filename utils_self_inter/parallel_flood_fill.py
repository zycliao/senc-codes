from queue import Queue

def parallel_flood_fill_face(
    f_colors,
    f0,
    f1,
    face_adjencency_dict,
    face_adjacency_to_edge_dict,
    no_cross_edges,
    birth_tri,
    old_face_adjcency,
):
    # print(no_cross_edges)
    # print("f0:", f0)
    # print("f1:", f1)
    f_colors_backup = f_colors.copy()
    f_colored_1 = []
    f_colored_2 = []
    queue0 = Queue()
    queue1 = Queue()
    queue0.put(f0)
    queue1.put(f1)
    while True:
        if queue0.empty() and queue1.empty():
            break
        if not queue0.empty():
            next_f0 = queue0.get()
            if f_colors[next_f0] == 1:
                continue
            elif f_colors[next_f0] == 2:
                print("AAAAAAAAAAAAA")
                f_colors = f_colors_backup
                return False
            else:
                f_colors[next_f0] = 1
                f_colored_1.append(next_f0)
                old_adj_faces0 = old_face_adjcency[birth_tri[next_f0]]
                for neighbor_f0 in face_adjencency_dict[next_f0]:
                    edge = face_adjacency_to_edge_dict[
                        tuple(sorted([neighbor_f0, next_f0]))
                    ]
                    if (
                        (edge[0], edge[1]) not in no_cross_edges
                        and f_colors[neighbor_f0] != 1
                        and (
                            birth_tri[neighbor_f0] in old_adj_faces0
                            or birth_tri[next_f0] == birth_tri[neighbor_f0]
                        )
                    ):
                        queue0.put(neighbor_f0)
        else:
            if len(f_colored_1) <= len(f_colored_2):
                for face in f_colored_2:
                    f_colors[face] = 0
                return f_colored_1

        if not queue1.empty():
            next_f1 = queue1.get()
            if f_colors[next_f1] == 2:
                continue
            elif f_colors[next_f1] == 1:
                print("AAAAAAAAAAAAA")
                f_colors = f_colors_backup
                return False
            else:
                f_colors[next_f1] = 2
                f_colored_2.append(next_f1)
                old_adj_faces1 = old_face_adjcency[birth_tri[next_f1]]
                for neighbor_f1 in face_adjencency_dict[next_f1]:
                    edge = face_adjacency_to_edge_dict[
                        tuple(sorted([neighbor_f1, next_f1]))
                    ]
                    if (
                        (edge[0], edge[1]) not in no_cross_edges
                        and f_colors[neighbor_f1] != 2
                        and (
                            birth_tri[neighbor_f1] in old_adj_faces1
                            or birth_tri[next_f1] == birth_tri[neighbor_f1]
                        )
                    ):
                        queue1.put(neighbor_f1)
        else:
            if len(f_colored_2) <= len(f_colored_1):
                for face in f_colored_2:
                    f_colors[face] = 1
                for face in f_colored_1:
                    f_colors[face] = 0
                return f_colored_2

    if len(f_colored_1) <= len(f_colored_2):
        for face in f_colored_2:
            f_colors[face] = 0
        return f_colored_1
    else:
        for face in f_colored_2:
            f_colors[face] = 1
        for face in f_colored_1:
            f_colors[face] = 0
        return f_colored_2


def parallel_flood_fill(
    v_colors, v0, v1, vertex_edge_dict, no_cross_edges, no_reach_vertices
):
    visited_no_cross_edges_1 = set()
    visited_no_cross_edges_2 = set()
    v_colored_1 = []
    v_colored_2 = []
    queue0 = Queue()
    queue1 = Queue()
    queue0.put(v0)
    queue1.put(v1)
    while True:
        if queue0.empty() and queue1.empty():
            break
        if not queue0.empty():
            next_v0 = queue0.get()
            if v_colors[next_v0] == 1:
                continue
            else:
                v_colors[next_v0] = 1
                v_colored_1.append(next_v0)
                for neighbor_v0 in vertex_edge_dict[next_v0]:
                    edge = tuple(sorted([neighbor_v0, next_v0]))
                    if edge not in no_cross_edges:
                        if neighbor_v0 not in no_reach_vertices:
                            queue0.put(neighbor_v0)
                    else:
                        visited_no_cross_edges_1.add(edge)
        else:
            if len(v_colored_1) <= len(v_colored_2):
                return v_colored_1, visited_no_cross_edges_1

        if not queue1.empty():
            next_v1 = queue1.get()
            if v_colors[next_v1] == 2:
                continue
            else:
                v_colors[next_v1] = 2
                v_colored_2.append(next_v1)
                for neighbor_v1 in vertex_edge_dict[next_v1]:
                    edge = tuple(sorted([neighbor_v1, next_v1]))
                    if edge not in no_cross_edges:
                        if neighbor_v1 not in no_reach_vertices:
                            queue1.put(neighbor_v1)
                    else:
                        visited_no_cross_edges_2.add(edge)
        else:
            if len(v_colored_2) <= len(v_colored_1):
                return v_colored_2, visited_no_cross_edges_2

    if len(v_colored_1) <= len(v_colored_2):
        return v_colored_1, visited_no_cross_edges_1
    else:
        return v_colored_2, visited_no_cross_edges_2


def parallel_flood_fill_one_loop_end(
    v_colors, v0, v1, vertex_edge_dict, no_cross_edges, no_reach_vertices
):
    visited_no_cross_edges_1 = set()
    visited_no_cross_edges_2 = set()
    v_colored_1 = []
    v_colored_2 = []
    queue0 = Queue()
    queue1 = Queue()
    queue0.put(v0)
    queue1.put(v1)
    while True:
        if queue0.empty() and queue1.empty():
            break
        if not queue0.empty():
            next_v0 = queue0.get()
            if v_colors[next_v0] == 1:
                continue
            else:
                if v_colors[next_v0] == 2:
                    return visited_no_cross_edges_1.union(visited_no_cross_edges_2)
                v_colors[next_v0] = 1
                v_colored_1.append(next_v0)
                for neighbor_v0 in vertex_edge_dict[next_v0]:
                    edge = tuple(sorted([neighbor_v0, next_v0]))
                    if edge not in no_cross_edges:
                        if neighbor_v0 not in no_reach_vertices:
                            queue0.put(neighbor_v0)
                    else:
                        visited_no_cross_edges_1.add(edge)
        else:
            if len(v_colored_1) <= len(v_colored_2):
                return v_colored_1, visited_no_cross_edges_1

        if not queue1.empty():
            next_v1 = queue1.get()
            if v_colors[next_v1] == 2:
                continue
            else:
                if v_colors[next_v1] == 1:
                    return visited_no_cross_edges_1.union(visited_no_cross_edges_2)
                v_colors[next_v1] = 2
                v_colored_2.append(next_v1)
                for neighbor_v1 in vertex_edge_dict[next_v1]:
                    edge = tuple(sorted([neighbor_v1, next_v1]))
                    if edge not in no_cross_edges:
                        if neighbor_v1 not in no_reach_vertices:
                            queue1.put(neighbor_v1)
                    else:
                        visited_no_cross_edges_2.add(edge)
        else:
            if len(v_colored_2) <= len(v_colored_1):
                return v_colored_2, visited_no_cross_edges_2

    if len(v_colored_1) <= len(v_colored_2):
        return v_colored_1, visited_no_cross_edges_1
    else:
        return v_colored_2, visited_no_cross_edges_2


def flood_fill(graph, start_v, no_reach_vertices):
    vertices_1_side = []
    queue = Queue()
    queue.put(start_v)
    vertices_1_side.append(start_v)
    while not queue.empty():
        next_v = queue.get()
        for neighbor_v in graph[next_v]:
            if (neighbor_v not in no_reach_vertices) and (
                neighbor_v not in vertices_1_side
            ):
                queue.put(neighbor_v)
                vertices_1_side.append(neighbor_v)
    return vertices_1_side


def flood_fill_f_whole(face_adjacency_dict, start_f, no_cross_edges):
    while graph:
        vertices_1_side = []
        queue = Queue()
        queue.put(start_v)
        vertices_1_side.append(start_v)
        while not queue.empty():
            next_v = queue.get()
            for neighbor_v in graph[next_v]:
                if (neighbor_v not in no_reach_vertices) and (
                    neighbor_v not in vertices_1_side
                ):
                    queue.put(neighbor_v)
                    vertices_1_side.append(neighbor_v)
        return vertices_1_side
