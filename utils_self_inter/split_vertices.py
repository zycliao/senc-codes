from utils_self_inter.weighted_shortest_path import *
from utils_self_inter.parallel_flood_fill import flood_fill

def split_vertices(vertices_to_split, start, end, garment_vertices, garment_vertex_edge_dict):
    # First find the shortest path from start to end
    #print("vertices to split:", vertices_to_split)
    #print("start:", start)
    #print("end:", end)
    vertices_to_split.append(start)
    vertices_to_split.append(end)
    inner_graph = dict()
    for v in vertices_to_split:
        for neighbor_v in garment_vertex_edge_dict[v]:
            if neighbor_v in vertices_to_split:
                inner_graph.setdefault(v, []).append(neighbor_v)
    edges = []
    graph = Graph()
    for v in inner_graph:
        for neighbor_v in inner_graph[v]:
            edges.append([neighbor_v, v, compute_distance_two_vertices(garment_vertices[neighbor_v], garment_vertices[v])])
    for edge in edges:
        graph.add_edge(*edge)
    partition_path = dijsktra(graph, start, end)
    #print("partition path:", partition_path)
    #print("partition path:", partition_path)
    # Then split the vertices, by starting from any vertices in vertices_to_split
    if len(vertices_to_split)>3:
        for v in vertices_to_split:
            if v not in partition_path:
                start_v = v
        vertices_side_1 = flood_fill(inner_graph, start_v, partition_path)
        #print("vertices_side_1:", vertices_side_1)
        vertices_side_2 = []
        for v in vertices_to_split:
            if (not (v in vertices_side_1)) and (not (v in partition_path)):
                vertices_side_2.append(v)
        #print("split vertices:", len(vertices_side_1), len(vertices_side_2))
        return vertices_side_1, vertices_side_2, partition_path
    elif len(vertices_to_split) == 2:
        return [vertices_to_split[0]], [], partition_path
    else:
        return [], [], partition_path