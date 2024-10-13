import igl
from model.cloth import Garment
from utils.mesh import faces_to_edges_and_adjacency, compute_boundary_paths
import numpy as np
def get_closed_garment(boundary_paths, vertices, faces):
    for num_path, boundary in enumerate(boundary_paths):
        new_v = np.mean(vertices[boundary], axis=0)
        vertices = np.vstack((vertices, new_v))
        v_idx = len(vertices) - 1
        for i, v in enumerate(boundary):
            if num_path != 3:
                nv = boundary[(i + 1) % len(boundary)]
                faces = np.vstack((faces, np.array([nv, v, v_idx])))
            else:
                nv = boundary[(i + 1) % len(boundary)]
                faces = np.vstack((faces, np.array([v, nv, v_idx])))             
    return vertices, faces
#garment = Garment("/userhome/cs/wsn1226/SSG/body_models/smpl_female_neutral/tshirt.obj")
# garment = Garment("/userhome/cs/wsn1226/SSG/body_models/mannequin/tshirt.obj")

# igl.write_triangle_mesh("/userhome/cs/wsn1226/SSG/body_models/mannequin/test_rewrite_tshirt.obj", garment.vertices, garment.faces)

# print(garment.closed_four_v_dict)
v, f = igl.read_triangle_mesh("/userhome/cs/wsn1226/SSG/body_models/smpl_female_neutral/dress_dodge_low_one_layer.obj")
(edges_np,
face_adjacency,
face_adjacency_edges,
vertex_edge_dict,
adjacency_edge_idx,
boundary_edges_dict,
face_adjacency_dict )= faces_to_edges_and_adjacency(f)
boundary_paths = compute_boundary_paths(boundary_edges_dict)
closed_v, closed_f = get_closed_garment(boundary_paths, v, f)
igl.write_triangle_mesh("/userhome/cs/wsn1226/SSG/body_models/smpl_female_neutral/oriented_closed_dress_dodge_low_one_layer.obj", closed_v, closed_f)


