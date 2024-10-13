import numpy as np

for i in range(vertices.shape[0]):
    vertices_np = vertices[i].numpy()
    #closed_v, closed_f = get_closed_garment(self.garment.boundary_paths, vertices_np, self.garment.faces)
    #vv, ff, inter_f_pairs, birth_tri, unique_vv = igl.copyleft.cgal.remesh_self_intersections(closed_v, closed_f)
    vv, ff, inter_f_pairs, birth_tri, unique_vv = igl.copyleft.cgal.remesh_self_intersections(vertices_np, self.garment.faces)
    for i in range(len(inter_f_pairs)):
        for j in range(len(inter_f_pairs)):
            for k in range(len(inter_f_pairs)):
                if i!=j and j!=k and i!=k:
                    fset = set()
                    fset.add(inter_f_pairs[i][0])
                    fset.add(inter_f_pairs[i][1])
                    fset.add(inter_f_pairs[j][0])
                    fset.add(inter_f_pairs[j][1])
                    fset.add(inter_f_pairs[k][0])
                    fset.add(inter_f_pairs[k][1])
                    if len(fset)==3:
                        print("OH MY GOD")