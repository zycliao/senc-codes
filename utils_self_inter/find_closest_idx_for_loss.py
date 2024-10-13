from utils_self_inter.min_distance_p_tri import min_distance_p_tri
# import tensorflow as tf

def find_closest_idx_for_loss(v_and_f_in_intersection_path_all_batches, vertices, faces):

    vf_pairs_all_batches = []
    barycentric_all_batches = []
    for batch_i in range(len(v_and_f_in_intersection_path_all_batches)):
        vf_pairs_one_batch = []
        barycentric_one_batch = []
        for (
            v_and_f_in_intersection_path_one_path
        ) in v_and_f_in_intersection_path_all_batches[batch_i]:
            v1 = v_and_f_in_intersection_path_one_path[0]
            f1 = v_and_f_in_intersection_path_one_path[1]
            v2 = v_and_f_in_intersection_path_one_path[2]
            f2 = v_and_f_in_intersection_path_one_path[3]
            print(v_and_f_in_intersection_path_one_path)
            for v in v1:
                closest_d = float('inf')
                for f in f2:
                    tri_0, tri_1, tri_2 = faces[f]
                    barycentric, distance = min_distance_p_tri(
                            vertices[batch_i][v],
                            vertices[batch_i][tri_0],
                            vertices[batch_i][tri_1],
                            vertices[batch_i][tri_2]
                        )
                    if closest_d > distance:
                        closest_f = f
                        closest_barycentric = barycentric
                        closest_d  = distance
                vf_pairs_one_batch.append([v, closest_f])
                barycentric_one_batch.append(closest_barycentric)
            for v in v2:
                closest_d = float('inf')
                for f in f1:
                    tri_0, tri_1, tri_2 = faces[f]
                    barycentric, distance = min_distance_p_tri(
                            vertices[batch_i][v],
                            vertices[batch_i][tri_0],
                            vertices[batch_i][tri_1],
                            vertices[batch_i][tri_2]
                        )
                    if closest_d > distance:
                        closest_f = f
                        closest_barycentric = barycentric
                        closest_d = distance
                vf_pairs_one_batch.append([v, closest_f])
                barycentric_one_batch.append(closest_barycentric)

        vf_pairs_all_batches.append(vf_pairs_one_batch)
        barycentric_all_batches.append(barycentric_one_batch)
    #print(len(vf_pairs_all_batches[0]))
    #print(len(barycentric_all_batches[0]))

    #print(len(vf_pairs_all_batches[1]))
    #print(len(barycentric_all_batches[1]))
    # return tf.ragged.constant(vf_pairs_all_batches,ragged_rank=1), tf.ragged.constant(barycentric_all_batches,ragged_rank=1)
    