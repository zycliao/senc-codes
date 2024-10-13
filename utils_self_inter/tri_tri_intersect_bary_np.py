import numpy as np

class Ray:
    def __init__(self, origin=None, direction=None):
        self.origin = origin
        self.direction = direction

def min_max(a, b):
    if a < b:
        return a, b
    else:
        return b, a

def permutation(t, ids_p, ids):
    # the point is t0 * ids_p0 + ...
    new_t = []
    for i in range(3):
        for j in range(3):
            if ids[i] == ids_p[j]:
                new_t.append(t[j])
    return new_t



def compute_barycentric(intersec_point, a_, b_, c_):
    ba = a_ - b_
    cb = b_ - c_
    oa = a_ - intersec_point
    ob = b_ - intersec_point
    oc = c_ - intersec_point

    cross_ab = np.cross(oa, ob)
    cross_bc = np.cross(ob, oc)
    cross_ca = np.cross(oc, oa)

    whole_area = np.linalg.norm(np.cross(ba, cb))
    b_area = np.linalg.norm(cross_ca)
    a_area = np.linalg.norm(cross_bc)
    c_area = np.linalg.norm(cross_ab)

    alpha_a = a_area / whole_area
    alpha_b = b_area / whole_area
    alpha_c = c_area / whole_area

    return alpha_a, alpha_b, alpha_c


def ray_triangle_intersect_bary_np(r, a_, b_, c_, max_t):
    ba = a_ - b_
    cb = b_ - c_
    pn = np.cross(ba, cb)
    pn /= np.linalg.norm(pn)
    pc = np.dot(pn, a_)

    if (abs(np.dot(pn, r.direction))>0.001):
        t = (pc - np.dot(pn, r.origin)) / np.dot(pn, r.direction)
    else:
        return

    if t<=0 or t>=max_t:
        return
    else:
        intersec_point = r.origin + r.direction* t
        oa = a_ - intersec_point
        ob = b_ - intersec_point
        oc = c_-(intersec_point)

        cross_ab = np.cross(oa, ob)
        cross_bc = np.cross(ob, oc)
        cross_ca = np.cross(oc, oa)
        
        s1 = np.dot(cross_ab, cross_bc)
        s2 = np.dot(cross_bc, cross_ca)
        s3 = np.dot(cross_ca, cross_ab)

        if (s1>0 and s2>0 and s3>0):
            whole_area = np.linalg.norm(np.cross(ba, cb))
            b_area = np.linalg.norm(cross_ca)
            a_area = np.linalg.norm(cross_bc)
            c_area = np.linalg.norm(cross_ab)

            alpha_a = a_area / whole_area
            alpha_b = b_area / whole_area
            alpha_c = c_area / whole_area

            return alpha_a, alpha_b, alpha_c
        else:
            return

# This function will return two
def find_intersection_point_bary_np(u0, u1, u2, v0, v1, v2, ids_u, ids_v, tri_idx_u, tri_idx_v, bary_dict):  
    inter_num = 0
    intersection_points = []

    # u0->u1 intersect v
    u0_to_u1 = u1 - u0
    r0 = Ray(u0, u0_to_u1)
    t0 = ray_triangle_intersect_bary_np(r0, v0, v1, v2, np.linalg.norm(u0_to_u1))
    if t0:
        s, l = min_max(ids_u[0], ids_u[1])
        intersection_points.append((s, l, tri_idx_v))
        bary_dict[(s, l, tri_idx_v)] = (t0[0], t0[1], t0[2]) 
        inter_num += 1

    # u0->u2 intersect v
    u0_to_u2 = u2-(u0)
    r1 = Ray(u0, u0_to_u2 / np.linalg.norm(u0_to_u2))
    t1 = ray_triangle_intersect_bary_np(r1, v0, v1, v2, np.linalg.norm(u0_to_u2))
    if t1:
        s, l = min_max(ids_u[0],ids_u[2])
        intersection_points.append((s, l, tri_idx_v))
        bary_dict[(s, l, tri_idx_v)] = (t1[0], t1[1], t1[2])
        inter_num += 1

    # u1->u2 intersect v
    if inter_num < 2:
        u1_to_u2 = u2-(u1)
        r2 = Ray(u1, u1_to_u2 / np.linalg.norm(u1_to_u2))
        t2 = ray_triangle_intersect_bary_np(r2, v0, v1, v2, np.linalg.norm(u1_to_u2))
        if t2:
            s, l = min_max(ids_u[1], ids_u[2])
            intersection_points.append((s, l, tri_idx_v))
            bary_dict[(s, l, tri_idx_v)] = (t2[0], t2[1], t2[2])
            inter_num += 1

    # v's edges intersect with u
    if inter_num < 2:
        v0_to_v1 = v1-(v0)
        r3 = Ray(v0, v0_to_v1 / np.linalg.norm(v0_to_v1))
        t3 = ray_triangle_intersect_bary_np(r3, u0, u1, u2, np.linalg.norm(v0_to_v1))
        if t3:
            s, l = min_max(ids_v[0], ids_v[1])
            intersection_points.append((s, l, tri_idx_u))
            bary_dict[(s, l, tri_idx_u)] = (t3[0], t3[1], t3[2])
            inter_num += 1

    if inter_num < 2:
        v0_to_v2 = v2-(v0)
        r4 = Ray(v0, v0_to_v2 / np.linalg.norm(v0_to_v2))
        t4 = ray_triangle_intersect_bary_np(r4, u0, u1, u2, np.linalg.norm(v0_to_v2))
        if t4:
            s, l = min_max(ids_v[0], ids_v[2])
            intersection_points.append((s, l, tri_idx_u))
            bary_dict[(s, l, tri_idx_u)] = (t4[0], t4[1], t4[2])
            inter_num += 1

    if inter_num < 2:
        v1_to_v2 = v2-(v1)
        r5 = Ray(v1, v1_to_v2 / np.linalg.norm(v1_to_v2))
        t5 = ray_triangle_intersect_bary_np(r5, u0, u1, u2, np.linalg.norm(v1_to_v2))
        if t5:
            s, l = min_max(ids_v[1], ids_v[2])
            intersection_points.append((s, l, tri_idx_u))
            bary_dict[(s, l, tri_idx_u)] = (t5[0], t5[1], t5[2])
            inter_num += 1

    return intersection_points

def find_intersection_point_share_one_v_bary_np(u0, u1, shared_v_pos, v0, v1, ids_u_s, ids_v_s,
    tri_idx_u, tri_idx_v, bary_dict, ids_u, ids_v):
    inter_num = 0
    intersection_points = []

    # u0->u1 intersect v
    u0_to_u1 = u1-u0
    r0 = Ray(u0, u0_to_u1 / np.linalg.norm(u0_to_u1))
    t0 = ray_triangle_intersect_bary_np(r0, v0, v1, shared_v_pos, np.linalg.norm(u0_to_u1))
    if t0:
        s, l = min_max(ids_u_s[0], ids_u_s[1])
        intersection_points.append((s, l, tri_idx_v))
        perm_t = permutation(t0, ids_v_s, ids_v)
        bary_dict[(s, l, tri_idx_v)] = (perm_t[0], perm_t[1], perm_t[2])
        inter_num += 1

    # v's edges intersect with u
    if inter_num == 0:
        v0_to_v1 = v1-(v0)
        r3 = Ray(v0, v0_to_v1 / np.linalg.norm(v0_to_v1))
        t3 = ray_triangle_intersect_bary_np(r3, u0, u1, shared_v_pos, np.linalg.norm(v0_to_v1))
        if t3:
            s, l = min_max(ids_v_s[0], ids_v_s[1])
            intersection_points.append((s, l, tri_idx_u))
            perm_t = permutation(t3, ids_u_s, ids_u)
            bary_dict[(s, l, tri_idx_u)] = (perm_t[0], perm_t[1], perm_t[2])
            inter_num += 1

    return intersection_points