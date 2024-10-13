import math
import random
from timeit import default_timer as timer
import numpy as np
from utils_self_inter.tri_tri_intersect import Vec3, Ray

def min_max(a, b):
    if a < b:
        return a, b
    else:
        return b, a


def ray_triangle_intersect_bary(r, a_, b_, c_, max_t):
    ba = a_.sub(b_)
    cb = b_.sub(c_)
    pn = ba.cross(cb).normalize()
    pc = pn.dot(a_)

    if (abs(pn.dot(r.direction))>1e-15):
        t = (pc - pn.dot(r.origin)) / pn.dot(r.direction)
    else:
        return

    if t<=0 or t>=max_t:
        return
    else:
        intersec_point = (r.origin).add((r.direction).scaler_multiply(t))
        oa = a_.sub(intersec_point)
        ob = b_.sub(intersec_point)
        oc = c_.sub(intersec_point)

        cross_ab = oa.cross(ob)
        cross_bc = ob.cross(oc)
        cross_ca = oc.cross(oa)
        
        s1 = cross_ab.dot(cross_bc)
        s2 = cross_bc.dot(cross_ca)
        s3 = cross_ca.dot(cross_ab)

        if (s1>0 and s2>0 and s3>0):
            whole_area = (ba.cross(cb)).length()
            b_area = cross_ca.length()
            a_area = cross_bc.length()
            c_area = cross_ab.length()

            alpha_a = a_area / whole_area
            alpha_b = b_area / whole_area
            alpha_c = c_area / whole_area

            return alpha_a, alpha_b, alpha_c
        else:
            return



# This function will return two
def find_intersection_point_bary(u0, u1, u2, v0, v1, v2, ids_u, ids_v, tri_idx_u, tri_idx_v, bary_dict):  
    inter_num = 0
    intersection_points = []
    U0 = Vec3(u0[0], u0[1], u0[2])
    U1 = Vec3(u1[0], u1[1], u1[2])
    U2 = Vec3(u2[0], u2[1], u2[2])

    V0 = Vec3(v0[0], v0[1], v0[2])
    V1 = Vec3(v1[0], v1[1], v1[2])
    V2 = Vec3(v2[0], v2[1], v2[2])

    # u0->u1 intersect v
    U0_to_U1 = U1.sub(U0)
    r0 = Ray(U0, U0_to_U1.normalize())
    t0 = ray_triangle_intersect_bary(r0, V0, V1, V2, U0_to_U1.length())
    if t0:
        s, l = min_max(ids_u[0], ids_u[1])
        intersection_points.append((s, l, tri_idx_v))
        bary_dict[(s, l, tri_idx_v)] = t0 
        inter_num += 1

    # u0->u2 intersect v
    U0_to_U2 = U2.sub(U0)
    r1 = Ray(U0, U0_to_U2.normalize())
    t1 = ray_triangle_intersect_bary(r1, V0, V1, V2, U0_to_U2.length())
    if t1:
        s, l = min_max(ids_u[0],ids_u[2])
        intersection_points.append((s, l, tri_idx_v))
        bary_dict[(s, l, tri_idx_v)] = t1
        inter_num += 1

    # u1->u2 intersect v
    if inter_num < 2:
        U1_to_U2 = U2.sub(U1)
        r2 = Ray(U1, U1_to_U2.normalize())
        t2 = ray_triangle_intersect_bary(r2, V0, V1, V2, U1_to_U2.length())
        if t2:
            s, l = min_max(ids_u[1], ids_u[2])
            intersection_points.append((s, l, tri_idx_v))
            bary_dict[(s, l, tri_idx_v)] = t2
            inter_num += 1

    # v's edges intersect with u
    if inter_num < 2:
        V0_to_V1 = V1.sub(V0)
        r3 = Ray(V0, V0_to_V1.normalize())
        t3 = ray_triangle_intersect_bary(r3, U0, U1, U2, V0_to_V1.length())
        if t3:
            s, l = min_max(ids_v[0], ids_v[1])
            intersection_points.append((s, l, tri_idx_u))
            bary_dict[(s, l, tri_idx_u)] = t3
            inter_num += 1

    if inter_num < 2:
        V0_to_V2 = V2.sub(V0)
        r4 = Ray(V0, V0_to_V2.normalize())
        t4 = ray_triangle_intersect_bary(r4, U0, U1, U2, V0_to_V2.length())
        if t4:
            s, l = min_max(ids_v[0], ids_v[2])
            intersection_points.append((s, l, tri_idx_u))
            bary_dict[(s, l, tri_idx_u)] = t4
            inter_num += 1

    if inter_num < 2:
        V1_to_V2 = V2.sub(V1)
        r5 = Ray(V1, V1_to_V2.normalize())
        t5 = ray_triangle_intersect_bary(r5, U0, U1, U2, V1_to_V2.length())
        if t5:
            s, l = min_max(ids_v[1], ids_v[2])
            intersection_points.append((s, l, tri_idx_u))
            bary_dict[(s, l, tri_idx_u)] = t5
            inter_num += 1

    return intersection_points

def find_intersection_point_share_one_v_bary(u0, u1, shared_v_pos, v0, v1, ids_u_s, ids_v_s,
    tri_idx_u, tri_idx_v, bary_dict, ids_u, ids_v):
    inter_num = 0
    intersection_points = []
    U0 = Vec3(u0[0], u0[1], u0[2])
    U1 = Vec3(u1[0], u1[1], u1[2])

    V = Vec3(shared_v_pos[0], shared_v_pos[1], shared_v_pos[2])

    V0 = Vec3(v0[0], v0[1], v0[2])
    V1 = Vec3(v1[0], v1[1], v1[2])


    # u0->u1 intersect v
    U0_to_U1 = U1.sub(U0)
    r0 = Ray(U0, U0_to_U1.normalize())
    t0 = ray_triangle_intersect_bary(r0, V0, V1, V, U0_to_U1.length())
    if t0:
        s, l = min_max(ids_u_s[0], ids_u_s[1])
        intersection_points.append((s, l, tri_idx_v))
        perm_t = permutation(t0, ids_v_s, ids_v)
        bary_dict[(s, l, tri_idx_v)] = (perm_t[0], perm_t[1], perm_t[2])
        inter_num += 1

    # v's edges intersect with u
    if inter_num == 0:
        V0_to_V1 = V1.sub(V0)
        r3 = Ray(V0, V0_to_V1.normalize())
        t3 = ray_triangle_intersect_bary(r3, U0, U1, V, V0_to_V1.length())
        if t3:
            s, l = min_max(ids_v_s[0], ids_v_s[1])
            intersection_points.append((s, l, tri_idx_u))
            perm_t = permutation(t3, ids_u_s, ids_u)
            bary_dict[(s, l, tri_idx_u)] = (perm_t[0], perm_t[1], perm_t[2])
            inter_num += 1

    return intersection_points

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
