import math
import random
from timeit import default_timer as timer


class Vec3:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def sub(self, v):
        return Vec3(self.x - v.x, self.y - v.y, self.z - v.z)

    def add(self, v):
        return Vec3(self.x + v.x, self.y + v.y, self.z + v.z)

    def dot(self, v):
        return self.x * v.x + self.y * v.y + self.z * v.z

    def cross(self, v):
        return Vec3(
            self.y * v.z - self.z * v.y,
            self.z * v.x - self.x * v.z,
            self.x * v.y - self.y * v.x,
        )

    def length(self):
        return math.sqrt(self.x * self.x + self.y * self.y + self.z * self.z)

    def normalize(self):
        l = self.length()
        return Vec3(self.x / l, self.y / l, self.z / l)

    def scaler_multiply(self, c):
        return Vec3(c * self.x, c * self.y, c * self.z)
    
    def print_vec(self):
        print(self.x, self.y, self.z)


class Ray:
    def __init__(self, origin=None, direction=None):
        self.origin = origin
        self.direction = direction


def ray_triangle_intersect(r, a_, b_, c_, max_t):
    ba = a_.sub(b_)
    cb = b_.sub(c_)
    pn = ba.cross(cb).normalize()
    pc = pn.dot(a_)

    if (abs(pn.dot(r.direction))>0.001):
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

            return 1
        else:
            return



# This function will return two
def find_intersection_point(u0, u1, u2, v0, v1, v2, indices_u, indices_v, tri_idx_u, tri_idx_v):  
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
    t0 = ray_triangle_intersect(r0, V0, V1, V2, U0_to_U1.length())
    if t0:
        if indices_u[0]<indices_u[1]:
            intersection_points.append((indices_u[0],indices_u[1],tri_idx_v))
        else:
            intersection_points.append((indices_u[1],indices_u[0],tri_idx_v))   
        inter_num += 1

    # u0->u2 intersect v
    U0_to_U2 = U2.sub(U0)
    r1 = Ray(U0, U0_to_U2.normalize())
    t1 = ray_triangle_intersect(r1, V0, V1, V2, U0_to_U2.length())
    if t1:
        if indices_u[0]<indices_u[2]:
            intersection_points.append((indices_u[0],indices_u[2],tri_idx_v))
        else:
            intersection_points.append((indices_u[2],indices_u[0],tri_idx_v))
        inter_num += 1

    # u1->u2 intersect v
    if inter_num < 2:
        U1_to_U2 = U2.sub(U1)
        r2 = Ray(U1, U1_to_U2.normalize())
        t2 = ray_triangle_intersect(r2, V0, V1, V2, U1_to_U2.length())
        if t2:
            if indices_u[1]<indices_u[2]:
                intersection_points.append((indices_u[1],indices_u[2],tri_idx_v))
            else:
                intersection_points.append((indices_u[2],indices_u[1],tri_idx_v))
            inter_num += 1

    # v's edges intersect with u
    if inter_num < 2:
        V0_to_V1 = V1.sub(V0)
        r3 = Ray(V0, V0_to_V1.normalize())
        t3 = ray_triangle_intersect(r3, U0, U1, U2, V0_to_V1.length())
        if t3:
            if indices_v[0]<indices_v[1]:
                intersection_points.append((indices_v[0],indices_v[1],tri_idx_u))
            else:
                intersection_points.append((indices_v[1],indices_v[0],tri_idx_u))
            inter_num += 1

    if inter_num < 2:
        V0_to_V2 = V2.sub(V0)
        r4 = Ray(V0, V0_to_V2.normalize())
        t4 = ray_triangle_intersect(r4, U0, U1, U2, V0_to_V2.length())
        if t4:
            if indices_v[0]<indices_v[2]:
                intersection_points.append((indices_v[0],indices_v[2],tri_idx_u))
            else:
                intersection_points.append((indices_v[2],indices_v[0],tri_idx_u))
            inter_num += 1

    if inter_num < 2:
        V1_to_V2 = V2.sub(V1)
        r5 = Ray(V1, V1_to_V2.normalize())
        t5 = ray_triangle_intersect(r5, U0, U1, U2, V1_to_V2.length())
        if t5:
            if indices_v[1]<indices_v[2]:
                intersection_points.append((indices_v[1],indices_v[2],tri_idx_u))
            else:
                intersection_points.append((indices_v[2],indices_v[1],tri_idx_u))
            inter_num += 1

    return intersection_points

def find_intersection_point_share_one_v(u0, u1, shared_v_pos, v0, v1, indices_u, indices_v, shared_v_idx, tri_idx_u, tri_idx_v):  
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
    t0 = ray_triangle_intersect(r0, V0, V1, V, U0_to_U1.length())
    if t0:
        if indices_u[0]<indices_u[1]:
            intersection_points.append((indices_u[0],indices_u[1], tri_idx_v))
        else:
            intersection_points.append((indices_u[1],indices_u[0], tri_idx_v))   
        inter_num += 1

    # v's edges intersect with u
    if inter_num == 0:
        V0_to_V1 = V1.sub(V0)
        r3 = Ray(V0, V0_to_V1.normalize())
        t3 = ray_triangle_intersect(r3, U0, U1, V, V0_to_V1.length())
        if t3:
            if indices_v[0]<indices_v[1]:
                intersection_points.append((indices_v[0],indices_v[1], tri_idx_u))
            else:
                intersection_points.append((indices_v[1],indices_v[0], tri_idx_u))
            inter_num += 1

    return intersection_points



""" def compute_barycentric(intersec_point, a_, b_, c_):
    ba = a_.sub(b_)
    cb = b_.sub(c_)
    oa = a_.sub(intersec_point)
    ob = b_.sub(intersec_point)
    oc = c_.sub(intersec_point)

    cross_ab = oa.cross(ob)
    cross_bc = ob.cross(oc)
    cross_ca = oc.cross(oa)

    whole_area = (ba.cross(cb)).length()
    b_area = (oa.cross(oc)).length()
    a_area = (ob.cross(oc)).length()
    c_area = (oa.cross(ob)).length()

    alpha_a = a_area / whole_area
    alpha_b = b_area / whole_area
    alpha_c = c_area / whole_area

    return alpha_a, alpha_b, alpha_c """