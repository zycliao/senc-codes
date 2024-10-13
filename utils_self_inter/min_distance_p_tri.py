from utils_self_inter.tri_tri_intersect import Vec3


def min_distance_p_tri(p, B, V0, V1):
    P = Vec3(p[0], p[1], p[2])
    B = Vec3(B[0], B[1], B[2])
    V0 = Vec3(V0[0], V0[1], V0[2])
    V1 = Vec3(V1[0], V1[1], V1[2])

    # T(s,t) = B + s*E0 + t*E1
    E0 = V0.sub(B)
    E1 = V1.sub(B)

    # Initialize some quantities
    D = B.sub(P)
    a = E0.dot(E0)
    b = E0.dot(E1)
    c = E1.dot(E1)
    d = E0.dot(D)
    e = E1.dot(D)
    f = D.dot(D)

    s = b * e - c * d
    t = b * d - a * e
    det = abs(a * c - b * b)

    if (s + t) <= det:
        if s < 0:
            if t < 0:
                # region 4
                if d < 0:
                    t = 0.0
                    if -d >= a:
                        s = 1.0
                    else:
                        s = -d / a
                else:
                    s = 0.0
                    if e >= 0:
                        t = 0.0
                    elif -e >= c:
                        t = 1.0
                    else:
                        t = -e / c
            else:
                # region 3
                s = 0.0
                if e >= 0:
                    t = 0.0
                elif -e >= c:
                    t = 1.0
                else:
                    t = -e / c
        elif t < 0:
            # region 5
            t = 0.0
            if d >= 0:
                s = 0.0
            elif -d >= a:
                s = 1.0
            else:
                s = -d / a
        else:
            # region 0
            s /= det
            t /= det
    else:
        if s < 0:
            # region 2
            tmp0 = b + d
            tmp1 = c + e
            if tmp1 > tmp0:
                numer = tmp1 - tmp0
                denom = a - 2.0 * b + c
                if numer >= denom:
                    s = 1.0
                else:
                    s = numer / denom
                t = 1.0 - s
            else:
                s = 0.0
                if tmp1 <= 0:
                    t = 1.0
                elif e >= 0:
                    t = 0.0
                else:
                    t = -e / c
        elif t < 0:
            # region 6
            tmp0 = b + e
            tmp1 = a + d
            if tmp1 > tmp0:
                numer = tmp1 - tmp0
                denom = a - 2.0 * b + c
                if numer >= denom:
                    t = 1.0
                else:
                    t = numer / denom
                s = 1.0 - t
            else:
                t = 0.0
                if tmp1 <= 0:
                    s = 1.0
                elif d >= 0:
                    s = 0.0
                else:
                    s = -d / a
        else:
            # region 1
            numer = (c + e) - (b + d)
            if numer <= 0:
                s = 0.0
            else:
                denom = a - 2.0 * b + c
                if numer >= denom:
                    s = 1.0
                else:
                    s = numer / denom
            t = 1.0 - s
    
    closest_p = B.add((E0.scaler_multiply(s)).add(E1.scaler_multiply(t)))
    distance = (closest_p.sub(P)).length()
    return [1.0 - s - t, s, t], distance
