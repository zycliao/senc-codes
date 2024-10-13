def find_shared_vertex(ids_u, ids_v):
    ids_u_s, ids_v_s = [], []
    for i in range(3):
        if ids_u[i] in ids_v:
            shared_v = ids_u[i]
            break
    for i in range(3):
        if ids_u[i] != shared_v:
            ids_u_s.append(ids_u[i])
        if ids_v[i] != shared_v:
            ids_v_s.append(ids_v[i])
    ids_u_s.append(shared_v)
    ids_v_s.append(shared_v)
    return ids_u_s, ids_v_s, shared_v