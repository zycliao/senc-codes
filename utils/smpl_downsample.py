import networkx as nx
import scipy.sparse
import torch
import numpy as np
import trimesh
import itertools

def scipy_to_pytorch(A, U, D):
    """Convert scipy sparse matrices to pytorch sparse matrix."""
    ptU = []
    ptD = []

    for i in range(len(U)):
        u = scipy.sparse.coo_matrix(U[i])
        i = torch.LongTensor(np.array([u.row, u.col]))
        v = torch.FloatTensor(u.data)
        ptU.append(torch.sparse.FloatTensor(i, v, u.shape))

    for i in range(len(D)):
        d = scipy.sparse.coo_matrix(D[i])
        i = torch.LongTensor(np.array([d.row, d.col]))
        v = torch.FloatTensor(d.data)
        ptD.append(torch.sparse.FloatTensor(i, v, d.shape))

    return ptU, ptD


def adjmat_sparse(adjmat, nsize=1):
    """Create row-normalized sparse graph adjacency matrix."""
    adjmat = scipy.sparse.csr_matrix(adjmat)
    if nsize > 1:
        orig_adjmat = adjmat.copy()
        for _ in range(1, nsize):
            adjmat = adjmat * orig_adjmat
    adjmat.data = np.ones_like(adjmat.data)
    for i in range(adjmat.shape[0]):
        adjmat[i, i] = 1
    num_neighbors = np.array(1 / adjmat.sum(axis=-1))
    adjmat = adjmat.multiply(num_neighbors)
    adjmat = scipy.sparse.coo_matrix(adjmat)
    row = adjmat.row
    col = adjmat.col
    data = adjmat.data
    i = torch.LongTensor(np.array([row, col]))
    v = torch.from_numpy(data).float()
    adjmat = torch.sparse.FloatTensor(i, v, adjmat.shape)
    return adjmat


def get_graph_params(filename, nsize=1):
    """Load and process graph adjacency matrix and upsampling/downsampling matrices."""
    data = np.load(filename, encoding='latin1', allow_pickle=True)
    A = data['A']
    U = data['U']
    D = data['D']
    U, D = scipy_to_pytorch(A, U, D)
    A = [adjmat_sparse(a, nsize=nsize) for a in A]
    return A, U, D


class SparseMM(torch.autograd.Function):
    """Redefine sparse @ dense matrix multiplication to enable backpropagation.
    The builtin matrix multiplication operation does not support backpropagation in some cases.
    """
    @staticmethod
    def forward(ctx, sparse, dense):
        ctx.req_grad = dense.requires_grad
        ctx.save_for_backward(sparse)
        return torch.matmul(sparse, dense)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = None
        sparse, = ctx.saved_tensors
        if ctx.req_grad:
            grad_input = torch.matmul(sparse.t(), grad_output)
        return None, grad_input

def spmm(sparse, dense):
    return SparseMM.apply(sparse, dense)

def edge_to_mesh(edges):
    num_verts = edges.max() + 1
    vert_to_edge = {i: [] for i in range(num_verts)}
    for i, (v1, v2) in enumerate(edges):
        vert_to_edge[v1].append(i)
        vert_to_edge[v2].append(i)

    faces = []
    for i, (v1, v2) in enumerate(edges):
        # 找到和v1和v2共享顶点的所有边
        for e1, e2 in itertools.product(vert_to_edge[v1], vert_to_edge[v2]):
            if e1 == e2:
                continue
            common_vertex = set(edges[e1]) & set(edges[e2])
            if len(common_vertex) == 0:
                continue
            v3 = list(common_vertex)[0]
            if v3 == v1 or v3 == v2:
                continue
            face = tuple(sorted([v1, v2, v3]))
            faces.append(face)

    # 删除重复的面
    faces = list(set(faces))
    return np.array(faces)



def unify_mesh_normals_old(mesh):
    # 创建一个空的数组来存储每个面的法线方向
    normals = np.zeros_like(mesh.face_normals)

    # 计算第一个面的法线
    normals[0] = mesh.face_normals[0]

    # 获取每个面的邻居
    neighbors = mesh.face_adjacency
    face_adjacency = [[] for k in range(len(mesh.faces))]
    for i, (f1, f2) in enumerate(neighbors):
        face_adjacency[f1].append(f2)
        face_adjacency[f2].append(f1)

    # BFS
    queue = [0]
    visited = set()
    faces = np.array(mesh.faces)
    flip_ids = []
    while queue:
        face_id = queue.pop(0)
        visited.add(face_id)
        cur_face = list(faces[face_id])
        for neighbor_id in face_adjacency[face_id]:
            if neighbor_id in visited:
                continue
            neighbor_face = list(faces[neighbor_id])
            # find the common edge
            common_edge = set(cur_face) & set(neighbor_face)
            assert len(common_edge) == 2
            # check if the common edge is in the same direction in both faces
            common_edge = list(common_edge)
            e1, e2 = common_edge
            ce1, ce2 = cur_face.index(e1), cur_face.index(e2)
            ne1, ne2 = neighbor_face.index(e1), neighbor_face.index(e2)
            if (ce1, ce2) in [(0, 1), (1, 2), (2, 0)]:
                c_ordered = True
            else:
                c_ordered = False
            if (ne1, ne2) in [(0, 1), (1, 2), (2, 0)]:
                n_ordered = True
            else:
                n_ordered = False
            # if not, flip the neighbor face
            if c_ordered != n_ordered:
                faces[neighbor_id] = faces[neighbor_id][::-1]
                flip_ids.append(neighbor_id)
            visited.add(neighbor_id)
            queue.append(neighbor_id)

    queue = [0]
    visited = set()
    flip_ids = []
    while queue:
        face_id = queue.pop(0)
        visited.add(face_id)
        cur_face = list(faces[face_id])
        for neighbor_id in face_adjacency[face_id]:
            if neighbor_id in visited:
                continue
            neighbor_face = list(faces[neighbor_id])
            # find the common edge
            common_edge = set(cur_face) & set(neighbor_face)
            assert len(common_edge) == 2
            # check if the common edge is in the same direction in both faces
            common_edge = list(common_edge)
            e1, e2 = common_edge
            ce1, ce2 = cur_face.index(e1), cur_face.index(e2)
            ne1, ne2 = neighbor_face.index(e1), neighbor_face.index(e2)
            if (ce1, ce2) in [(0, 1), (1, 2), (2, 0)]:
                c_ordered = True
            else:
                c_ordered = False
            if (ne1, ne2) in [(0, 1), (1, 2), (2, 0)]:
                n_ordered = True
            else:
                n_ordered = False
            # if not, flip the neighbor face
            if c_ordered != n_ordered:
                faces[neighbor_id] = faces[neighbor_id][::-1]
                flip_ids.append(neighbor_id)
            visited.add(neighbor_id)
            queue.append(neighbor_id)


    return faces


def unify_mesh_normals(mesh, downsample_matrix, template_normal):
    # downsample_matrix: (V_low, V_high)
    indices = np.argmax(downsample_matrix, axis=1)
    faces = np.array(mesh.faces)
    for i, face in enumerate(faces):
        face_normal = mesh.face_normals[i]
        v1, v2, v3 = face
        n1 = template_normal[indices[v1]]
        n2 = template_normal[indices[v2]]
        n3 = template_normal[indices[v3]]
        v = np.dot(face_normal, n1) + np.dot(face_normal, n2) + np.dot(face_normal, n3)
        if v < 0:
            faces[i] = faces[i][::-1]
    return faces




class MeshCMR(object):
    """Mesh object that is used for handling certain graph operations."""
    def __init__(self, filename="/root/data/neural_cloth/mesh_downsampling.npz",
                 num_downsampling=2, nsize=1, device=torch.device('cuda')):
        self._A, self._U, self._D = get_graph_params(filename=filename, nsize=nsize)
        self._A = [a.to(device) for a in self._A]
        self._U = [u.to(device) for u in self._U]
        self._D = [d.to(device) for d in self._D]

        template_mesh = trimesh.load_mesh("/root/data/neural_cloth/human_motion/stretch.obj")
        template_verts = template_mesh.vertices
        template_verts_torch = torch.from_numpy(template_verts).float().to(device)
        template_normals = template_mesh.vertex_normals

        self.faces = []
        for i, _A in enumerate(self._A):
            G = nx.from_numpy_matrix(_A.to_dense().cpu().numpy())

            faces = edge_to_mesh(np.array(G.edges))
            if i != 0:
                verts = self.downsample(template_verts_torch[None], n2=i)[0]
                mesh = trimesh.Trimesh(vertices=verts.cpu().numpy(), faces=faces)
                faces = unify_mesh_normals(mesh, self._D[i-1].to_dense().cpu().numpy(), template_normals)
                new_mesh = trimesh.Trimesh(vertices=verts.cpu().numpy(), faces=faces)
                template_normals = np.array(new_mesh.vertex_normals)
                self.faces.append(faces)
            else:
                self.faces.append(template_mesh.faces)

        self.num_downsampling = num_downsampling

        # # load template vertices from SMPL and normalize them
        # smpl = SMPL()
        # ref_vertices = smpl.v_template
        # center = 0.5*(ref_vertices.max(dim=0)[0] + ref_vertices.min(dim=0)[0])[None]
        # ref_vertices -= center
        # ref_vertices /= ref_vertices.abs().max().item()
        #
        # self._ref_vertices = ref_vertices.to(device)
        # self.faces = smpl.faces.int().to(device)

    @property
    def adjmat(self):
        """Return the graph adjacency matrix at the specified subsampling level."""
        return self._A[self.num_downsampling].float()

    @property
    def ref_vertices(self):
        """Return the template vertices at the specified subsampling level."""
        ref_vertices = self._ref_vertices
        for i in range(self.num_downsampling):
            ref_vertices = torch.spmm(self._D[i], ref_vertices)
        return ref_vertices

    def downsample(self, x, n1=0, n2=None):
        """Downsample mesh."""
        if n2 is None:
            n2 = self.num_downsampling
        if x.ndimension() < 3:
            for i in range(n1, n2):
                x = spmm(self._D[i], x)
        elif x.ndimension() == 3:
            out = []
            for i in range(x.shape[0]):
                y = x[i]
                for j in range(n1, n2):
                    y = spmm(self._D[j], y)
                out.append(y)
            x = torch.stack(out, dim=0)
        return x

    def upsample(self, x, n1=1, n2=0):
        """Upsample mesh."""
        if x.ndimension() < 3:
            for i in reversed(range(n2, n1)):
                x = spmm(self._U[i], x)
        elif x.ndimension() == 3:
            out = []
            for i in range(x.shape[0]):
                y = x[i]
                for j in reversed(range(n2, n1)):
                    y = spmm(self._U[j], y)
                out.append(y)
            x = torch.stack(out, dim=0)
        return x