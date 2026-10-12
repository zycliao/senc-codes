// C++ implementation of the `cmake_example` module used by utils_self_inter/.
// Functionally equivalent to the pure-Python versions in
//   utils_self_inter/mesh.py::faces_to_edges_and_adjacency_in_progress_array
//   utils_self_inter/reassemble_path_newv.py::reassemble_path_newv
// Containers are returned in the same insertion order as the Python code.

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace py = pybind11;

static inline int64_t edge_key(int64_t a, int64_t b) {
    if (a > b) std::swap(a, b);
    return (a << 32) | (b & 0xffffffffLL);
}

py::tuple faces_to_edges_and_adjacency_in_progress_array(
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> faces,
    int64_t vertices_num) {
    if (faces.ndim() != 2) throw std::invalid_argument("faces must be a 2D array");
    const auto F = faces.shape(0);
    const auto K = faces.shape(1);
    auto fa = faces.unchecked<2>();

    std::vector<std::vector<int64_t>> vertex_edge(vertices_num);
    // edges in insertion order, each with the list of faces sharing it
    std::vector<std::pair<int64_t, int64_t>> edges;
    std::vector<std::vector<int64_t>> edge_faces;
    std::unordered_map<int64_t, size_t> edge_index;
    edge_index.reserve(static_cast<size_t>(F * K));

    for (py::ssize_t fidx = 0; fidx < F; ++fidx) {
        for (py::ssize_t i = 0; i < K; ++i) {
            int64_t v = fa(fidx, i);
            int64_t nv = fa(fidx, (i + 1) % K);
            if (v < 0 || nv < 0 || v >= vertices_num || nv >= vertices_num)
                throw std::out_of_range("face index out of range");
            int64_t key = edge_key(v, nv);
            auto it = edge_index.find(key);
            size_t e;
            if (it == edge_index.end()) {
                e = edges.size();
                edge_index.emplace(key, e);
                edges.emplace_back(std::min(v, nv), std::max(v, nv));
                edge_faces.emplace_back();
                vertex_edge[v].push_back(nv);
                vertex_edge[nv].push_back(v);
            } else {
                e = it->second;
            }
            edge_faces[e].push_back(fidx);
        }
    }

    std::vector<std::vector<int64_t>> face_adjacency(F);
    py::dict face_adjacency_to_edge;
    py::dict edge_to_face_adjacency;

    py::list edges_list;
    for (size_t e = 0; e < edges.size(); ++e) {
        py::tuple edge = py::make_tuple(edges[e].first, edges[e].second);
        edges_list.append(edge);
        const auto &fl = edge_faces[e];
        if (fl.size() < 2) continue;
        py::list pairs;
        for (size_t i = 0; i + 1 < fl.size(); ++i) {
            for (size_t j = i + 1; j < fl.size(); ++j) {
                py::tuple fp = py::make_tuple(fl[i], fl[j]);
                pairs.append(fp);
                face_adjacency[fl[i]].push_back(fl[j]);
                face_adjacency[fl[j]].push_back(fl[i]);
                face_adjacency_to_edge[fp] = edge;
            }
        }
        edge_to_face_adjacency[edge] = pairs;
    }

    return py::make_tuple(edges_list, py::cast(face_adjacency), face_adjacency_to_edge,
                          py::cast(vertex_edge), edge_to_face_adjacency);
}

// Same test as utils_self_inter/remesh_crsp.py::on_line
static inline bool on_line(const double *v0, const double *v1, const double *v2) {
    double a[3] = {v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]};
    double b[3] = {v2[0] - v1[0], v2[1] - v1[1], v2[2] - v1[2]};
    double cx = a[1] * b[2] - a[2] * b[1];
    double cy = a[2] * b[0] - a[0] * b[2];
    double cz = a[0] * b[1] - a[1] * b[0];
    double norm = std::sqrt(cx * cx + cy * cy + cz * cz);
    double dot = a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    return norm < 1e-16 && dot > 0;
}

std::vector<std::vector<int64_t>> reassemble_path_newv(
    const std::vector<std::vector<int64_t>> &assembled_v_one_batch,
    py::array_t<double, py::array::c_style | py::array::forcecast> vv,
    py::iterable vs_left_in,
    const std::vector<std::vector<int64_t>> &new_vertex_edge_dict,
    const std::vector<std::pair<int64_t, int64_t>> &new_edges) {
    if (vv.ndim() != 2 || vv.shape(1) != 3) throw std::invalid_argument("vv must be (N, 3)");
    const double *vp = vv.data();
    const int64_t N = vv.shape(0);
    auto pos = [&](int64_t i) -> const double * {
        if (i < 0 || i >= N) throw std::out_of_range("vertex index out of range");
        return vp + 3 * i;
    };

    // keep the iteration order of the Python set
    std::vector<int64_t> vs_left;
    for (auto h : vs_left_in) vs_left.push_back(h.cast<int64_t>());

    std::unordered_set<int64_t> edge_set;
    edge_set.reserve(new_edges.size() * 2);
    for (const auto &e : new_edges) edge_set.insert(edge_key(e.first, e.second));

    auto adjacent = [&](int64_t a, int64_t b) {
        if (a < 0 || a >= static_cast<int64_t>(new_vertex_edge_dict.size())) return false;
        for (int64_t n : new_vertex_edge_dict[a])
            if (n == b) return true;
        return false;
    };

    std::vector<std::vector<int64_t>> out;
    out.reserve(assembled_v_one_batch.size());
    for (const auto &path : assembled_v_one_batch) {
        std::vector<int64_t> new_path;
        if (path.empty()) throw std::invalid_argument("empty path");
        for (size_t i = 0; i + 1 < path.size(); ++i) {
            new_path.push_back(path[i]);
            if (edge_set.count(edge_key(path[i + 1], path[i]))) continue;
            std::vector<int64_t> cand;
            const double *a = pos(path[i + 1]);
            const double *c = pos(path[i]);
            for (int64_t v : vs_left)
                if (on_line(a, pos(v), c)) cand.push_back(v);
            while (!cand.empty()) {
                bool found = false;
                for (size_t j = 0; j < cand.size(); ++j) {
                    if (adjacent(new_path.back(), cand[j])) {
                        new_path.push_back(cand[j]);
                        cand.erase(cand.begin() + j);
                        found = true;
                        break;
                    }
                }
                // The Python version spins forever here; stop instead.
                if (!found) break;
            }
        }
        new_path.push_back(path.back());
        out.push_back(std::move(new_path));
    }
    return out;
}

PYBIND11_MODULE(cmake_example, m) {
    m.doc() = "Accelerated helpers for SENC self-intersection detection";
    m.def("faces_to_edges_and_adjacency_in_progress_array",
          &faces_to_edges_and_adjacency_in_progress_array, py::arg("faces"),
          py::arg("vertices_num"));
    m.def("reassemble_path_newv", &reassemble_path_newv, py::arg("assembled_v_one_batch"),
          py::arg("vv"), py::arg("vs_left"), py::arg("new_vertex_edge_dict"),
          py::arg("new_edges"));
}
