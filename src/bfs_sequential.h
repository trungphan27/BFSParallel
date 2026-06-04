#ifndef BFS_SEQUENTIAL_H
#define BFS_SEQUENTIAL_H

#include "graph.h"

/*
 * BFS tuần tự (top-down, queue-based).
 * Dùng làm baseline để tính speedup và verify kết quả.
 *
 * Tham số:
 *   g      : đồ thị CSR
 *   source : đỉnh nguồn
 *   dist   : mảng output, kích thước g->num_vertices
 *            dist[v] = khoảng cách BFS từ source, -1 nếu không thăm được
 *
 * Trả về: số cạnh đã duyệt (để tính MTEPS)
 */
edge_t bfs_sequential(const Graph *g, vertex_t source, int *dist);

#endif /* BFS_SEQUENTIAL_H */