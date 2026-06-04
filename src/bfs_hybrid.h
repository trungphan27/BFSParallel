#ifndef BFS_HYBRID_H
#define BFS_HYBRID_H

#include "graph.h"
#include <mpi.h>

/* ------------------------------------------------------------------ *
 * Tham số direction-optimizing (Beamer et al. 2012)                    *
 *                                                                      *
 * Chuyển sang bottom-up khi:                                           *
 *   frontier_edges > unvisited_edges / ALPHA                           *
 *   (frontier quá lớn → top-down lãng phí)                            *
 *                                                                      *
 * Quay lại top-down khi:                                               *
 *   frontier_size < num_vertices / BETA                                *
 *   (frontier đã nhỏ trở lại)                                          *
 * ------------------------------------------------------------------ */
#define BFS_ALPHA  14
#define BFS_BETA   24

/* ------------------------------------------------------------------ *
 * Phân vùng đỉnh giữa các MPI rank                                     *
 *                                                                      *
 * Rank r quản lý các đỉnh: [part_start(r), part_end(r))               *
 * ------------------------------------------------------------------ */
typedef struct {
    vertex_t total_vertices;  /* |V| toàn cục                          */
    int      num_ranks;       /* số MPI process                        */
    int      my_rank;         /* rank hiện tại                         */
    vertex_t local_start;     /* đỉnh đầu tiên của rank này            */
    vertex_t local_end;       /* đỉnh cuối + 1                         */
    vertex_t local_count;     /* = local_end - local_start             */
} Partition;

/* Tính phân vùng đều cho rank r trong num_ranks */
static inline vertex_t partition_start(vertex_t n, int num_ranks, int rank)
{
    return (vertex_t)((long long)n * rank / num_ranks);
}

static inline vertex_t partition_end(vertex_t n, int num_ranks, int rank)
{
    return (vertex_t)((long long)n * (rank + 1) / num_ranks);
}

/* ------------------------------------------------------------------ *
 * Kết quả BFS                                                          *
 * ------------------------------------------------------------------ */
typedef struct {
    int     *dist;           /* dist[v] toàn cục (chỉ hợp lệ ở rank 0) */
    int      num_levels;     /* số mức BFS                              */
    edge_t   visited_edges;  /* tổng số cạnh duyệt qua                 */
    double   time_ms;        /* thời gian tổng (wall-clock của rank 0)  */
} BFSResult;

/* ------------------------------------------------------------------ *
 * API chính                                                             *
 * ------------------------------------------------------------------ */

/*
 * Chạy BFS hybrid trên đồ thị g từ đỉnh source.
 *
 * Tất cả MPI rank đều gọi hàm này.
 * g phải giống nhau trên tất cả rank (replicated graph).
 *
 * Kết quả:
 *   - result->dist chỉ được ghi đầy đủ ở rank 0
 *   - Các rank khác: result->dist = NULL
 *
 * Gọi bfs_result_free() để giải phóng.
 */
BFSResult bfs_hybrid(const Graph *g, vertex_t source);

/* Giải phóng bộ nhớ trong BFSResult */
void bfs_result_free(BFSResult *r);

#endif /* BFS_HYBRID_H */