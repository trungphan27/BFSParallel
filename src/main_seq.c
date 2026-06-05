#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "graph.h"
#include "bfs_sequential.h"
#include "utils.h"

static void usage(const char *prog)
{
    fprintf(stderr,
            "Usage: %s <num_vertices> <scale_factor> <seed>\n"
            "  num_vertices  : number of graph vertices (e.g. 1000000)\n"
            "  scale_factor  : avg degree (e.g. 16)\n"
            "  seed          : random seed (e.g. 42)\n",
            prog);
}

int main(int argc, char *argv[])
{
    if (argc < 4) { usage(argv[0]); return 1; }

    vertex_t num_vertices = (vertex_t)atoi(argv[1]);
    int      scale        = atoi(argv[2]);
    uint64_t seed         = (uint64_t)atoll(argv[3]);

    if (num_vertices <= 0 || scale <= 0) {
        fprintf(stderr, "[SEQ] Invalid arguments\n");
        return 1;
    }

    /* ---- Sinh đồ thị --------------------------------------------- */
    printf("[SEQ] Generating R-MAT graph: %d vertices, scale=%d, seed=%llu\n",
           num_vertices, scale, (unsigned long long)seed);

    Timer t_gen;
    timer_start(&t_gen);
    Graph *g = graph_rmat_generate(num_vertices, scale, seed, NULL);
    double gen_ms = timer_elapsed_ms(&t_gen);

    if (!g) { fprintf(stderr, "[SEQ] Graph generation failed\n"); return 1; }

    graph_print_info(g);
    printf("[SEQ] Graph generated in %.2f ms\n\n", gen_ms);

    /* ---- Chọn source --------------------------------------------- */
    vertex_t source = pick_source(g, seed);
    printf("[SEQ] BFS from source vertex %d\n", source);

    /* ---- Cấp phát dist[] ----------------------------------------- */
    int *dist = (int *)malloc((size_t)g->num_vertices * sizeof(int));
    if (!dist) { fprintf(stderr, "[SEQ] malloc dist failed\n"); return 1; }

    /* ---- Chạy BFS ------------------------------------------------ */
    Timer t_bfs;
    timer_start(&t_bfs);
    edge_t visited_edges = bfs_sequential(g, source, dist);
    double bfs_ms = timer_elapsed_ms(&t_bfs);

    /* ---- In kết quả --------------------------------------------- */
    vertex_t visited_verts = count_visited(dist, g->num_vertices);
    double   mteps         = compute_mteps(visited_edges, bfs_ms);

    printf("[SEQ] Graph: %d vertices, %lld edges\n",
           g->num_vertices, (long long)g->num_edges);
    printf("[SEQ] Visited: %d vertices, %lld edges\n",
           (int)visited_verts, (long long)visited_edges);
    printf("[SEQ] Time: %.2f ms\n", bfs_ms);
    printf("[SEQ] MTEPS: %.2f\n", mteps);

    free(dist);
    graph_free(g);
    return 0;
}