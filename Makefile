#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <omp.h>

#include "graph.h"
#include "bfs_hybrid.h"
#include "bfs_sequential.h"
#include "utils.h"

static void usage(const char *prog)
{
    fprintf(stderr,
            "Usage: mpirun -np <N> %s <num_vertices> <scale_factor> <seed> [--verify]\n"
            "  num_vertices  : number of graph vertices (e.g. 1000000)\n"
            "  scale_factor  : avg degree (e.g. 16)\n"
            "  seed          : random seed (e.g. 42)\n"
            "  --verify      : (optional) verify result against sequential BFS\n",
            prog);
}

int main(int argc, char *argv[])
{
    /* ---- Khởi tạo MPI -------------------------------------------- */
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    if (provided < MPI_THREAD_FUNNELED) {
        fprintf(stderr,
                "[HYBRID] Warning: MPI thread support level %d < MPI_THREAD_FUNNELED\n",
                provided);
    }

    int my_rank, num_ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

    /* ---- Parse arguments ----------------------------------------- */
    if (argc < 4) {
        if (my_rank == 0) usage(argv[0]);
        MPI_Finalize();
        return 1;
    }

    vertex_t num_vertices = (vertex_t)atoi(argv[1]);
    int      scale        = atoi(argv[2]);
    uint64_t seed         = (uint64_t)atoll(argv[3]);
    int      do_verify    = (argc >= 5 && strcmp(argv[4], "--verify") == 0);

    if (num_vertices <= 0 || scale <= 0) {
        if (my_rank == 0)
            fprintf(stderr, "[HYBRID] Invalid arguments\n");
        MPI_Finalize();
        return 1;
    }

    /* ---- In thông tin môi trường ---------------------------------- */
    int omp_threads = omp_get_max_threads();
    if (my_rank == 0) {
        printf("[HYBRID] ============================================\n");
        printf("[HYBRID] BFS Hybrid (MPI + OpenMP + Direction-Opt)\n");
        printf("[HYBRID] MPI ranks   : %d\n", num_ranks);
        printf("[HYBRID] OMP threads : %d per rank\n", omp_threads);
        printf("[HYBRID] Total cores : %d\n", num_ranks * omp_threads);
        printf("[HYBRID] Alpha       : %d\n", BFS_ALPHA);
        printf("[HYBRID] Beta        : %d\n", BFS_BETA);
        printf("[HYBRID] ============================================\n\n");
        fflush(stdout);
    }

    /* ---- Sinh đồ thị (tất cả rank đều sinh, cùng seed) ------------ *
     * Dùng replicated graph: mỗi rank giữ toàn bộ CSR.               *
     * Đây là lựa chọn phù hợp cho cluster nhỏ (4 node).              *
     * ---------------------------------------------------------------- */
    if (my_rank == 0) {
        printf("[HYBRID] Generating R-MAT graph: %d vertices, scale=%d, seed=%llu\n",
               num_vertices, scale, (unsigned long long)seed);
        fflush(stdout);
    }

    /* Barrier để rank 0 in trước */
    MPI_Barrier(MPI_COMM_WORLD);

    Timer t_gen;
    timer_start(&t_gen);
    Graph *g = graph_rmat_generate(num_vertices, scale, seed, NULL);
    double gen_ms = timer_elapsed_ms(&t_gen);

    if (!g) {
        fprintf(stderr, "[HYBRID] rank %d: graph generation failed\n", my_rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    /* Chỉ rank 0 in info */
    if (my_rank == 0) {
        graph_print_info(g);
        printf("[HYBRID] Graph generated in %.2f ms\n\n", gen_ms);
        fflush(stdout);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    /* ---- Chọn source --------------------------------------------- */
    vertex_t source = pick_source(g, seed);
    if (my_rank == 0) {
        printf("[HYBRID] BFS from source vertex %d\n\n", source);
        fflush(stdout);
    }

    /* ---- Chạy BFS Hybrid ----------------------------------------- */
    MPI_Barrier(MPI_COMM_WORLD);

    BFSResult result = bfs_hybrid(g, source);

    MPI_Barrier(MPI_COMM_WORLD);

    /* ---- In kết quả (chỉ rank 0) ---------------------------------- */
    if (my_rank == 0) {
        vertex_t visited_verts = count_visited(result.dist, g->num_vertices);
        double   mteps         = compute_mteps(result.visited_edges, result.time_ms);

        printf("\n[HYBRID] ============================================\n");
        printf("[HYBRID] Graph    : %d vertices, %lld edges\n",
               g->num_vertices, (long long)g->num_edges);
        printf("[HYBRID] Visited  : %d vertices, %lld edges\n",
               (int)visited_verts, (long long)result.visited_edges);
        printf("[HYBRID] Levels   : %d\n", result.num_levels);
        printf("[HYBRID] Time     : %.2f ms\n", result.time_ms);
        printf("[HYBRID] MTEPS    : %.2f\n", mteps);
        fflush(stdout);
    }

    /* ---- Verify (tùy chọn) --------------------------------------- */
    if (do_verify && my_rank == 0) {
        printf("\n[HYBRID] Running sequential BFS for verification...\n");
        fflush(stdout);

        int *dist_seq = (int *)malloc((size_t)g->num_vertices * sizeof(int));
        if (!dist_seq) {
            fprintf(stderr, "[HYBRID] malloc dist_seq failed\n");
        } else {
            Timer t_seq;
            timer_start(&t_seq);
            edge_t seq_edges = bfs_sequential(g, source, dist_seq);
            double seq_ms    = timer_elapsed_ms(&t_seq);

            int errors = verify_bfs(dist_seq, result.dist, g->num_vertices);

            double seq_mteps    = compute_mteps(seq_edges, seq_ms);
            double speedup      = seq_ms / result.time_ms;

            printf("[HYBRID] Sequential : %.2f ms | MTEPS: %.2f\n",
                   seq_ms, seq_mteps);
            printf("[HYBRID] Speedup    : %.2fx\n", speedup);

            if (errors == 0)
                printf("[HYBRID] Verify     : PASSED ✓\n");
            else
                printf("[HYBRID] Verify     : FAILED ✗ (%d mismatches)\n", errors);

            free(dist_seq);
        }
    }

    if (my_rank == 0) {
        printf("[HYBRID] ============================================\n");
        fflush(stdout);
    }

    /* ---- Dọn dẹp ------------------------------------------------- */
    bfs_result_free(&result);
    graph_free(g);

    MPI_Finalize();
    return 0;
}