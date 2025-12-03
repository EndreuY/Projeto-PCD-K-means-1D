#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* kmeans_1d_mpi.c
   Implementação distribuída de K-means 1D usando MPI.
   Uso: mpirun -np <P> ./kmeans_1d_mpi dados.csv centroides_iniciais.csv [max_iter=50] [eps=1e-4] [assign.csv] [centroids.csv]
*/

static int count_rows(const char *path)
{
    FILE *f = fopen(path, "r");
    if (!f)
    {
        return -1;
    }
    int rows = 0;
    char line[8192];
    while (fgets(line, sizeof(line), f))
    {
        int only_ws = 1;
        for (char *p = line; *p; p++)
        {
            if (*p != ' ' && *p != '\t' && *p != '\n' && *p != '\r')
            {
                only_ws = 0;
                break;
            }
        }
        if (!only_ws)
            rows++;
    }
    fclose(f);
    return rows;
}

static double *read_csv_1col_root(const char *path, int *n_out)
{
    int R = count_rows(path);
    if (R <= 0)
    {
        fprintf(stderr, "Erro ao abrir/ler %s\n", path);
        return NULL;
    }
    double *A = (double *)malloc((size_t)R * sizeof(double));
    if (!A)
    {
        fprintf(stderr, "Sem memoria para %d linhas\n", R);
        return NULL;
    }

    FILE *f = fopen(path, "r");
    if (!f)
    {
        fprintf(stderr, "Erro ao abrir %s\n", path);
        free(A);
        return NULL;
    }

    char line[8192];
    int r = 0;
    while (fgets(line, sizeof(line), f))
    {
        int only_ws = 1;
        for (char *p = line; *p; p++)
        {
            if (*p != ' ' && *p != '\t' && *p != '\n' && *p != '\r')
            {
                only_ws = 0;
                break;
            }
        }
        if (only_ws)
            continue;
        const char *delim = ",; \t";
        char *tok = strtok(line, delim);
        if (!tok)
        {
            fprintf(stderr, "Linha %d sem valor em %s\n", r + 1, path);
            free(A);
            fclose(f);
            return NULL;
        }
        A[r] = atof(tok);
        r++;
        if (r > R)
            break;
    }
    fclose(f);
    *n_out = R;
    return A;
}

static void write_assign_csv_root(const char *path, const int *assign, int N)
{
    if (!path)
        return;
    FILE *f = fopen(path, "w");
    if (!f)
    {
        fprintf(stderr, "Erro ao abrir %s para escrita\n", path);
        return;
    }
    for (int i = 0; i < N; i++)
        fprintf(f, "%d\n", assign[i]);
    fclose(f);
}

static void write_centroids_csv_root(const char *path, const double *C, int K)
{
    if (!path)
        return;
    FILE *f = fopen(path, "w");
    if (!f)
    {
        fprintf(stderr, "Erro ao abrir %s para escrita\n", path);
        return;
    }
    for (int c = 0; c < K; c++)
        fprintf(f, "%.6f\n", C[c]);
    fclose(f);
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 3)
    {
        if (rank == 0)
            fprintf(stderr, "Uso: %s dados.csv centroides_iniciais.csv [max_iter=50] [eps=1e-4] [assign.csv] [centroids.csv]\n", argv[0]);
        MPI_Finalize();
        return 1;
    }
    const char *pathX = argv[1];
    const char *pathC = argv[2];
    int max_iter = (argc > 3) ? atoi(argv[3]) : 50;
    double eps = (argc > 4) ? atof(argv[4]) : 1e-4;
    const char *outAssign = (argc > 5) ? argv[5] : NULL;
    const char *outCentroid = (argc > 6) ? argv[6] : NULL;

    int N = 0, K = 0;
    double *X_all = NULL;
    double *C = NULL;

    if (rank == 0)
    {
        X_all = read_csv_1col_root(pathX, &N);
        if (!X_all)
        {
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        C = read_csv_1col_root(pathC, &K);
        if (!C)
        {
            free(X_all);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    /* Envia (broadcast) K e N para todos os ranks */
    MPI_Bcast(&K, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);

    /* Verificações de integridade dos dados (N e K) */
    if (K <= 0 || N <= 0)
    {
        if (rank == 0)
            fprintf(stderr, "Dados inválidos: N=%d K=%d\n", N, K);
        MPI_Finalize();
        return 1;
    }

    /* Distribui centroides iniciais: o root possui C, outros alocam e recebem */
    if (rank != 0)
    {
        C = (double *)malloc((size_t)K * sizeof(double));
        if (!C)
        {
            fprintf(stderr, "rank %d: sem memoria\n", rank);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }
    MPI_Bcast(C, K, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    /* Prepara arrays `sendcounts` e `displs` para usar em MPI_Scatterv e distribuir os pontos */
    int *sendcounts = (int *)malloc((size_t)size * sizeof(int));
    int *displs = (int *)malloc((size_t)size * sizeof(int));
    int base = N / size;
    int rem = N % size;
    for (int r = 0; r < size; r++)
    {
        sendcounts[r] = base + (r < rem ? 1 : 0);
    }
    displs[0] = 0;
    for (int r = 1; r < size; r++)
        displs[r] = displs[r - 1] + sendcounts[r - 1];

    int Nloc = sendcounts[rank];
    double *Xloc = (double *)malloc((size_t)Nloc * sizeof(double));
    if (!Xloc)
    {
        fprintf(stderr, "rank %d: sem memoria Xloc\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    MPI_Scatterv(X_all, sendcounts, displs, MPI_DOUBLE, Xloc, Nloc, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    int *assign_local = (int *)malloc((size_t)Nloc * sizeof(int));
    if (!assign_local)
    {
        fprintf(stderr, "rank %d: sem memoria assign_local\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int iters = 0;
    double prev_sse = 1e300;
    double sse = 0.0;

    double comm_time = 0.0; /* acumula tempo gasto em chamadas de comunicação (max entre ranks) */

    double t0 = MPI_Wtime();

    for (int it = 0; it < max_iter; it++)
    {
        double *sum_local = (double *)calloc((size_t)K, sizeof(double));
        int *cnt_local = (int *)calloc((size_t)K, sizeof(int));
        if (!sum_local || !cnt_local)
        {
            fprintf(stderr, "rank %d: sem memoria sums\n", rank);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        double sse_local = 0.0;

        /* Assignment local */
        for (int i = 0; i < Nloc; i++)
        {
            int best = -1;
            double bestd = 1e300;
            for (int c = 0; c < K; c++)
            {
                double diff = Xloc[i] - C[c];
                double d = diff * diff;
                if (d < bestd)
                {
                    bestd = d;
                    best = c;
                }
            }
            assign_local[i] = best;
            cnt_local[best] += 1;
            sum_local[best] += Xloc[i];
            sse_local += bestd;
        }

        /* Preparar buffers globais */
        double *sum_global = (double *)calloc((size_t)K, sizeof(double));
        int *cnt_global = (int *)calloc((size_t)K, sizeof(int));
        if (!sum_global || !cnt_global)
        {
            fprintf(stderr, "rank %d: sem memoria global\n", rank);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        /* 1) Allreduce para sums e counts, medindo tempo local de cada chamada */
        double ar_sum_t0 = MPI_Wtime();
        MPI_Allreduce(sum_local, sum_global, K, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        double ar_sum_t1 = MPI_Wtime();
        double local_ar_sum = ar_sum_t1 - ar_sum_t0;

        double ar_cnt_t0 = MPI_Wtime();
        MPI_Allreduce(cnt_local, cnt_global, K, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        double ar_cnt_t1 = MPI_Wtime();
        double local_ar_cnt = ar_cnt_t1 - ar_cnt_t0;

        /* Reduzir os tempos locais para obter o máximo observado entre ranks */
        double max_ar_sum = 0.0, max_ar_cnt = 0.0;
        MPI_Reduce(&local_ar_sum, &max_ar_sum, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&local_ar_cnt, &max_ar_cnt, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

        /* 2) Reduce SSE_local -> sse_global (usar MPI_Reduce) */
        double sse_global = 0.0;
        double rd_sse_t0 = MPI_Wtime();
        MPI_Reduce(&sse_local, &sse_global, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        double rd_sse_t1 = MPI_Wtime();
        double local_rd_sse = rd_sse_t1 - rd_sse_t0;
        double max_rd_sse = 0.0;
        MPI_Reduce(&local_rd_sse, &max_rd_sse, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

        /* Atualiza centróides localmente em cada rank usando os resultados do Allreduce */
        for (int c = 0; c < K; c++)
        {
            if (cnt_global[c] > 0)
                C[c] = sum_global[c] / (double)cnt_global[c];
            /* se cnt_global[c]==0, mantém C[c] */
        }

        int converged = 0;
        if (rank == 0)
        {
            sse = sse_global;
            double rel = fabs(sse - prev_sse) / (prev_sse > 0.0 ? prev_sse : 1.0);
            prev_sse = sse;
            if (rel < eps)
            {
                converged = 1;
                iters = it + 1;
            }
            else
                iters = it + 1;

            /* acumula o tempo de comunicação (usando máximos por operação) */
            comm_time += max_ar_sum + max_ar_cnt + max_rd_sse;
        }

        free(sum_local);
        free(cnt_local);
        free(sum_global);
        free(cnt_global);

        /* Broadcast do sinal de convergência para todos os ranks */
        MPI_Bcast(&converged, 1, MPI_INT, 0, MPI_COMM_WORLD);

        if (converged)
            break;
    }

    double t1 = MPI_Wtime();
    double ms = 1000.0 * (t1 - t0);
    double comm_ms = 1000.0 * comm_time;

    if (outAssign && rank == 0)
    {
        int *assign_all = (int *)malloc((size_t)N * sizeof(int));
        if (assign_all)
        {
            MPI_Gatherv(assign_local, Nloc, MPI_INT, assign_all, sendcounts, displs, MPI_INT, 0, MPI_COMM_WORLD);
            write_assign_csv_root(outAssign, assign_all, N);
            free(assign_all);
        }
    }
    else if (outAssign)
    {
        MPI_Gatherv(assign_local, Nloc, MPI_INT, NULL, sendcounts, displs, MPI_INT, 0, MPI_COMM_WORLD);
    }

    if (rank == 0)
    {
        if (outCentroid)
            write_centroids_csv_root(outCentroid, C, K);

        printf("K-means 1D (MPI)\n");
        printf("N=%d K=%d max_iter=%d eps=%g\n", N, K, max_iter, eps);
        printf("Iterações: %d | SSE final: %.6f | Tempo: %.1f ms\n", iters, sse, ms);
        printf("Tempo Comunicação (reductions/allreduces - max entre ranks): %.1f ms\n", comm_ms);
    }

    free(Xloc);
    free(assign_local);
    free(sendcounts);
    free(displs);
    free(C);
    if (rank == 0)
        free(X_all);

    MPI_Finalize();
    return 0;
}
