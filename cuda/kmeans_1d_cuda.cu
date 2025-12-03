#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

// Includes do CUDA
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

/* Macro para verificação de erros CUDA */
#define CUDA_CHECK(err) { \
    cudaError_t err_ = (err); \
    if (err_ != cudaSuccess) { \
        fprintf(stderr, "Erro CUDA (%s:%d): %s\n", __FILE__, __LINE__, cudaGetErrorString(err_)); \
        exit(1); \
    } \
}

/* ===================================================================
 * Funções auxiliares (util.c) - copiadas da versão sequencial
 * =================================================================== */
static int count_rows(const char *path) {
    FILE *f = fopen(path, "r");
    if(!f) { fprintf(stderr, "Erro ao abrir %s\n", path); exit(1); }
    int rows = 0; char line[8192];
    while(fgets(line, sizeof(line), f)){
        int only_ws=1;
        for(char *p=line; *p; p++) {
            if(*p!=' ' && *p!='\t' && *p!='\n' && *p!='\r'){ only_ws=0; break; }
        }
        if(!only_ws) rows++;
    }
    fclose(f);
    return rows;
}

static double *read_csv_1col(const char *path, int *n_out){
    int R = count_rows(path);
    if(R<=0){ fprintf(stderr, "Arquivo vazio: %s\n", path); exit(1); }
    double *A = (double*)malloc((size_t)R * sizeof(double));
    if(!A){ fprintf(stderr, "Sem memoria para %d linhas\n", R); exit(1); }
    FILE *f = fopen(path, "r");
    if(!f) { fprintf(stderr, "Erro ao abrir %s\n", path); free(A); exit(1); }
    
    char line[8192];
    int r=0;
    while(fgets(line, sizeof(line), f)){
        int only_ws=1;
        for(char *p=line; *p; p++) {
            if(*p!=' ' && *p!='\t' && *p!='\n' && *p!='\r'){ only_ws=0; break; }
        }
        if(only_ws) continue;
        
        const char *delim = ",; \t";
        char *tok = strtok(line, delim);
        if(!tok) { fprintf(stderr, "Linha %d sem valor em %s\n", r+1, path); free(A); exit(1); }
        A[r] = atof(tok);
        r++;
        if(r>R) break; 
    }
    fclose(f);
    *n_out = R;
    return A;
}

static void write_assign_csv(const char *path, const int *assign, int N){
    if(!path) return;
    FILE *f = fopen(path, "w");
    if(!f) { fprintf(stderr, "Erro ao abrir %s para escrita\n", path); return; }
    for(int i=0;i<N;i++) fprintf(f, "%d\n", assign[i]);
    fclose(f);
}

static void write_centroids_csv(const char *path, const double *C, int K){
    if(!path) return;
    FILE *f = fopen(path, "w");
    if(!f) { fprintf(stderr, "Erro ao abrir %s para escrita\n", path); return; }
    for(int c=0;c<K;c++) fprintf(f, "%.6f\n", C[c]);
    fclose(f);
}

/* ===================================================================
 * Lógica K-means (Sequencial - apenas Update)
 * =================================================================== */

/* update: média dos pontos de cada cluster (1D)
 * Esta função é executada no HOST (CPU) */
static void update_step_1d(const double *X, double *C, const int *assign, int N, int K) {
    double *sum = (double*) calloc((size_t)K, sizeof(double));
    int *cnt = (int*)calloc((size_t)K, sizeof(int));
    if(!sum || !cnt) { fprintf(stderr, "Sem memoria no update\n"); exit(1); }

    for(int i=0;i<N;i++){
        int a = assign[i];
        cnt[a] += 1;
        sum[a] += X[i];
    }
    
    for(int c=0; c<K; c++) {
        if(cnt[c]>0) C[c] = sum[c] / (double)cnt[c];
        else C[c] = X[0];
    }
    free(sum); free(cnt);
}

/* ===================================================================
 * Lógica K-means (CUDA - Kernel de Assignment)
 * =================================================================== */

__global__
void assignment_kernel(const double *X, const double *C, int *assign, double *errors, int N, int K) 
{
    // ID global da thread (1 thread por ponto i)
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        int best = -1;
        double bestd = 1e300;
        double xi = X[i]; 
        // Varre K centróides
        for (int c = 0; c < K; c++) {
            double diff = xi - C[c];
            double d = diff * diff; // (X[i] - C[c])^2
            if (d < bestd) {
                bestd = d;
                best = c;
            }
        }
        assign[i] = best; // escreve assign[i] 
        errors[i] = bestd; // Salva erro para redução do SSE 
    }
}

/* Função principal do K-means (Host) */
static void kmeans_1d_cuda(const double *h_X, double *h_C, int *h_assign,
                           int N, int K, int max_iter, double eps,
                           int *iters_out, double *sse_out, double *ms_out,
                           int block_size) // <-- PARÂMETRO
{
    double *d_X, *d_C, *d_errors;
    int *d_assign;
    CUDA_CHECK(cudaMalloc(&d_X, (size_t)N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_C, (size_t)K * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_assign, (size_t)N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_errors, (size_t)N * sizeof(double)));

    double *h_errors = (double*)malloc((size_t)N * sizeof(double));
    if (!h_errors) { fprintf(stderr, "Sem memoria para h_errors\n"); exit(1); }

    clock_t t0_total = clock(); // Medir tempo total
    
    // Medir H2D inicial 
    clock_t t0_h2d = clock();
    CUDA_CHECK(cudaMemcpy(d_X, h_X, (size_t)N * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_C, h_C, (size_t)K * sizeof(double), cudaMemcpyHostToDevice));
    clock_t t1_h2d = clock();
    double ms_h2d_init = 1000.0 * (double)(t1_h2d - t0_h2d) / (double)CLOCKS_PER_SEC;

    double prev_sse = 1e300;
    double sse = 0.0;
    int it;
    double ms_kernel_total = 0.0;
    double ms_d2h_total = 0.0;
    double ms_h2d_total = ms_h2d_init;
    double ms_update_host_total = 0.0;

    for (it = 0; it < max_iter; it++) {
        
        // Lançar Kernel de Assignment 
        int gridSize = (N + block_size - 1) / block_size;
        
        clock_t t0_kernel = clock();
        assignment_kernel<<<gridSize, block_size>>>(d_X, d_C, d_assign, d_errors, N, K);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize()); // Sincroniza para medir o tempo
        clock_t t1_kernel = clock();
        ms_kernel_total += 1000.0 * (double)(t1_kernel - t0_kernel) / (double)CLOCKS_PER_SEC;


        // Calcular SSE (Redução no Host)
        clock_t t0_d2h_err = clock();
        CUDA_CHECK(cudaMemcpy(h_errors, d_errors, (size_t)N * sizeof(double), cudaMemcpyDeviceToHost));
        clock_t t1_d2h_err = clock();
        ms_d2h_total += 1000.0 * (double)(t1_d2h_err - t0_d2h_err) / (double)CLOCKS_PER_SEC;
        
        sse = 0.0;
        for (int i = 0; i < N; i++) {
            sse += h_errors[i];
        }

        //Verificar convergência
        double rel = fabs(sse - prev_sse) / (prev_sse > 0.0 ? prev_sse : 1.0);
        if (rel < eps) { 
            it++;
            break; 
        }
        prev_sse = sse;

        //Update (Opção A: Host)
        clock_t t0_d2h_assign = clock();
        CUDA_CHECK(cudaMemcpy(h_assign, d_assign, (size_t)N * sizeof(int), cudaMemcpyDeviceToHost));
        clock_t t1_d2h_assign = clock();
        ms_d2h_total += 1000.0 * (double)(t1_d2h_assign - t0_d2h_assign) / (double)CLOCKS_PER_SEC;
        
        clock_t t0_update = clock();
        update_step_1d(h_X, h_C, h_assign, N, K);
        clock_t t1_update = clock();
        ms_update_host_total += 1000.0 * (double)(t1_update - t0_update) / (double)CLOCKS_PER_SEC;

        clock_t t0_h2d_centroids = clock();
        CUDA_CHECK(cudaMemcpy(d_C, h_C, (size_t)K * sizeof(double), cudaMemcpyHostToDevice));
        clock_t t1_h2d_centroids = clock();
        ms_h2d_total += 1000.0 * (double)(t1_h2d_centroids - t0_h2d_centroids) / (double)CLOCKS_PER_SEC;
    }

    clock_t t1_total = clock();
    *ms_out = 1000.0 * (double)(t1_total - t0_total) / (double)CLOCKS_PER_SEC;

    // Copia 'assign' final para o host (para salvar no arquivo)
    clock_t t0_d2h_final = clock();
    CUDA_CHECK(cudaMemcpy(h_assign, d_assign, (size_t)N * sizeof(int), cudaMemcpyDeviceToHost));
    clock_t t1_d2h_final = clock();
    ms_d2h_total += 1000.0 * (double)(t1_d2h_final - t0_d2h_final) / (double)CLOCKS_PER_SEC;


    // Imprime medições de tempo detalhadas 
    printf("--- Detalhe Tempos (ms) ---\n");
    printf("Total H2D: %.1f\n", ms_h2d_total);
    printf("Total D2H: %.1f\n", ms_d2h_total);
    printf("Total Kernel: %.1f\n", ms_kernel_total);
    printf("Total Update (Host): %.1f\n", ms_update_host_total);
    printf("---------------------------\n");


    free(h_errors);
    CUDA_CHECK(cudaFree(d_X));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaFree(d_assign));
    CUDA_CHECK(cudaFree(d_errors));

    *iters_out = it;
    *sse_out = sse;
}

/* ===================================================================
 * Main
 * =================================================================== */
int main(int argc, char **argv) {
    if(argc < 3){
        printf("Uso: %s dados.csv centroides_iniciais.csv [max_iter=50] [eps=1e-4] [out_assign.csv] [out_centroids.csv] [blockSize=256]\n", argv[0]);
        return 1;
    }

    const char *pathX = argv[1];
    const char *pathC = argv[2];
    int max_iter = (argc > 3) ? atoi(argv[3]) : 50;
    double eps = (argc > 4) ? atof(argv[4]) : 1e-4;
    const char *outAssign = (argc > 5) ? argv[5] : NULL;
    const char *outCentroid = (argc > 6) ? argv[6] : NULL;
    int block_size = (argc > 7) ? atoi(argv[7]) : 256;

    if(max_iter <= 0 || eps <= 0.0) {
        fprintf(stderr, "Parâmetros inválidos: max_iter>0 e eps>0\n"); return 1;
    }

    int N=0, K=0;
    double *X = read_csv_1col(pathX, &N);
    double *C = read_csv_1col(pathC, &K);
    int *assign = (int*)malloc((size_t)N * sizeof(int));
    if(!assign) { fprintf(stderr, "Sem memoria para assign\n"); free(X); free(C); return 1; }

    int iters = 0; 
    double sse = 0.0;
    double ms = 0.0;
    
    // Passa o block_size para a função
    kmeans_1d_cuda(X, C, assign, N, K, max_iter, eps, &iters, &sse, &ms, block_size);

    printf("K-means 1D (CUDA - Etapa 2)\n");
    printf("N=%d K=%d max_iter=%d eps=%g blockSize=%d\n", N, K, max_iter, eps, block_size);
    printf("Iterações: %d | SSE final: %.6f | Tempo Total: %.1f ms\n", iters, sse, ms); // [cite: 23]

    write_assign_csv(outAssign, assign, N);
    write_centroids_csv(outCentroid, C, K);

    free(assign); 
    free(X); 
    free(C);
    
    CUDA_CHECK(cudaDeviceReset());
    return 0;
}