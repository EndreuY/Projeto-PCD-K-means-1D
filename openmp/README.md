# K-Means 1D OpenMP

Versão paralela usando OpenMP.

## Compilação
```
gcc -O2 -fopenmp -std=c99 kmeans_1d_omp.c -o kmeans_1d_omp -lm
```

## Uso
```
export OMP_NUM_THREADS=8
./kmeans_1d_omp ../dados.csv ../centroides_iniciais.csv [max_iter=50] [eps=1e-4] [assign.csv] [centroids.csv] [schedule] [chunk]
```
Parâmetros:
- schedule: static | dynamic
- chunk: inteiro (>0) ou -1 para padrão

## Testes
Script `run_tests_openmp.sh` executa combinações de threads, schedule e chunk e gera `omp_results.csv`.

## Métricas
Speedup calculado comparando com baseline serial.
