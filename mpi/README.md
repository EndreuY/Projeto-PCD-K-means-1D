# K-Means 1D MPI (Distribuído)

Implementação distribuída com MPI do K-Means 1D.

## Requisitos
- MPI (OpenMPI ou MPICH)
- gcc

## Compilação
```
mpicc -O2 -std=c99 kmeans_1d_mpi.c -o kmeans_1d_mpi -lm
```

## Uso
```
mpirun -np <NUM_PROC> ./kmeans_1d_mpi ../dados.csv ../centroides_iniciais.csv [max_iter=50] [eps=1e-4] [assign.csv] [centroids.csv]
```

## Testes
Use `run_tests_mpi.sh` para executar benchmarks.
