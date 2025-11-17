# K-Means 1D CUDA

Versão paralela em GPU usando CUDA.

## Compilação
```
nvcc -O2 kmeans_1d_cuda.cu -o kmeans_1d_cuda
```

## Uso
```
./kmeans_1d_cuda ../dados.csv ../centroides_iniciais.csv [max_iter=50] [eps=1e-4] [assign.csv] [centroids.csv]
```

## Testes
`run_tests_cuda.sh` compila e executa múltiplas rodadas, salvando resultados em `results_cuda.csv`.

## Observação
Ajuste blocos e threads dentro do código conforme a GPU disponível.
