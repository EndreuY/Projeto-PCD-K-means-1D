# K-Means 1D Serial

Implementação sequencial (C99) do K-Means 1D.

## Compilação
```
gcc -O2 -std=c99 kmeans_1d_naive.c -o kmeans_1d_naive -lm
```

## Uso
```
./kmeans_1d_naive ../dados.csv ../centroides_iniciais.csv [max_iter=50] [eps=1e-4] [assign.csv] [centroids.csv]
```
Saída inclui SSE final, iterações e tempo em ms.

## Arquivos
- kmeans_1d_naive.c: código fonte.

## Testes
Use o script `run_tests_serial.sh` para gerar um baseline.
