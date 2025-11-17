# K-Means 1D: Serial, OpenMP e CUDA

Projeto de comparação de desempenho entre implementações do algoritmo K-Means em 1 dimensão:
- Versão Serial (C)
- Versão OpenMP (paralela em CPU)
- Versão CUDA (paralela em GPU)

## Estrutura do Projeto
```
serial/
  kmeans_1d_naive.c
  run_tests_serial.sh
  README.md
openmp/
  kmeans_1d_omp.c
  run_tests_openmp.sh
  README.md
cuda/
  kmeans_1d_cuda.cu
  run_tests_cuda.sh
  README.md
centroides_iniciais.csv   # Centróides iniciais (compartilhado)
dados.csv                 # Dados de entrada 1D (compartilhado)
gera_dados.py             # Geração de dados sintéticos (compartilhado)
```

## Requisitos
- GCC (para Serial/OpenMP)
- Suporte a OpenMP (`-fopenmp`)
- NVIDIA CUDA Toolkit (para versão GPU)
- Python 3 + pandas + seaborn + matplotlib (para análise)

Instalar dependências Python:
```
pip install pandas seaborn matplotlib
```

## Compilação
### Serial
```
cd serial
gcc -O2 -std=c99 kmeans_1d_naive.c -o kmeans_1d_naive -lm
```
### OpenMP
```
cd openmp
gcc -O2 -fopenmp -std=c99 kmeans_1d_omp.c -o kmeans_1d_omp -lm
```
### CUDA
```
cd cuda
nvcc -O2 kmeans_1d_cuda.cu -o kmeans_1d_cuda
```

## Uso Básico
### Serial
```
./serial/kmeans_1d_naive dados.csv centroides_iniciais.csv [max_iter=50] [eps=1e-4] [assign.csv] [centroids.csv]
```
### OpenMP
```
export OMP_NUM_THREADS=8
./openmp/kmeans_1d_omp dados.csv centroides_iniciais.csv [max_iter=50] [eps=1e-4] [assign.csv] [centroids.csv] [schedule] [chunk]
Exemplo:
./openmp/kmeans_1d_omp dados.csv centroides_iniciais.csv 50 1e-4 assign.csv centroids.csv static 100
```
### CUDA
```
./cuda/kmeans_1d_cuda dados.csv centroides_iniciais.csv [max_iter=50] [eps=1e-4] [assign.csv] [centroids.csv]
```

## Scripts de Teste
Os scripts estão em cada subpasta:
- Serial: `serial/run_tests_serial.sh` gera baseline em `serial_results.csv`
- OpenMP: `openmp/run_tests_openmp.sh` gera `omp_results.csv`
- CUDA: `cuda/run_tests_cuda.sh` gera `results_cuda.csv`

Executar exemplo (OpenMP):
```
cd openmp
bash run_tests_openmp.sh
```

## Métricas
Cada execução imprime:
```
Iterações: <n> | SSE final: <valor> | Tempo: <ms>
```
Speedup = Tempo Serial / Tempo Paralelo.

## Geração de Dados
```
python3 gera_dados.py
```

