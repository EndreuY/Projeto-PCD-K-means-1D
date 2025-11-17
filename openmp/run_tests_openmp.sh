#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

DATA="$ROOT_DIR/dados.csv"
CENT="$ROOT_DIR/centroides_iniciais.csv"
MAX_ITER=100
EPS=1e-6
OUT="$SCRIPT_DIR/omp_results.csv"

# Baseline serial
if ! grep -q "Serial" "$OUT" 2>/dev/null; then
  echo "Gerando baseline serial..."
  (cd "$ROOT_DIR/serial" && gcc -O2 -std=c99 kmeans_1d_naive.c -o kmeans_1d_naive -lm)
  serial_out=$("$ROOT_DIR/serial/kmeans_1d_naive" "$DATA" "$CENT" "$MAX_ITER" "$EPS")
  metrics_serial=$(echo "$serial_out" | grep "Iterações:")
  iters_serial=$(echo $metrics_serial | awk '{print $2}')
  sse_serial=$(echo $metrics_serial | awk '{print $6}')
  time_serial=$(echo $metrics_serial | awk '{print $9}')
  echo "Threads,Schedule,Chunk,Tempo_ms,SSE_Final,Iteracoes" > "$OUT"
  echo "1,Serial,N/A,$time_serial,$sse_serial,$iters_serial" >> "$OUT"
fi

echo "Compilando versão OpenMP..."
gcc -O2 -fopenmp -std=c99 "$SCRIPT_DIR/kmeans_1d_omp.c" -o "$SCRIPT_DIR/kmeans_1d_omp" -lm

for threads in 1 2 4 8 16; do
  export OMP_NUM_THREADS=$threads
  echo "== Threads $threads =="
  for schedule in static dynamic; do
    for chunk in default 100 1000 10000; do
      chunk_val=$chunk; [ "$chunk" = "default" ] && chunk_val=-1
      out=$("$SCRIPT_DIR/kmeans_1d_omp" "$DATA" "$CENT" "$MAX_ITER" "$EPS" "$SCRIPT_DIR/assign.tmp" "$SCRIPT_DIR/centroids.tmp" $schedule $chunk_val)
      metrics=$(echo "$out" | tail -n 1)
      iters=$(echo $metrics | awk '{print $2}')
      sse=$(echo $metrics | awk '{print $6}')
      time_ms=$(echo $metrics | awk '{print $9}')
      echo "$threads,$schedule,$chunk,$time_ms,$sse,$iters" >> "$OUT"
    done
  done
done

echo "Resultados em $OUT"
