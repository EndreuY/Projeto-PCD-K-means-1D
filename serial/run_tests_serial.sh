#!/bin/bash
set -e
# Descobre diretório do script
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

DATA="$ROOT_DIR/dados.csv"
CENT="$ROOT_DIR/centroides_iniciais.csv"
MAX_ITER=100
EPS=1e-6
OUT="$SCRIPT_DIR/serial_results.csv"

echo "Compilando versão serial..."
gcc -O2 -std=c99 "$SCRIPT_DIR/kmeans_1d_naive.c" -o "$SCRIPT_DIR/kmeans_1d_naive" -lm

echo "Executando baseline serial..."
output=$("$SCRIPT_DIR/kmeans_1d_naive" "$DATA" "$CENT" "$MAX_ITER" "$EPS")
metrics=$(echo "$output" | grep "Iterações:")
iter=$(echo $metrics | awk '{print $2}')
sse=$(echo $metrics | awk '{print $6}')
time_ms=$(echo $metrics | awk '{print $9}')

echo "Iteracoes,SSE,Tempo_ms" > "$OUT"
echo "$iter,$sse,$time_ms" >> "$OUT"

echo "Resultado salvo em $OUT"
