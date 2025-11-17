#!/bin/bash
#
# run_tests_cuda_500k.sh
# Script para testar a implementação CUDA (Etapa 2)
#
# Foco: Manter os dados (N=500k, K=8) fixos e variar o Tamanho do Bloco.
#
# NOVO: Salva os resultados em um CSV com TODOS os tempos detalhados.
#

# --- Configuração ---
MAX_ITER=50
EPS=1e-6

# Diretórios robustos
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# 1. Parâmetros dos Dados (Fixos)
DATA_N=500000
DATA_K=8

# 2. Parâmetros CUDA (Variáveis)
declare -a BLOCK_SIZES=(128 256 512 1024)

# 3. Arquivos (usar caminhos absolutos)
NVCC="nvcc"
NVCC_FLAGS="-O2"
EXE="$SCRIPT_DIR/kmeans_1d_cuda"
SRC="$SCRIPT_DIR/kmeans_1d_cuda.cu"
CSV_FILE="$SCRIPT_DIR/results_cuda.csv"
DATA_FILE="$ROOT_DIR/dados.csv"
CENT_FILE="$ROOT_DIR/centroides_iniciais.csv"

# --- Baselines (Serial & OpenMP) ---
SERIAL_CSV="$ROOT_DIR/serial/serial_results.csv"
OMP_CSV="$ROOT_DIR/openmp/omp_results.csv"
SERIAL_TIME_MS=""
OMP_BEST_TIME_MS=""

get_serial_time() {
    if [ -f "$SERIAL_CSV" ]; then
        SERIAL_TIME_MS=$(tail -n 1 "$SERIAL_CSV" | awk -F',' '{print $3}')
    elif [ -f "$ROOT_DIR/serial/run_tests_serial.sh" ]; then
        printf "Gerando baseline serial...\n"
        (cd "$ROOT_DIR/serial" && bash run_tests_serial.sh >/dev/null)
        [ -f "$SERIAL_CSV" ] && SERIAL_TIME_MS=$(tail -n 1 "$SERIAL_CSV" | awk -F',' '{print $3}')
    fi
}

get_omp_best_time() {
    if [ -f "$OMP_CSV" ]; then
        OMP_BEST_TIME_MS=$(awk -F',' 'NR>1 && $2!="Serial" { if(min=="" || $4<min) min=$4 } END{ if(min!="") printf "%s", min }' "$OMP_CSV")
    elif [ -f "$ROOT_DIR/openmp/run_tests_openmp.sh" ]; then
        printf "Gerando resultados OpenMP para baseline de speedup...\n"
        (cd "$ROOT_DIR/openmp" && bash run_tests_openmp.sh >/dev/null)
        [ -f "$OMP_CSV" ] && OMP_BEST_TIME_MS=$(awk -F',' 'NR>1 && $2!="Serial" { if(min=="" || $4<min) min=$4 } END{ if(min!="") printf "%s", min }' "$OMP_CSV")
    fi
}

get_serial_time
get_omp_best_time

# --- Funções ---
hr() {
    printf '=%.0s' $(seq 1 $(($(tput cols)-1)))
    printf "\n"
}

# --- Preparação ---
printf "Preparando ambiente de teste (CUDA - Foco: BlockSize)...\n"

# 1. Verifica arquivos
if [ ! -f "$SRC" ]; then
    printf "Erro: Arquivo $SRC não encontrado!\n"
    exit 1
fi

# 2. Compila o código CUDA
printf "Compilando $SRC com $NVCC $NVCC_FLAGS...\n"
$NVCC $NVCC_FLAGS "$SRC" -o "$EXE"
if [ $? -ne 0 ]; then
    printf "Falha na compilação! Abortando.\n"
    exit 1
fi
printf "Compilação bem-sucedida: $EXE\n"

# Informações de baseline
if [ -n "$SERIAL_TIME_MS" ]; then
    printf "Baseline Serial (Tempo_ms): %s\n" "$SERIAL_TIME_MS"
else
    printf "Baseline Serial indisponível.\n"
fi
if [ -n "$OMP_BEST_TIME_MS" ]; then
    printf "Melhor Tempo OpenMP (ms): %s\n" "$OMP_BEST_TIME_MS"
else
    printf "Baseline OpenMP indisponível.\n"
fi

# --- Execução dos Testes ---
hr
printf "Iniciando testes (MAX_ITER=$MAX_ITER, EPS=$EPS)...\n"
printf "Dataset: N=%'d, K=%d (Fixo)\n" $DATA_N $DATA_K
hr

# 4. Cria o cabeçalho do CSV (agora com Throughput e Speedups)
echo "BlockSize,Tempo_Total_ms,SSE_Final,Iteracoes,Tempo_H2D_ms,Tempo_D2H_ms,Tempo_Kernel_ms,Tempo_Update_Host_ms,Throughput_pts_s,Speedup_Serial,Speedup_OMP" > $CSV_FILE
printf "Salvando resultados em $CSV_FILE\n\n"

# Loop pelos tamanhos de bloco
for B in ${BLOCK_SIZES[@]}; do
    # Substitui printf por echo para evitar erro "invalid option" observado
    echo "--- Executando com blockSize = $B ---"
    LOG=$($EXE "$DATA_FILE" "$CENT_FILE" $MAX_ITER $EPS NULL NULL $B)
    echo "$LOG"
    # --- Extração de Dados ---
    ITERS=$(echo "$LOG" | grep -oP 'Iterações: \K[0-9]+')
    SSE=$(echo "$LOG" | grep -oP 'SSE final: \K[0-9\.]+')
    TEMPO_TOTAL=$(echo "$LOG" | grep -oP 'Tempo Total: \K[0-9\.]+')
    TEMPO_H2D=$(echo "$LOG" | grep -oP 'Total H2D: \K[0-9\.]+')
    TEMPO_D2H=$(echo "$LOG" | grep -oP 'Total D2H: \K[0-9\.]+')
    TEMPO_KERNEL=$(echo "$LOG" | grep -oP 'Total Kernel: \K[0-9\.]+')
    TEMPO_UPDATE=$(echo "$LOG" | grep -oP 'Total Update \(Host\): \K[0-9\.]+')
    # --- Métricas adicionais ---
    THROUGHPUT=$(awk -v n=$DATA_N -v t=$TEMPO_TOTAL 'BEGIN{ if(t>0){ printf "%.2f", n*1000/t } else { print "0" } }')
    if [ -n "$SERIAL_TIME_MS" ] && [ -n "$TEMPO_TOTAL" ]; then
        SPEEDUP_SERIAL=$(awk -v s=$SERIAL_TIME_MS -v t=$TEMPO_TOTAL 'BEGIN{ if(t>0){ printf "%.4f", s/t } else { print "0" } }')
    else
        SPEEDUP_SERIAL="NA"
    fi
    if [ -n "$OMP_BEST_TIME_MS" ] && [ -n "$TEMPO_TOTAL" ]; then
        SPEEDUP_OMP=$(awk -v o=$OMP_BEST_TIME_MS -v t=$TEMPO_TOTAL 'BEGIN{ if(t>0){ printf "%.4f", o/t } else { print "0" } }')
    else
        SPEEDUP_OMP="NA"
    fi
    # 5. Escreve a linha de dados completa no arquivo CSV
    echo "$B,$TEMPO_TOTAL,$SSE,$ITERS,$TEMPO_H2D,$TEMPO_D2H,$TEMPO_KERNEL,$TEMPO_UPDATE,$THROUGHPUT,$SPEEDUP_SERIAL,$SPEEDUP_OMP" >> $CSV_FILE
    printf "\n"
done

hr
printf "Testes concluídos. Resultados detalhados salvos em %s\n" $CSV_FILE