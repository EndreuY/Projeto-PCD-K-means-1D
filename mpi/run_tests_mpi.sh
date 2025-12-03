#!/usr/bin/env bash
set -euo pipefail

# Script de testes para a versão MPI (gera mpi_results.csv)
# Usa ./kmeans_1d_mpi (compilado via mpicc) e executa mpirun para vários NP.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

DATA="$ROOT_DIR/dados.csv"
CENT="$ROOT_DIR/centroides_iniciais.csv"
MAXI=${MAXI:-50}
EPS=${EPS:-1e-4}
OUT="$SCRIPT_DIR/mpi_results.csv"
ASSIGN="$SCRIPT_DIR/assign_mpi.tmp"
CENTOUT="$SCRIPT_DIR/centroids_mpi.tmp"

# Lista de números de processos a testar (padrão)
NP_LIST=${NP_LIST:-"1 2 4 8"}

# Paths para baselines
SERIAL_CSV="$ROOT_DIR/serial/serial_results.csv"
SERIAL_TIME_MS=""

get_serial_time() {
    if [ -f "$SERIAL_CSV" ]; then
        SERIAL_TIME_MS=$(tail -n 1 "$SERIAL_CSV" | awk -F',' '{print $3}')
    else
        if [ -f "$ROOT_DIR/serial/run_tests_serial.sh" ]; then
            printf "Gerando baseline serial...\n"
            (cd "$ROOT_DIR/serial" && bash run_tests_serial.sh >/dev/null)
            if [ -f "$SERIAL_CSV" ]; then
                SERIAL_TIME_MS=$(tail -n 1 "$SERIAL_CSV" | awk -F',' '{print $3}')
            fi
        fi
    fi
}

# Compila
printf "Compilando versão MPI...\n"
(cd "$SCRIPT_DIR" && mpicc -O2 -std=c99 kmeans_1d_mpi.c -o kmeans_1d_mpi -lm)

if [ ! -x "$SCRIPT_DIR/kmeans_1d_mpi" ]; then
    printf "Erro: executável kmeans_1d_mpi não encontrado após compilação\n"
    exit 1
fi

get_serial_time
if [ -n "$SERIAL_TIME_MS" ]; then
    printf "Baseline serial detectado: %sms\n" "$SERIAL_TIME_MS"
else
    printf "Baseline serial não disponível. Speedup serial ficará como NA.\n"
fi

# Cabeçalho CSV
echo "NP,Tempo_ms,SSE_Final,Iteracoes,Speedup_Serial,Speedup_Distribuido,Tempo_Communicacao_ms" > "$OUT"

# Determina o menor NP da lista para usar como referência do speedup distribuído
MIN_NP=$(echo "$NP_LIST" | tr ' ' '\n' | awk '{print $1}' | sort -n | head -n1)
BASE_MPI_TIME=""

for NP in $NP_LIST; do
    printf "== Executando MPI com NP=%s ==\n" "$NP"
    # Executa no diretório do script
    LOG=$(cd "$SCRIPT_DIR" && mpirun --oversubscribe -np "$NP" ./kmeans_1d_mpi "$DATA" "$CENT" "$MAXI" "$EPS" "$ASSIGN" "$CENTOUT" 2>&1)
    # Exibe log para o usuário
    printf "%s\n" "$LOG"

    # Extrai métricas (compatível com saída do serial: "Iterações: %d | SSE final: %f | Tempo: %f ms")
    ITERS=$(echo "$LOG" | grep -oP 'Iterações: \K[0-9]+' || true)
    SSE=$(echo "$LOG" | grep -oP 'SSE final: \K[0-9\.eE+-]+' || true)
    TEMPO_MS=$(echo "$LOG" | grep -oP 'Tempo: \K[0-9\.eE+-]+' || true)

    # Extrai tempo de comunicação impresso pelo binário (em ms)
    # Tentativas robustas para extrair o número do campo "Tempo Comunicação"
    COMM_MS=""
    # 1) se grep -P estiver disponível e funcionando, tenta PCRE
    COMM_MS=$(echo "$LOG" | grep -oP 'Tempo Comunicação[^:]*: \\K[0-9\\.eE+-]+' 2>/dev/null || true)
    # 2) se não obteve resultado, usa grep -E + awk
    if [ -z "$COMM_MS" ]; then
        COMM_MS=$(echo "$LOG" | grep -oE 'Tempo Comunicação[^:]*: [0-9\\.eE+-]+' | awk -F': ' '{print $2}' || true)
    fi
    # 3) por fim, tenta sed (mais portátil)
    if [ -z "$COMM_MS" ]; then
        COMM_MS=$(echo "$LOG" | sed -n 's/.*Tempo Comunicação[^:]*: \([0-9.eE+-]\+\).*/\1/p' || true)
    fi
    # Se nada funcionou, ficará vazio e depois será convertido em NA
    COMM_MS=${COMM_MS:-}

    # Fallbacks vazios para evitar linhas quebradas
    [ -z "$ITERS" ] && ITERS="NA"
    [ -z "$SSE" ] && SSE="NA"
    [ -z "$TEMPO_MS" ] && TEMPO_MS="NA"
    [ -z "$COMM_MS" ] && COMM_MS="NA"

    if [ -n "$SERIAL_TIME_MS" ] && [ "$TEMPO_MS" != "NA" ]; then
        SPEEDUP=$(awk -v s="$SERIAL_TIME_MS" -v t="$TEMPO_MS" 'BEGIN{ if(t>0){ printf "%.4f", s/t } else { print "NA" } }')
    else
        SPEEDUP="NA"
    fi

    # Guarda tempo base do MPI (menor NP) para calcular speedup distribuído
    if [ "$NP" = "$MIN_NP" ] && [ "$TEMPO_MS" != "NA" ]; then
        BASE_MPI_TIME="$TEMPO_MS"
    fi

    if [ -n "$BASE_MPI_TIME" ] && [ "$TEMPO_MS" != "NA" ]; then
        SPEEDUP_DIST=$(awk -v b="$BASE_MPI_TIME" -v t="$TEMPO_MS" 'BEGIN{ if(t>0){ printf "%.4f", b/t } else { print "NA" } }')
    else
        SPEEDUP_DIST="NA"
    fi

    echo "$NP,$TEMPO_MS,$SSE,$ITERS,$SPEEDUP,$SPEEDUP_DIST,$COMM_MS" >> "$OUT"
done

printf "Resultados salvos em %s\n" "$OUT"
