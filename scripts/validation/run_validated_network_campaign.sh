#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/validation/run_validated_network_campaign.sh [options]

Run the validated network-first anomaly campaign on the active UPF ogstun path.

Default manual scenarios:
  controlled_delay        levels: 0,50,100,200 ms
  controlled_jitter       levels: 100 ms base + 0,20,50,80 ms jitter
  tunnel_packet_loss      levels: 0,1,5,15,30 %
  tunnel_bandwidth_limit  levels: unlimited,5,2,1,0.5 Mbit/s

Optional playbook scenarios can also be launched by adding them to --scenarios:
  upf_stress, radio_interference, far_ue_poor_radio

Options:
  --scenarios "LIST"          Space/comma separated scenario names.
                              default: controlled_delay controlled_jitter tunnel_packet_loss tunnel_bandwidth_limit
  --prometheus-url URL        default: $PROMETHEUS_URL or http://172.28.2.76:31004
  --out-root DIR              default: results/manual-network
  --k8s-ssh-target TARGET     default: root@sopnode-f1
  --iperf-ssh-target TARGET   default: root@sopnode-f1
  --qhat-ssh-target TARGET    default: root@qhat01
  --ue-host NAME              UE identifier written to metadata.
                              default: host part of --qhat-ssh-target
  --ue-ip IP                  default: 14.1.0.1
  --ue-gateway IP             default: 14.1.1.1
  --ue-iface IFACE            default: wwan0
  --target-ip IP              default: 172.28.2.76
  --ul-match-mode MODE        UL IFB filter: ue or all, default: ue
  --iperf-port PORT           default: 5201
  --iperf-parallel N          TCP streams per iperf run, default: 1
  --iperf-timeout-extra N     Extra seconds before killing iperf/ssh, default: 45
  --namespace NAME            default: open5gs
  --upf-pod-regex REGEX       default: open5gs-upf2
  --upf-iface IFACE           default: ogstun
  --kubeconfig PATH           default: /root/.kube/config
  --delay-levels "LIST"       default: 0 50 100 200
  --jitter-levels "LIST"      default: 0 20 50 80
  --jitter-base-ms N          default: 100
  --loss-levels "LIST"        default: 0 1 5 15 30
  --bandwidth-levels "LIST"   default: 0 5 2 1 0.5
  --directions "LIST"         Traffic/anomaly directions: dl, ul, or "dl ul".
                              default: dl
  --iperf-seconds N           default: 45
  --ping-count N              default: 120
  --ping-interval SECONDS     default: 0.2
  --settle-seconds SECONDS    default: 3
  --tail-seconds SECONDS      default: 3
  --step STEP                 Prometheus query_range step, default: 2s
  --discover-metrics          Also discover matching Prometheus metrics.
                              Disabled by default to avoid huge CSVs.
  --discover-metrics-regex R  Regex for discovered Prometheus metric names.
  --discover-metrics-limit N  Max discovered metric names, default: 160.
  --all-prometheus-metrics    Discover all metric names with regex '.*'.
                              Use with a coarse --step first to control disk use.
  --min-free-gb N             Stop if free disk under N GiB, default: 5.
  --max-run-gb N              Stop if this campaign grows over N GiB, default: 25.
                              Use 0 to disable either disk guard.
  --health-ping-count N       Per-level clean-path health ping count, default: 5.
  --health-ping-interval S    Per-level clean-path health ping interval, default: 0.2.
  --no-level-health-gate      Do not skip levels that fail health checks.
  --no-gzip-prometheus        Keep Prometheus CSV files uncompressed.
  --skip-preflight-probes     Do not fail early on ping/iperf preflight.
  --no-iperf                  Skip iperf probes and collect ping/Prometheus only.
  -h, --help                  Show this help.

Example:
  export PROMETHEUS_URL=http://172.28.2.76:31004
  scripts/validation/run_validated_network_campaign.sh

Full campaign including playbook-based radio/stress scenarios:
  scripts/validation/run_validated_network_campaign.sh \
    --scenarios "controlled_delay controlled_jitter tunnel_packet_loss tunnel_bandwidth_limit upf_stress radio_interference far_ue_poor_radio"
EOF
}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

SCENARIOS_RAW="${SCENARIOS:-controlled_delay controlled_jitter tunnel_packet_loss tunnel_bandwidth_limit}"
PROM_URL="${PROMETHEUS_URL:-http://172.28.2.76:31004}"
OUT_ROOT="$ROOT_DIR/results/manual-network"
K8S_SSH_TARGET="${K8S_SSH_TARGET:-root@sopnode-f1}"
IPERF_SSH_TARGET="${IPERF_SSH_TARGET:-root@sopnode-f1}"
QHAT_SSH_TARGET="${QHAT_SSH_TARGET:-root@qhat01}"
UE_HOST="${UE_HOST:-}"
UE_IP="${UE_IP:-14.1.0.1}"
UE_GATEWAY="${UE_GATEWAY:-14.1.1.1}"
UE_IFACE="${UE_IFACE:-wwan0}"
TARGET_IP="${TARGET_IP:-172.28.2.76}"
UL_MATCH_MODE="${UL_MATCH_MODE:-ue}"
IPERF_PORT="${IPERF_PORT:-5201}"
IPERF_PARALLEL="${IPERF_PARALLEL:-1}"
IPERF_TIMEOUT_EXTRA="${IPERF_TIMEOUT_EXTRA:-45}"
VALIDATION_NAMESPACE="${VALIDATION_NAMESPACE:-open5gs}"
UPF_POD_REGEX="${UPF_POD_REGEX:-open5gs-upf2}"
UPF_IFACE="${UPF_IFACE:-ogstun}"
KUBECONFIG_REMOTE="${KUBECONFIG_REMOTE:-/root/.kube/config}"
DELAY_LEVELS_RAW="${DELAY_LEVELS:-0 50 100 200}"
JITTER_LEVELS_RAW="${JITTER_LEVELS:-0 20 50 80}"
JITTER_BASE_MS="${JITTER_BASE_MS:-100}"
LOSS_LEVELS_RAW="${LOSS_LEVELS:-0 1 5 15 30}"
BANDWIDTH_LEVELS_RAW="${BANDWIDTH_LEVELS:-0 5 2 1 0.5}"
DIRECTIONS_RAW="${DIRECTIONS:-dl}"
IPERF_SECONDS="${IPERF_SECONDS:-45}"
PING_COUNT="${PING_COUNT:-120}"
PING_INTERVAL="${PING_INTERVAL:-0.2}"
SETTLE_SECONDS="${SETTLE_SECONDS:-3}"
TAIL_SECONDS="${TAIL_SECONDS:-3}"
STEP="${STEP:-2s}"
RUN_IPERF=1
DISCOVER_METRICS=0
GZIP_PROMETHEUS=1
PREFLIGHT_PROBES=1
MIN_FREE_GB="${MIN_FREE_GB:-5}"
MAX_RUN_GB="${MAX_RUN_GB:-25}"
HEALTH_PING_COUNT="${HEALTH_PING_COUNT:-5}"
HEALTH_PING_INTERVAL="${HEALTH_PING_INTERVAL:-0.2}"
LEVEL_HEALTH_GATE="${LEVEL_HEALTH_GATE:-1}"
TBF_BURST="${TBF_BURST:-64kb}"
TBF_LATENCY="${TBF_LATENCY:-1000ms}"
DISCOVER_REGEX="${DISCOVER_REGEX:-(?i)(gtp|tcp|same_packet|throughput|goodput|mac_throughput|oai_gnb|rsrp|mcs|bler|snr|prach|container_cpu|container_memory|node_netstat_Tcp)}"
DISCOVER_LIMIT="${DISCOVER_LIMIT:-160}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --scenarios) SCENARIOS_RAW="${2:-}"; shift 2 ;;
    --prometheus-url) PROM_URL="${2:-}"; shift 2 ;;
    --out-root) OUT_ROOT="${2:-}"; shift 2 ;;
    --k8s-ssh-target) K8S_SSH_TARGET="${2:-}"; shift 2 ;;
    --iperf-ssh-target) IPERF_SSH_TARGET="${2:-}"; shift 2 ;;
    --qhat-ssh-target) QHAT_SSH_TARGET="${2:-}"; shift 2 ;;
    --ue-host) UE_HOST="${2:-}"; shift 2 ;;
    --ue-ip) UE_IP="${2:-}"; shift 2 ;;
    --ue-gateway) UE_GATEWAY="${2:-}"; shift 2 ;;
    --ue-iface) UE_IFACE="${2:-}"; shift 2 ;;
    --target-ip) TARGET_IP="${2:-}"; shift 2 ;;
    --ul-match-mode) UL_MATCH_MODE="${2:-}"; shift 2 ;;
    --iperf-port) IPERF_PORT="${2:-}"; shift 2 ;;
    --iperf-parallel) IPERF_PARALLEL="${2:-}"; shift 2 ;;
    --iperf-timeout-extra) IPERF_TIMEOUT_EXTRA="${2:-}"; shift 2 ;;
    --namespace) VALIDATION_NAMESPACE="${2:-}"; shift 2 ;;
    --upf-pod-regex) UPF_POD_REGEX="${2:-}"; shift 2 ;;
    --upf-iface) UPF_IFACE="${2:-}"; shift 2 ;;
    --kubeconfig) KUBECONFIG_REMOTE="${2:-}"; shift 2 ;;
    --delay-levels) DELAY_LEVELS_RAW="${2:-}"; shift 2 ;;
    --jitter-levels) JITTER_LEVELS_RAW="${2:-}"; shift 2 ;;
    --jitter-base-ms) JITTER_BASE_MS="${2:-}"; shift 2 ;;
    --loss-levels) LOSS_LEVELS_RAW="${2:-}"; shift 2 ;;
    --bandwidth-levels) BANDWIDTH_LEVELS_RAW="${2:-}"; shift 2 ;;
    --directions) DIRECTIONS_RAW="${2:-}"; shift 2 ;;
    --iperf-seconds) IPERF_SECONDS="${2:-}"; shift 2 ;;
    --ping-count) PING_COUNT="${2:-}"; shift 2 ;;
    --ping-interval) PING_INTERVAL="${2:-}"; shift 2 ;;
    --settle-seconds) SETTLE_SECONDS="${2:-}"; shift 2 ;;
    --tail-seconds) TAIL_SECONDS="${2:-}"; shift 2 ;;
    --step) STEP="${2:-}"; shift 2 ;;
    --discover-metrics) DISCOVER_METRICS=1; shift ;;
    --discover-metrics-regex) DISCOVER_REGEX="${2:-}"; shift 2 ;;
    --discover-metrics-limit) DISCOVER_LIMIT="${2:-}"; shift 2 ;;
    --all-prometheus-metrics) DISCOVER_METRICS=1; DISCOVER_REGEX='.*'; DISCOVER_LIMIT=5000; shift ;;
    --min-free-gb) MIN_FREE_GB="${2:-}"; shift 2 ;;
    --max-run-gb) MAX_RUN_GB="${2:-}"; shift 2 ;;
    --health-ping-count) HEALTH_PING_COUNT="${2:-}"; shift 2 ;;
    --health-ping-interval) HEALTH_PING_INTERVAL="${2:-}"; shift 2 ;;
    --no-level-health-gate) LEVEL_HEALTH_GATE=0; shift ;;
    --no-gzip-prometheus) GZIP_PROMETHEUS=0; shift ;;
    --skip-preflight-probes) PREFLIGHT_PROBES=0; shift ;;
    --no-iperf) RUN_IPERF=0; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "error: unknown argument: $1" >&2; usage >&2; exit 1 ;;
  esac
done

if [[ -z "$UE_HOST" ]]; then
  UE_HOST="${QHAT_SSH_TARGET##*@}"
fi

require_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo "error: missing required file: $path" >&2
    exit 1
  fi
}

require_file "$ROOT_DIR/scripts/validation/prometheus_range_export.py"
require_file "$ROOT_DIR/scripts/validation/paper_prometheus_queries.json"

split_words() {
  local raw="$1"
  raw="${raw//,/ }"
  # shellcheck disable=SC2206
  echo ${raw}
}

log() {
  printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"
}

gb_to_kib() {
  awk -v gb="$1" 'BEGIN { printf "%.0f", gb * 1024 * 1024 }'
}

check_disk_budget() {
  local stage="$1"
  local avail_kib min_kib used_kib max_kib

  avail_kib="$(df -Pk "$OUT_ROOT" | awk 'NR == 2 {print $4}')"
  used_kib="$(du -sk "$RUN_DIR" 2>/dev/null | awk '{print $1 + 0}')"
  log "disk[$stage]: free=$(awk -v k="$avail_kib" 'BEGIN {printf "%.2f GiB", k/1024/1024}') run=$(awk -v k="$used_kib" 'BEGIN {printf "%.2f GiB", k/1024/1024}')"

  if [[ "$MIN_FREE_GB" != "0" ]]; then
    min_kib="$(gb_to_kib "$MIN_FREE_GB")"
    if awk -v a="$avail_kib" -v m="$min_kib" 'BEGIN {exit !(a < m)}'; then
      echo "ERROR: free disk below --min-free-gb=$MIN_FREE_GB during $stage" >&2
      exit 122
    fi
  fi

  if [[ "$MAX_RUN_GB" != "0" ]]; then
    max_kib="$(gb_to_kib "$MAX_RUN_GB")"
    if awk -v u="$used_kib" -v m="$max_kib" 'BEGIN {exit !(u > m)}'; then
      echo "ERROR: campaign directory exceeds --max-run-gb=$MAX_RUN_GB during $stage" >&2
      exit 122
    fi
  fi
}

safe_name() {
  printf '%s' "$1" | tr '.' '_' | tr '/' '_'
}

validate_directions() {
  local direction
  for direction in $(split_words "$DIRECTIONS_RAW"); do
    case "$direction" in
      dl|ul) ;;
      *)
        echo "error: unknown direction: $direction (expected dl or ul)" >&2
        exit 2
        ;;
    esac
  done
}

validate_directions

case "$UL_MATCH_MODE" in
  ue|all) ;;
  *)
    echo "error: unknown --ul-match-mode: $UL_MATCH_MODE (expected ue or all)" >&2
    exit 2
    ;;
esac

timestamp="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="$OUT_ROOT/validated_network_campaign_$timestamp"
mkdir -p "$RUN_DIR"

cat >"$RUN_DIR/campaign_metadata.env" <<EOF
PROMETHEUS_URL=$PROM_URL
SCENARIOS=$SCENARIOS_RAW
K8S_SSH_TARGET=$K8S_SSH_TARGET
IPERF_SSH_TARGET=$IPERF_SSH_TARGET
QHAT_SSH_TARGET=$QHAT_SSH_TARGET
UE_HOST=$UE_HOST
UE_IP=$UE_IP
UE_GATEWAY=$UE_GATEWAY
UE_IFACE=$UE_IFACE
TARGET_IP=$TARGET_IP
UL_MATCH_MODE=$UL_MATCH_MODE
IPERF_PORT=$IPERF_PORT
IPERF_PARALLEL=$IPERF_PARALLEL
IPERF_TIMEOUT_EXTRA=$IPERF_TIMEOUT_EXTRA
VALIDATION_NAMESPACE=$VALIDATION_NAMESPACE
UPF_POD_REGEX=$UPF_POD_REGEX
UPF_IFACE=$UPF_IFACE
KUBECONFIG_REMOTE=$KUBECONFIG_REMOTE
DELAY_LEVELS=$DELAY_LEVELS_RAW
JITTER_LEVELS=$JITTER_LEVELS_RAW
JITTER_BASE_MS=$JITTER_BASE_MS
LOSS_LEVELS=$LOSS_LEVELS_RAW
BANDWIDTH_LEVELS=$BANDWIDTH_LEVELS_RAW
DIRECTIONS=$DIRECTIONS_RAW
IPERF_SECONDS=$IPERF_SECONDS
PING_COUNT=$PING_COUNT
PING_INTERVAL=$PING_INTERVAL
SETTLE_SECONDS=$SETTLE_SECONDS
TAIL_SECONDS=$TAIL_SECONDS
STEP=$STEP
RUN_IPERF=$RUN_IPERF
DISCOVER_METRICS=$DISCOVER_METRICS
GZIP_PROMETHEUS=$GZIP_PROMETHEUS
PREFLIGHT_PROBES=$PREFLIGHT_PROBES
MIN_FREE_GB=$MIN_FREE_GB
MAX_RUN_GB=$MAX_RUN_GB
HEALTH_PING_COUNT=$HEALTH_PING_COUNT
HEALTH_PING_INTERVAL=$HEALTH_PING_INTERVAL
LEVEL_HEALTH_GATE=$LEVEL_HEALTH_GATE
EOF

remote_tc() {
  local mode="$1"
  local value="${2:-0}"
  local direction="${3:-dl}"
  ssh "$K8S_SSH_TARGET" \
    "KUBECONFIG_REMOTE='$KUBECONFIG_REMOTE' VALIDATION_NAMESPACE='$VALIDATION_NAMESPACE' UPF_POD_REGEX='$UPF_POD_REGEX' UPF_IFACE='$UPF_IFACE' TC_MODE='$mode' TC_VALUE='$value' TC_DIRECTION='$direction' UE_IP='$UE_IP' UL_MATCH_MODE='$UL_MATCH_MODE' JITTER_BASE_MS='$JITTER_BASE_MS' TBF_BURST='$TBF_BURST' TBF_LATENCY='$TBF_LATENCY' bash -s" <<'REMOTE'
set -euo pipefail

export KUBECONFIG="$KUBECONFIG_REMOTE"
POD="$(
  kubectl -n "$VALIDATION_NAMESPACE" get pods \
    --field-selector=status.phase=Running \
    --no-headers \
  | awk -v re="$UPF_POD_REGEX" '$1 ~ re {print $1; exit}'
)"

if [ -z "$POD" ]; then
  echo "ERROR: no running pod matched regex $UPF_POD_REGEX in namespace $VALIDATION_NAMESPACE" >&2
  exit 2
fi

echo "pod=$POD namespace=$VALIDATION_NAMESPACE iface=$UPF_IFACE direction=$TC_DIRECTION mode=$TC_MODE value=$TC_VALUE"

kubectl -n "$VALIDATION_NAMESPACE" exec -i "$POD" -- sh -s -- "$UPF_IFACE" "$TC_MODE" "$TC_VALUE" "$TC_DIRECTION" "$UE_IP" "$UL_MATCH_MODE" "$JITTER_BASE_MS" "$TBF_BURST" "$TBF_LATENCY" <<'PODSH'
iface="$1"
mode="$2"
value="$3"
direction="$4"
ue_ip="$5"
ul_match_mode="$6"
jitter_base_ms="$7"
tbf_burst="$8"
tbf_latency="$9"
ifb="ifb5g${direction}"

show_all() {
  echo "=== qdisc $iface ==="
  tc -s qdisc show dev "$iface" || true
  echo "=== ingress filters $iface ==="
  tc -s filter show dev "$iface" ingress || true
  if ip link show "$ifb" >/dev/null 2>&1; then
    echo "=== qdisc $ifb ==="
    tc -s qdisc show dev "$ifb" || true
  fi
}

cleanup_all() {
  tc qdisc del dev "$iface" root 2>/dev/null || true
  tc qdisc del dev "$iface" clsact 2>/dev/null || true
  tc qdisc del dev "$ifb" root 2>/dev/null || true
  ip link del "$ifb" 2>/dev/null || true
}

apply_qdisc() {
  target="$1"
  tc qdisc del dev "$target" root 2>/dev/null || true
  case "$mode" in
    delay)
      if [ "$value" != "0" ]; then
        tc qdisc add dev "$target" root netem delay "${value}ms"
      fi
      ;;
    jitter)
      if [ "$value" = "0" ]; then
        tc qdisc add dev "$target" root netem delay "${jitter_base_ms}ms"
      else
        tc qdisc add dev "$target" root netem delay "${jitter_base_ms}ms" "${value}ms" distribution normal 2>/dev/null \
          || tc qdisc add dev "$target" root netem delay "${jitter_base_ms}ms" "${value}ms"
      fi
      ;;
    loss)
      if [ "$value" != "0" ]; then
        tc qdisc add dev "$target" root netem loss "${value}%"
      fi
      ;;
    bandwidth)
      if [ "$value" != "0" ]; then
        tc qdisc add dev "$target" root tbf rate "${value}mbit" burst "$tbf_burst" latency "$tbf_latency"
      fi
      ;;
    *)
      echo "ERROR: unknown tc mode: $mode" >&2
      exit 2
      ;;
  esac
}

case "$mode" in
  cleanup)
    show_all
    cleanup_all
    ;;
  show)
    show_all
    ;;
  delay|jitter|loss|bandwidth)
    cleanup_all
    if [ "$direction" = "dl" ]; then
      apply_qdisc "$iface"
    elif [ "$direction" = "ul" ]; then
      if [ "$mode" = "jitter" ] || [ "$value" != "0" ]; then
        ip link add "$ifb" type ifb
        ip link set "$ifb" up
        apply_qdisc "$ifb"
        tc qdisc add dev "$iface" clsact
        if [ "$ul_match_mode" = "all" ]; then
          tc filter add dev "$iface" ingress protocol ip prio 10 matchall \
            action mirred egress redirect dev "$ifb" 2>/dev/null \
            || tc filter add dev "$iface" ingress protocol ip prio 10 u32 \
              match ip src 0.0.0.0/0 \
              action mirred egress redirect dev "$ifb"
        else
          tc filter add dev "$iface" ingress protocol ip prio 10 u32 \
            match ip src "$ue_ip/32" \
            action mirred egress redirect dev "$ifb"
        fi
      fi
    else
      echo "ERROR: unknown tc direction: $direction" >&2
      exit 2
    fi
    show_all
    ;;
  *)
    echo "ERROR: unknown tc mode: $mode" >&2
    exit 2
    ;;
esac
PODSH
REMOTE
}

cleanup_remote() {
  remote_tc cleanup 0 dl >/dev/null 2>&1 || true
  remote_tc cleanup 0 ul >/dev/null 2>&1 || true
}
trap cleanup_remote EXIT INT TERM

ensure_ue_route() {
  log "ensure UE route to $TARGET_IP through $UE_IFACE"
  ssh "$QHAT_SSH_TARGET" \
    "ip route replace '$TARGET_IP/32' via '$UE_GATEWAY' dev '$UE_IFACE' src '$UE_IP'; ip route get '$TARGET_IP'"
}

quick_ping_check() {
  ssh "$QHAT_SSH_TARGET" \
    "ping -I '$UE_IP' -c '$HEALTH_PING_COUNT' -i '$HEALTH_PING_INTERVAL' '$TARGET_IP'"
}

ensure_iperf_server() {
  if [[ "$RUN_IPERF" -eq 0 ]]; then
    return
  fi
  log "ensure iperf3 server on $IPERF_SSH_TARGET:$IPERF_PORT"
  ssh "$IPERF_SSH_TARGET" \
    "pkill -x iperf3 2>/dev/null || true; nohup iperf3 -s -B 0.0.0.0 -p '$IPERF_PORT' > /tmp/iperf3-server-${IPERF_PORT}.log 2>&1 & sleep 1; ss -lntp | grep ':$IPERF_PORT' || { cat /tmp/iperf3-server-${IPERF_PORT}.log >&2 || true; exit 2; }"
}

preflight_probes() {
  if [[ "$PREFLIGHT_PROBES" -eq 0 ]]; then
    return
  fi

  log "preflight: UE ping through user plane"
  ssh "$QHAT_SSH_TARGET" \
    "ping -I '$UE_IP' -c 5 -i 0.2 '$TARGET_IP'"

  if [[ "$RUN_IPERF" -eq 1 ]]; then
    local direction
    for direction in $(split_words "$DIRECTIONS_RAW"); do
      if [[ "$direction" == "dl" ]]; then
        log "preflight: DL iperf3 through user plane"
        timeout -k 5s 20s ssh "$QHAT_SSH_TARGET" \
          "iperf3 -R -B '$UE_IP' -c '$TARGET_IP' -p '$IPERF_PORT' -t 5 -P '$IPERF_PARALLEL'"
      else
        log "preflight: UL iperf3 through user plane"
        timeout -k 5s 20s ssh "$QHAT_SSH_TARGET" \
          "iperf3 -B '$UE_IP' -c '$TARGET_IP' -p '$IPERF_PORT' -t 5 -P '$IPERF_PARALLEL'"
      fi
    done
  fi
}

export_prometheus() {
  local start_epoch="$1"
  local end_epoch="$2"
  local out_csv="$3"
  local out_summary="$4"
  local out_path="$out_csv"

  if [[ "$GZIP_PROMETHEUS" -eq 1 && "$out_path" != *.gz ]]; then
    out_path="${out_path}.gz"
  fi

  local cmd=(
    python3 "$ROOT_DIR/scripts/validation/prometheus_range_export.py"
    --prometheus-url "$PROM_URL" \
    --start "$start_epoch" \
    --end "$end_epoch" \
    --step "$STEP" \
    --queries-json "$ROOT_DIR/scripts/validation/paper_prometheus_queries.json" \
    --out "$out_path" \
    --summary-json "$out_summary"
  )
  if [[ "$DISCOVER_METRICS" -eq 1 ]]; then
    cmd+=(--discover-metrics-regex "$DISCOVER_REGEX" --discover-metrics-limit "$DISCOVER_LIMIT")
  else
    cmd+=(--no-discover-metrics)
  fi

  "${cmd[@]}"
}

write_level_metadata() {
  local level_dir="$1"
  local scenario="$2"
  local label="$3"
  local level_name="$4"
  local tc_mode="$5"
  local tc_value="$6"
  local direction="$7"
  local iperf_reverse=0
  [[ "$direction" == "dl" ]] && iperf_reverse=1
  cat >"$level_dir/metadata.env" <<EOF
scenario=$scenario
label=$label
level_name=$level_name
tc_mode=$tc_mode
tc_value=$tc_value
upf_iface=$UPF_IFACE
traffic_direction=$direction
ul_match_mode=$UL_MATCH_MODE
iperf_reverse=$iperf_reverse
iperf_parallel=$IPERF_PARALLEL
iperf_seconds=$IPERF_SECONDS
iperf_timeout_extra=$IPERF_TIMEOUT_EXTRA
target_ip=$TARGET_IP
ue_host=$UE_HOST
qhat_ssh_target=$QHAT_SSH_TARGET
ue_ip=$UE_IP
EOF
}

mark_level_health() {
  local level_dir="$1"
  local status="$2"
  local valid="$3"
  local reason="$4"
  cat >>"$level_dir/metadata.env" <<EOF
level_status=$status
valid_for_dataset=$valid
invalid_reason=$reason
EOF
}

ping_loss_pct() {
  local ping_file="$1"
  grep -Eo '[0-9.]+% packet loss' "$ping_file" 2>/dev/null \
    | tail -1 \
    | sed 's/% packet loss//' \
    || true
}

ul_ifb_packets() {
  local tc_file="$1"
  local direction="$2"
  awk -v section="=== qdisc ifb5g${direction} ===" '
    $0 == section { in_section = 1; next }
    in_section && / Sent / { print $4; exit }
  ' "$tc_file" 2>/dev/null || true
}

validate_level_health() {
  local level_dir="$1"
  local tc_mode="$2"
  local tc_value="$3"
  local direction="$4"
  local loss_pct packets

  loss_pct="$(ping_loss_pct "$level_dir/ping.txt")"
  if [[ -z "$loss_pct" ]]; then
    echo "missing_ping_summary"
    return 1
  fi
  if awk -v loss="$loss_pct" 'BEGIN { exit !(loss >= 95.0) }'; then
    echo "ping_packet_loss_${loss_pct}pct"
    return 1
  fi

  if [[ "$direction" == "ul" && ( "$tc_mode" == "jitter" || "$tc_value" != "0" ) ]]; then
    packets="$(ul_ifb_packets "$level_dir/tc_before_cleanup.txt" "$direction")"
    if [[ -z "$packets" || "$packets" == "0" ]]; then
      echo "ul_ifb_zero_packets"
      return 1
    fi
  fi

  echo "ok"
  return 0
}

run_level() {
  local scenario="$1"
  local label="$2"
  local level_name="$3"
  local tc_mode="$4"
  local tc_value="$5"
  local direction="$6"
  local scenario_dir="$RUN_DIR/$scenario"
  local level_dir_name="${level_name}__${direction}"
  local level_dir="$scenario_dir/$level_dir_name"
  mkdir -p "$level_dir"
  check_disk_budget "$scenario/$level_dir_name before"

  if [[ "$LEVEL_HEALTH_GATE" -eq 1 ]]; then
    log "$scenario/$level_dir_name: clean-path health check before injection"
    cleanup_remote
    write_level_metadata "$level_dir" "$scenario" "$label" "$level_dir_name" "$tc_mode" "$tc_value" "$direction"
    set +e
    ensure_ue_route >"$level_dir/precheck_ue_route.txt" 2>&1
    local route_rc="$?"
    quick_ping_check >"$level_dir/precheck_ping.txt" 2>&1
    local ping_rc="$?"
    set -e
    if [[ "$route_rc" -ne 0 || "$ping_rc" -ne 0 ]]; then
      mark_level_health "$level_dir" "precheck_failed" "0" "clean_path_unreachable"
      log "$scenario/$level_dir_name: SKIP, clean path is unreachable before anomaly"
      return 0
    fi
  fi

  log "$scenario/$level_dir_name: apply $tc_mode $tc_value direction=$direction"
  remote_tc "$tc_mode" "$tc_value" "$direction" | tee "$level_dir/tc_after_apply.txt"
  write_level_metadata "$level_dir" "$scenario" "$label" "$level_dir_name" "$tc_mode" "$tc_value" "$direction"

  sleep "$SETTLE_SECONDS"
  local start_epoch
  start_epoch="$(date +%s)"
  echo "$start_epoch" >"$level_dir/start_epoch.txt"

  if [[ "$RUN_IPERF" -eq 1 ]]; then
    local iperf_timeout
    iperf_timeout="$((IPERF_SECONDS + IPERF_TIMEOUT_EXTRA))"
    log "$scenario/$level_dir_name: ${direction^^} iperf3 for ${IPERF_SECONDS}s (timeout ${iperf_timeout}s)"
    set +e
    if [[ "$direction" == "dl" ]]; then
      timeout -k 10s "${iperf_timeout}s" ssh "$QHAT_SSH_TARGET" \
        "if command -v timeout >/dev/null 2>&1; then timeout '${iperf_timeout}s' iperf3 -R -B '$UE_IP' -c '$TARGET_IP' -p '$IPERF_PORT' -t '$IPERF_SECONDS' -P '$IPERF_PARALLEL' --json; else iperf3 -R -B '$UE_IP' -c '$TARGET_IP' -p '$IPERF_PORT' -t '$IPERF_SECONDS' -P '$IPERF_PARALLEL' --json; fi" \
        >"$level_dir/iperf_dl.json" 2>"$level_dir/iperf_dl.stderr"
      echo "$?" >"$level_dir/iperf_dl_rc.txt"
    else
      timeout -k 10s "${iperf_timeout}s" ssh "$QHAT_SSH_TARGET" \
        "if command -v timeout >/dev/null 2>&1; then timeout '${iperf_timeout}s' iperf3 -B '$UE_IP' -c '$TARGET_IP' -p '$IPERF_PORT' -t '$IPERF_SECONDS' -P '$IPERF_PARALLEL' --json; else iperf3 -B '$UE_IP' -c '$TARGET_IP' -p '$IPERF_PORT' -t '$IPERF_SECONDS' -P '$IPERF_PARALLEL' --json; fi" \
        >"$level_dir/iperf_ul.json" 2>"$level_dir/iperf_ul.stderr"
      echo "$?" >"$level_dir/iperf_ul_rc.txt"
    fi
    set -e
  fi

  log "$scenario/$level_dir_name: ping evidence"
  set +e
  ssh "$QHAT_SSH_TARGET" "ping -I '$UE_IP' -c '$PING_COUNT' -i '$PING_INTERVAL' '$TARGET_IP'" 2>&1 \
    | tee "$level_dir/ping.txt"
  echo "${PIPESTATUS[0]}" >"$level_dir/ping_rc.txt"
  set -e

  sleep "$TAIL_SECONDS"
  local end_epoch
  end_epoch="$(date +%s)"
  echo "$end_epoch" >"$level_dir/end_epoch.txt"

  log "$scenario/$level_dir_name: tc evidence before cleanup"
  remote_tc show "$tc_value" "$direction" | tee "$level_dir/tc_before_cleanup.txt"

  log "$scenario/$level_dir_name: cleanup tc"
  remote_tc cleanup 0 "$direction" | tee "$level_dir/tc_cleanup.txt"

  if [[ "$LEVEL_HEALTH_GATE" -eq 1 ]]; then
    local health_reason
    set +e
    health_reason="$(validate_level_health "$level_dir" "$tc_mode" "$tc_value" "$direction")"
    local health_rc="$?"
    set -e
    if [[ "$health_rc" -ne 0 ]]; then
      mark_level_health "$level_dir" "invalid" "0" "$health_reason"
      log "$scenario/$level_dir_name: SKIP Prometheus export, invalid level: $health_reason"
      return 0
    fi
    mark_level_health "$level_dir" "ok" "1" ""
  fi

  log "$scenario/$level_dir_name: export Prometheus"
  export_prometheus \
    "$start_epoch" \
    "$end_epoch" \
    "$level_dir/prometheus_timeseries.csv" \
    "$level_dir/prometheus_export_summary.json"
  check_disk_budget "$scenario/$level_dir_name after"
}

run_manual_scenario() {
  local scenario="$1"
  local value label level_name safe direction
  case "$scenario" in
    controlled_delay)
      for value in $(split_words "$DELAY_LEVELS_RAW"); do
        safe="$(printf '%03d' "$value")"
        label="clean_traffic"
        [[ "$value" != "0" ]] && label="controlled_delay"
        for direction in $(split_words "$DIRECTIONS_RAW"); do
          run_level "$scenario" "$label" "delay_${safe}ms" delay "$value" "$direction"
        done
      done
      ;;
    controlled_jitter)
      for value in $(split_words "$JITTER_LEVELS_RAW"); do
        safe="$(printf '%03d' "$value")"
        label="controlled_jitter"
        [[ "$value" = "0" ]] && label="controlled_delay"
        for direction in $(split_words "$DIRECTIONS_RAW"); do
          run_level "$scenario" "$label" "jitter_${safe}ms_base_${JITTER_BASE_MS}ms" jitter "$value" "$direction"
        done
      done
      ;;
    tunnel_packet_loss)
      for value in $(split_words "$LOSS_LEVELS_RAW"); do
        safe="$(safe_name "$value")"
        label="clean_traffic"
        [[ "$value" != "0" ]] && label="tunnel_packet_loss"
        for direction in $(split_words "$DIRECTIONS_RAW"); do
          run_level "$scenario" "$label" "loss_${safe}pct" loss "$value" "$direction"
        done
      done
      ;;
    tunnel_bandwidth_limit)
      for value in $(split_words "$BANDWIDTH_LEVELS_RAW"); do
        safe="$(safe_name "$value")"
        label="clean_traffic"
        [[ "$value" != "0" ]] && label="tunnel_bandwidth_limit"
        for direction in $(split_words "$DIRECTIONS_RAW"); do
          run_level "$scenario" "$label" "rate_${safe}mbit" bandwidth "$value" "$direction"
        done
      done
      ;;
    *)
      return 1
      ;;
  esac
}

run_playbook_scenario() {
  local scenario="$1"
  local playbook_scenario=""
  case "$scenario" in
    upf_stress) playbook_scenario="24_upf_stress_levels" ;;
    radio_interference) playbook_scenario="26_interference_gain_levels" ;;
    far_ue_poor_radio) playbook_scenario="34_far_ue_radio_levels" ;;
    *) return 1 ;;
  esac

  log "$scenario: launch playbook scenario $playbook_scenario"
  ansible-playbook -i "$ROOT_DIR/inventory/default/hosts.ini" "$ROOT_DIR/playbooks/run_tcp_paper_scenarios.yml" \
    -e "paper_scenario_names=$playbook_scenario" \
    -e core="$VALIDATION_NAMESPACE" \
    -e "paper_prometheus_url=$PROM_URL" \
    -e "paper_kubeconfig=$KUBECONFIG_REMOTE" \
    -e "paper_pod_log_control_host=${K8S_SSH_TARGET#*@}" \
    -e ebpf_probe_preflight_enabled=false

  local latest
  latest="$(ls -td "$ROOT_DIR"/results/tcp-paper-* 2>/dev/null | head -1 || true)"
  if [[ -n "$latest" ]]; then
    printf '%s\t%s\n' "$scenario" "$latest" >>"$RUN_DIR/external_runs.tsv"
  fi
}

log "campaign directory: $RUN_DIR"
check_disk_budget "campaign start"
cleanup_remote
remote_tc show 0 dl | tee "$RUN_DIR/preflight_tc_cleanup_dl.txt"
remote_tc show 0 ul | tee "$RUN_DIR/preflight_tc_cleanup_ul.txt"
ensure_ue_route | tee "$RUN_DIR/preflight_ue_route.txt"
ensure_iperf_server | tee "$RUN_DIR/preflight_iperf_server.txt"
preflight_probes | tee "$RUN_DIR/preflight_probes.txt"

for scenario in $(split_words "$SCENARIOS_RAW"); do
  log "scenario: $scenario"
  case "$scenario" in
    controlled_delay|controlled_jitter|tunnel_packet_loss|tunnel_bandwidth_limit)
      run_manual_scenario "$scenario"
      ;;
    upf_stress|radio_interference|far_ue_poor_radio)
      run_playbook_scenario "$scenario"
      ;;
    *)
      echo "error: unknown scenario: $scenario" >&2
      exit 2
      ;;
  esac
done

log "done"
echo "Campaign directory: $RUN_DIR"
echo "Build dataset with:"
echo "  python3 scripts/ml/build_manual_network_dataset.py --manual-root results/manual-network --out results/manual-network/dataset/window_features_manual_network.csv --window-seconds 30"
