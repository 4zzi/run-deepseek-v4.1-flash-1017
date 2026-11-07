#!/usr/bin/env bash
set -uo pipefail
export LC_ALL=C.UTF-8

SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
STRATA="${STRATA:-${HOME}/.local/bin/strata}"
MODEL_PATH="${MODEL_PATH:-/mnt/ai/AI_MODELS/DeepSeek-V4.1-Flash/IQ2_XXS}"
PORT="${PORT:-8080}"
URL="http://127.0.0.1:${PORT}"
BEST="${BEST:-${SCRIPT_DIR}/ds-best}"
FLAGS="${HOME}/.config/ds-flags"
STATE="${HOME}/.local/state/strata"
TLOG="${STATE}/ds-tune.log"
RESULTS="${STATE}/ds-tune-results.txt"
PEAKF="${STATE}/ds-tune-peak.txt"
MIN_TPS="${MIN_TPS:-0}"
RESERVE_MIB="${RESERVE_MIB:-1800}"
LOAD_TIMEOUT="${LOAD_TIMEOUT:-1800}"
POLL="${POLL:-1}"
RUNAWAY_LIMIT=4096
EXPERTS_MODE="${EXPERTS_MODE:-native}"   # native = fastest + what the model was trained for, max = search above native
SSD_CACHE_FORCED="${SSD_CACHE:-}"        # set SSD_CACHE=N to skip the cache sweep
SSD_CACHE="${SSD_CACHE:-6}"
CACHE_STEPS=(4 6 8 10 12 14 16)
POOL_FORCED="${POOL_CORES:-}"            # none | phys | all | 0,1,2,... skips the cpu pinning sweep
POOL_NAME=none; POOL_FLAG=""; POOL_ALL=""; POOL_PHYS=""
PW=68
START_TS=$(date +%s)
mkdir -p "${HOME}/.config" "$STATE"
touch "$TLOG" "$RESULTS"

TTY=0; [ -t 2 ] && TTY=1
if [ "$TTY" -eq 1 ] && [ -z "${NO_COLOR:-}" ]; then
  R=$'\e[0m'; B=$'\e[1m'
  case "${COLORTERM:-}" in
    *truecolor*|*24bit*)
      ACC=$'\e[38;2;122;162;247m'; GRN=$'\e[38;2;126;231;135m'; RED=$'\e[38;2;255;123;114m'
      YLW=$'\e[38;2;227;179;65m'; D=$'\e[38;2;125;133;145m'; BRD=$'\e[38;2;72;79;88m' ;;
    *)
      ACC=$'\e[36m'; GRN=$'\e[32m'; RED=$'\e[31m'; YLW=$'\e[33m'; D=$'\e[2m'; BRD=$'\e[2m' ;;
  esac
else
  R='' B='' ACC='' GRN='' RED='' YLW='' D='' BRD=''
fi
COLS=$(tput cols 2>/dev/null || echo 100)
HAS_NV=0; command -v nvidia-smi >/dev/null 2>&1 && HAS_NV=1

elapsed() { local s=$(( $(date +%s) - START_TS )); printf '%d:%02d' $((s / 60)) $((s % 60)); }
fmt_t() { printf '%d:%02d' $(( $1 / 60 )) $(( $1 % 60 )); }
gib() { awk "BEGIN{printf \"%.1f\", $1/1024}"; }
rep() { local i s=''; for ((i = 0; i < $2; i++)); do s+=$1; done; printf '%s' "$s"; }
padr() { local s=${1:0:$2}; printf '%s%s' "$s" "$(rep ' ' $(( $2 - ${#s} )))"; }

DISK_DEV=""; DISK_FILE=""; DS_PREV=""; DT_PREV=""; DISK_MBS=0; DISK_BUF=0
BENCHD="$STATE/ds-bench-disk.txt"
fmt_rate() { awk -v m="${1:-0}" 'BEGIN{ if (m >= 1000) printf "%.1f GB/s", m/1024; else printf "%d MB/s", m }'; }
disk_sectors() { awk -v d="$DISK_DEV" '$3==d{print $6; exit}' /proc/diskstats 2>/dev/null; }
disk_init() {
  local f src c mm name cands=()
  f=$(ls "$MODEL_PATH"/*.gguf 2>/dev/null | head -1)
  [ -n "$f" ] || return 0
  f=$(readlink -f "$f" 2>/dev/null || echo "$f"); DISK_FILE=$f
  src=$(findmnt -n -o SOURCE -T "$f" 2>/dev/null | sed 's/\[.*//')
  if [ -n "$src" ]; then
    src=$(readlink -f "$src" 2>/dev/null || echo "$src")
    cands+=("$(basename "$src")" "$(lsblk -no pkname "$src" 2>/dev/null | head -1)")
  fi
  mm=$(stat -c '%Hd:%Ld' "$f" 2>/dev/null)
  name=$(awk -v M="${mm%%:*}" -v N="${mm##*:}" '$1==M && $2==N{print $3; exit}' /proc/diskstats 2>/dev/null)
  cands+=("$name")
  for c in "${cands[@]}"; do
    [ -n "$c" ] || continue
    if awk -v d="$c" '$3==d{f=1} END{exit !f}' /proc/diskstats 2>/dev/null; then DISK_DEV=$c; break; fi
  done
  [ -n "$DISK_DEV" ] && { DS_PREV=$(disk_sectors); DT_PREV=$(date +%s.%N); }
  return 0
}
# read test on a model shard: 4 parallel direct reads (deep queue, page cache bypassed)
disk_speed() {
  [ -n "$DISK_FILE" ] || return 0
  DISK_BUF=0
  local tmp i t0 t1 tot=0 b pids=()
  tmp=$(mktemp -d 2>/dev/null) || return 0
  t0=$(date +%s.%N)
  for i in 0 1 2 3; do
    timeout 30 dd if="$DISK_FILE" of=/dev/null bs=8M count=32 skip=$((i * 32)) iflag=direct 2>"$tmp/$i" &
    pids+=($!)
  done
  wait "${pids[@]}" 2>/dev/null
  t1=$(date +%s.%N)
  for i in 0 1 2 3; do
    b=$(grep -m1 ' bytes ' "$tmp/$i" 2>/dev/null | awk '{print $1}')
    tot=$(( tot + ${b:-0} ))
  done
  rm -rf "$tmp"
  if [ "$tot" -gt 0 ]; then
    fmt_rate "$(awk -v b="$tot" -v a="$t0" -v z="$t1" 'BEGIN{printf "%d", b/1048576/(z-a)}')"
    return 0
  fi
  # filesystem refused direct reads: plain buffered read, may be inflated by RAM cache
  DISK_BUF=1
  local o; o=$(timeout 30 dd if="$DISK_FILE" of=/dev/null bs=4M count=256 skip=64 2>&1 | tail -n1)
  case "$o" in *B/s*) printf '%s' "${o##*, }" ;; esac
}

say()  { printf '   %s\n' "$*" >&2; }
ok()   { say "${GRN}✓${R}  $*"; }
warn() { say "${YLW}▲${R}  $*"; }
bad()  { say "${RED}✗${R}  $*"; }
info() { say "${ACC}◆${R}  $*"; }
section() {
  printf '\n  %s▍%s %s%s%s' "$ACC" "$R" "$B" "$1" "$R" >&2
  [ -n "${2:-}" ] && printf '  %s%s%s' "$D" "$2" "$R" >&2
  printf '\n\n' >&2
}
rule() { printf '  %s%s%s\n' "$BRD" "$(rep ─ $((PW + 2)))" "$R" >&2; }

panel_top() { printf '  %s%s%s\n' "$B" "$1" "$R" >&2; }
panel_bot() { :; }
prow() {
  local st=$1 label=$2 val=$3 sty=${4:-} g c
  case $st in ok) g='✓' c=$GRN ;; warn) g='▲' c=$YLW ;; bad) g='✗' c=$RED ;; *) g='◆' c=$ACC ;; esac
  printf '   %s%s%s  %s%s%s  %s\n' "$c" "$g" "$R" "$D" "$(padr "$label" 8)" "$R" "${sty:-$val}" >&2
}
tiles() {
  printf '   %s%s%s%s%s%s\n' "$B" "$(padr "$1" 14)" "$(padr "$2" 14)" "$(padr "$3" 14)" "$(padr "$4" 14)" "$R" >&2
  printf '   %s%s%s%s%s%s\n' "$D" "$(padr context 14)" "$(padr experts 14)" "$(padr vision 14)" "$(padr speed 14)" "$R" >&2
}

banner() {
  printf '\n' >&2
  printf '  %s▍%s %sstrata%s  %s·  DeepSeek auto-max launcher%s\n\n' "$ACC" "$R" "$B" "$R" "$D" "$R" >&2
  prow info model "$(basename "$(dirname "$MODEL_PATH")")/$(basename "$MODEL_PATH")"
  prow info profile "$BEST"
  prow info api "${URL}/v1"
  panel_bot
}

bar() {
  local f=$1 t=$2 w=${3:-10} n c
  [ "${t:-0}" -gt 0 ] 2>/dev/null || { printf '%s%s%s' "$D" "$(rep ▱ "$w")" "$R"; return; }
  n=$(( f * w / t )); [ "$n" -gt "$w" ] && n=$w
  c=$GRN; [ $(( f * 100 / t )) -ge 70 ] && c=$YLW; [ $(( f * 100 / t )) -ge 90 ] && c=$RED
  printf '%s%s%s%s%s' "$c" "$(rep ▰ "$n")" "$D" "$(rep ▱ $((w - n)))" "$R"
}

stats() {
  RAM_FREE=$(awk '/MemAvailable/{printf "%.1f", $2/1048576}' /proc/meminfo)
  SWAP=$(awk '/SwapTotal/{t=$2} /SwapFree/{f=$2} END{printf "%.1f", (t-f)/1048576}' /proc/meminfo)
  LOAD=$(cut -d' ' -f1 /proc/loadavg)
  VU=0 VT=0 GU=0 GT=0
  if [ "$HAS_NV" -eq 1 ]; then
    read -r VU VT GU GT < <(nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ',') || true
  fi
  VU=${VU:-0} VT=${VT:-0} GU=${GU:-0} GT=${GT:-0}
  if [ -n "$DISK_DEV" ]; then
    local s t r; s=$(disk_sectors); t=$(date +%s.%N)
    if [ -n "$s" ] && [ -n "$DS_PREV" ]; then
      r=$(awk -v s="$s" -v p="$DS_PREV" -v t="$t" -v q="$DT_PREV" 'BEGIN{d=t-q; if (d<0.3) {print "x"; exit} printf "%d", (s-p)*512/1048576/d}')
      [ "$r" != x ] && [ -n "$r" ] && { DISK_MBS=$r; DS_PREV=$s; DT_PREV=$t; }
    fi
  fi
}

preflight() {
  section "System check"
  local hard=0
  disk_init
  if [ -x "$STRATA" ]; then prow ok strata "$STRATA"; else prow bad strata "not found at $STRATA"; hard=1; fi
  if [ -d "$MODEL_PATH" ]; then
    local n=0 broken=0 f size
    for f in "$MODEL_PATH"/*.gguf; do
      [ -e "$f" ] || [ -L "$f" ] || continue
      n=$((n + 1))
      [ -e "$f" ] || { broken=$((broken + 1)); prow bad shard "broken: $(basename "$f") → $(readlink "$f" 2>/dev/null)"; }
    done
    size=$(du -shL "$MODEL_PATH" 2>/dev/null | cut -f1)
    if [ "$n" -gt 0 ]; then prow ok model "${n} shards · ${size:-?} · $(basename "$MODEL_PATH")"; else prow bad model "no .gguf files found"; hard=1; fi
    [ "$broken" -gt 0 ] && { prow warn shard "a shard links to another drive, is it mounted?"; hard=1; }
    local free; free=$(df --output=avail -BG "$MODEL_PATH" 2>/dev/null | tail -1 | tr -dc '0-9')
    if [ -n "$free" ]; then
      if [ "$free" -ge 115 ]; then prow ok disk "${free} GB free"; else prow warn disk "${free} GB free (strata wanted ~111 GB earlier)"; fi
    fi
    local fs dev kind=""
    fs=$(findmnt -n -o FSTYPE -T "$MODEL_PATH" 2>/dev/null)
    dev=$(findmnt -n -o SOURCE -T "$MODEL_PATH" 2>/dev/null | sed 's|/dev/||; s|[0-9]*$||; s|p$||')
    if [ -n "$dev" ] && [ -r "/sys/block/$dev/queue/rotational" ]; then
      [ "$(cat "/sys/block/$dev/queue/rotational")" = 1 ] && kind="spinning disk" || kind="SSD/NVMe"
    fi
    case "$fs" in
      ext4|xfs|btrfs|f2fs) prow ok drive "${fs}${kind:+ · $kind}${dev:+ · $dev}" ;;
      ntfs3|ntfs) prow warn drive "${fs}, slower than ext4${kind:+ · $kind}" ;;
      *) prow info drive "${fs:-unknown}${kind:+ · $kind}" ;;
    esac
    [ "$kind" = "spinning disk" ] && prow warn drive "spinning disk, loads will be slow"
    local sp; sp=$(disk_speed)
    if [ -n "$sp" ]; then prow info read "${sp} read ${D}$([ "$DISK_BUF" = 1 ] && echo "(buffered, may read high)" || echo "(4 parallel, uncached)")${R}"
    else prow info read "couldn't measure read speed"; fi
    [ -n "$DISK_FILE" ] && [ -z "$DISK_DEV" ] && prow warn diskstat "couldn't match the drive in /proc/diskstats, live disk speed is off"
  else
    prow bad model "folder not found, is the drive mounted? check /mnt"; hard=1
  fi
  stats
  local mt; mt=$(awk '/MemTotal/{printf "%.1f", $2/1048576}' /proc/meminfo)
  prow info cpu "$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | sed 's/^ *//') · $(nproc) threads"
  prow info ram "${RAM_FREE} GB free of ${mt} GB · swap used ${SWAP} GB"
  if [ "$HAS_NV" -eq 1 ]; then
    prow info gpu "$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
    prow info vram "$(rep x 16)  $(gib "$VU") / $(gib "$VT") GB · ${GT}°C" "$(bar "$VU" "$VT" 16)  $(gib "$VU") / $(gib "$VT") GB ${D}· ${GT}°C${R}"
    local others; others=$(nvidia-smi --query-compute-apps=name,used_memory --format=csv,noheader 2>/dev/null | head -3 | tr '\n' '·')
    [ -n "$others" ] && prow warn gpu "also in use by: $others"
  elif command -v rocm-smi >/dev/null 2>&1; then
    prow info gpu "$(rocm-smi --showproductname 2>/dev/null | grep -i 'card series' | head -1 | sed 's/^ *//')"
  else
    prow warn gpu "none detected by nvidia-smi or rocm-smi"
  fi
  if python3 -c "import gguf" 2>/dev/null; then prow ok python "gguf installed"
  else prow warn python "gguf missing: pip install gguf --break-system-packages"; fi
  [ "$hard" -eq 0 ] || { printf '\n' >&2; bad "fix the items above, then run ./start.sh again"; exit 1; }
}

native_experts() {
  python3 - "$MODEL_PATH" <<'PY'
import sys, glob, json, os
p = sys.argv[1]
try:
    cfg = os.path.join(p, "config.json")
    if os.path.exists(cfg):
        v = json.load(open(cfg)).get("num_experts_per_tok")
        if v:
            print(int(v)); sys.exit()
except Exception:
    pass
try:
    import gguf
    fs = sorted(glob.glob(p + "/**/*.gguf", recursive=True))
    if fs:
        r = gguf.GGUFReader(fs[0])
        for k, f in r.fields.items():
            if k.endswith(".expert_used_count"):
                print(int(f.parts[f.data[0]][0])); sys.exit()
except Exception:
    pass
PY
}

set_flags() {
  printf '%s\n' "--reserve-mib ${RESERVE_MIB} --prefill-chunk 512 --ssd-cache ${SSD_CACHE:-6} --ssd-policy lfu --experts $1 ${POOL_FLAG:-} ${EXTRA_FLAGS:-}" > "$FLAGS"
}

launch() {
  "$STRATA" launch \
    --model-path "$MODEL_PATH" --model ista-das-coder --family deepseek41 \
    --context "$1" --vision "$2" --mtp on --parallel 1 --speed faster \
    --tune skip --access local --no-api-key --port "${3:-$PORT}" --no-tui --yes --restart
}

bench() {
  python3 - "$URL" "${1:-}" "${2:-}" "${3:-}" "${4:-}" "${5:-}" <<'PY'
import json, sys, time, shutil, threading, re, textwrap, urllib.request
url, D, R, B, DEV, PFILE = sys.argv[1:7]
PROMPT = "In two sentences, what is a hash map?"
body = {"model": "deepseek-v4.1-flash", "max_tokens": 256, "temperature": 0.6,
        "reasoning_effort": "low", "stream": True,
        "stream_options": {"include_usage": True},
        "messages": [{"role": "user", "content": PROMPT}]}
err = sys.stderr
cols, rows = shutil.get_terminal_size()
live = err.isatty() and rows >= 18
LAB = 8                       # label column ("thinking")
PRE = 5 + LAB + 3             # 5 indent + label + " › "
W = max(24, cols - PRE - 2)   # wrap width, never reaches the terminal edge
TL, RL, H = 3, 5, 13          # thinking rows, reply rows, total panel height
lock = threading.Lock()
st = {"think": "", "reply": "", "cur": 0.0, "peak": 0.0, "final": False, "avg": 0.0}
t0 = time.time()

def fmt(m): return f"{m/1024:.1f} GB/s" if m >= 1000 else f"{int(m)} MB/s"
def sectors():
    if not DEV: return None
    try:
        for l in open("/proc/diskstats"):
            f = l.split()
            if f[2] == DEV: return int(f[5])
    except Exception: pass
    return None
def clean(t):
    t = t.replace("**", "").replace("`", "")
    t = re.sub(r"(^|\s)#{1,6}\s", " ", t)
    return " ".join(t.split())
def block(label, text, n, lstyle, tstyle):
    ls = textwrap.wrap(clean(text), W) if text.strip() else []
    if len(ls) > n:
        ls = ls[-n:]; ls[0] = "… " + ls[0]
    rows_ = []
    for i in range(n):
        body = ls[i] if i < len(ls) else ""
        head = f"     {lstyle}{label:<{LAB}} ›{R} " if i == 0 else " " * PRE
        rows_.append(head + (f"{tstyle}{body}{R}" if body else ""))
    return rows_
def panel():
    with lock:
        th, rp = st["think"], st["reply"]
        if st["final"]: disk = f"avg {fmt(st['avg'])}  ·  peak {fmt(st['peak'])}"
        else:           disk = f"{B}{fmt(st['cur'])}{R}{D}  ·  peak {fmt(st['peak'])}  ·  {time.time()-t0:.0f}s"
    L = [f"     {D}{'you':<{LAB}} ›{R} {PROMPT}", ""]
    L += block("thinking", th, TL, D, D) + [""]
    L += block("reply", rp, RL, B, "") + [""]
    L.append(f"     {D}{'disk read' if DEV else ''}{R}  {D}{disk}{R}" if DEV else "")
    return L
first = [True]
def draw():
    L = panel()
    buf = "" if first[0] else f"\033[{H}A"
    first[0] = False
    err.write(buf + "".join(f"\r\033[K{l}\n" for l in L)); err.flush()

stop = threading.Event()
def sampler():
    prev, pt = sectors(), time.time()
    while not stop.wait(0.25):
        s, now = sectors(), time.time()
        if s is not None and prev is not None and now - pt >= 0.5:
            with lock:
                st["cur"] = (s - prev) * 512 / 1048576 / max(now - pt, 1e-3)
                st["peak"] = max(st["peak"], st["cur"])
            prev, pt = s, now
        if live:
            with lock_draw: draw()
lock_draw = threading.Lock()
s0 = sectors()
err.write("\n")
if live:
    with lock_draw: draw()
th = threading.Thread(target=sampler, daemon=True); th.start()

n = 0; chunks = 0; raw_all = []; result = "fail"; errmsg = ""
try:
    req = urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1200) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            raw_all.append(line)
            if not line.startswith("data:"): continue
            p = line[5:].strip()
            if p == "[DONE]": break
            try: d = json.loads(p)
            except Exception: continue
            u = d.get("usage")
            if u and u.get("completion_tokens"): n = u["completion_tokens"]
            for c in d.get("choices") or []:
                dl = c.get("delta") or {}
                rs = dl.get("reasoning_content") or dl.get("reasoning") or ""
                tx = dl.get("content") or ""
                with lock:
                    if rs: st["think"] += rs; chunks += 1
                    if tx: st["reply"] += tx; chunks += 1
    if chunks == 0:   # server ignored stream=true and sent plain JSON
        try:
            d = json.loads("".join(raw_all)); m = d["choices"][0]["message"]
            n = (d.get("usage") or {}).get("completion_tokens", 0)
            with lock:
                st["think"] = m.get("reasoning_content") or ""; st["reply"] = m.get("content") or ""
        except Exception: pass
    result = f"{(n or chunks) / max(time.time() - t0, 1e-3):.2f}"
except Exception as e:
    errmsg = str(e)[:120]
finally:
    stop.set(); th.join(timeout=1)
    s1 = sectors(); el = max(time.time() - t0, 1e-3)
    with lock:
        st["final"] = True
        if s0 is not None and s1 is not None: st["avg"] = (s1 - s0) * 512 / 1048576 / el
    if DEV and s0 is not None and s1 is not None:
        try: open(PFILE, "w").write(str(int(st["peak"])))
        except Exception: pass
    if not live: first[0] = True
    with lock_draw: draw()
    if errmsg: err.write(f"     {D}request failed: {errmsg}{R}\n")
    print(result)
PY
}

LIVE_DRAWN=0
cursor_hide() { [ "$TTY" -eq 1 ] && printf '\033[?25l' >&2; }
cursor_show() { [ "$TTY" -eq 1 ] && printf '\033[?25h' >&2; }
live_draw() {
  if [ "$LIVE_DRAWN" -eq 1 ]; then printf '\033[2A' >&2; else cursor_hide; fi
  printf '\r\033[K%s\n\033[K%s\n' "$1" "$2" >&2
  LIVE_DRAWN=1
}
live_clear() {
  [ "$LIVE_DRAWN" -eq 1 ] && printf '\033[2A\033[J' >&2
  LIVE_DRAWN=0
  cursor_show
}

trial() {
  local before; before=$(wc -l < "$TLOG" 2>/dev/null || echo 0)
  launch "$1" "$2" >> "$TLOG" 2>&1 &
  local pid=$! waited=0 tick=0 last="" pkv=0 pkr=999 pkd=0 spin=(⠋ ⠙ ⠹ ⠸ ⠼ ⠴ ⠦ ⠧ ⠇ ⠏) l1 l2 swp
  LIVE_DRAWN=0
  while [ "$waited" -lt "$LOAD_TIMEOUT" ]; do
    if ! kill -0 "$pid" 2>/dev/null; then
      live_clear
      bad "engine exited during load ${D}· details in $TLOG${R}"
      echo "$pkv $pkr $waited $pkd" > "$PEAKF"; echo fail; return
    fi
    if curl -sf "$URL/v1/models" >/dev/null 2>&1; then
      live_clear
      stats
      [ "$VU" -gt "$pkv" ] && pkv=$VU
      [ "${DISK_MBS:-0}" -gt "$pkd" ] && pkd=$DISK_MBS
      local loaded=$waited tps bd; rm -f "$BENCHD"
      tps=$(bench "$D" "$R" "$B" "$DISK_DEV" "$BENCHD") || tps=fail
      bd=$(cat "$BENCHD" 2>/dev/null || echo 0)
      [ "${bd:-0}" -gt "$pkd" ] 2>/dev/null && pkd=$bd
      stats
      [ "$VU" -gt "$pkv" ] && pkv=$VU
      "$STRATA" launch --stop >/dev/null 2>&1; kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null
      echo "$pkv $pkr $loaded $pkd" > "$PEAKF"; echo "${tps:-fail}"; return
    fi
    stats
    [ "$VU" -gt "$pkv" ] && pkv=$VU
    awk "BEGIN{exit !($RAM_FREE < $pkr)}" && pkr=$RAM_FREE
    [ "${DISK_MBS:-0}" -gt "$pkd" ] && pkd=$DISK_MBS
    last=""
    if [ "$TTY" -eq 1 ]; then
      swp=$D; awk "BEGIN{exit !($SWAP > 0.5)}" && swp=$YLW
      l1="     ${ACC}${spin[tick % 10]}${R} ${B}$(printf '%5s' "$(fmt_t "$waited")")${R}  RAM ${B}${RAM_FREE}G${R} free  VRAM $(bar "$VU" "$VT" 10) $(gib "$VU")/$(gib "$VT")G  GPU ${GU}% ${GT}°C  disk ${B}$(fmt_rate "$DISK_MBS")${R}  swap ${swp}${SWAP}G${R}"
      l2="         ${D}↳ loading model weights…${R}"
      live_draw "$l1" "$l2"
    elif [ $((waited % 20)) -eq 0 ]; then
      say "loading ${waited}s │ RAM free ${RAM_FREE}G │ VRAM ${VU}/${VT}M │ GPU ${GU}% │ load ${LOAD} │ disk $(fmt_rate "$DISK_MBS") │ swap ${SWAP}G"
    fi
    tick=$((tick + 1))
    sleep "$POLL"; waited=$((waited + POLL))
  done
  live_clear
  bad "timed out after ${LOAD_TIMEOUT}s"
  "$STRATA" launch --stop >/dev/null 2>&1; kill "$pid" 2>/dev/null
  echo "$pkv $pkr $waited $pkd" > "$PEAKF"; echo fail
}

pass() { [ "$1" != fail ] && awk "BEGIN{exit !($1 >= $MIN_TPS)}"; }

VISION=on
EXPERTS_FOUND=""

cached() { grep -F -- "$1 -> " "$RESULTS" 2>/dev/null | tail -n1 | sed 's/.* -> //'; }

report_trial() {
  local r=$1 pkv pkr wt pkd prev tag=""
  read -r pkv pkr wt pkd < "$PEAKF" 2>/dev/null || { pkv=0; pkr=0; wt=0; pkd=0; }
  local meta="${D}·  loaded in $(fmt_t "${wt:-0}")  ·  VRAM peak $(gib "$pkv")G  ·  RAM low ${pkr}G${DISK_DEV:+  ·  disk peak $(fmt_rate "${pkd:-0}")}${R}"
  prev=$(grep -oE -- '-> [0-9.]+' "$RESULTS" 2>/dev/null | awk '{print $2}' | sort -g | tail -1)
  if [ "$r" = fail ]; then
    printf '     %s✗%s %sfailed%s  %s\n' "$RED" "$R" "$RED" "$R" "$meta" >&2
  else
    { [ -z "$prev" ] || awk "BEGIN{exit !($r > $prev)}"; } && tag="  ${YLW}★ best so far${R}"
    printf '     %s✓%s %s%s tok/s%s%s  %s\n' "$GRN" "$R" "$B" "$r" "$R" "$tag" "$meta" >&2
  fi
}

# atomic profile writer; keeps the user's USE_* settings
write_profile() {  # ctx experts vision tps partial
  local keep=""; [ -f "$BEST" ] && keep=$(grep -E '^(USE_|SSD_CACHE|CPU_POOL|SPEED_CTX)' "$BEST")
  { printf 'MODEL=%s\nCTX=%s\nVISION=%s\nEXPERTS=%s\nTPS=%s\nPARTIAL=%s\nDATE=%s\n' \
      "$MODEL_PATH" "$1" "$3" "$2" "$4" "$5" "$(date -Iseconds)"
    [ -n "$keep" ] && printf '%s\n' "$keep"
    true
  } > "$BEST.tmp" && mv "$BEST.tmp" "$BEST"
}
# best-so-far profile, written after every successful trial so an interrupt never loses progress
partial_experts() {  # experts tps
  local cur_e=0 cur_c=8k
  if [ -f "$BEST" ]; then eval "$( source "$BEST" 2>/dev/null; printf 'cur_e=%q cur_c=%q' "${EXPERTS:-0}" "${CTX:-8k}" )"; fi
  [ "$1" -gt "${cur_e:-0}" ] 2>/dev/null || [ ! -f "$BEST" ] && write_profile "$cur_c" "$1" "$VISION" "$2" 1
}
partial_ctx() {  # ctx_k tps
  local cur_c=0
  if [ -f "$BEST" ]; then eval "$( source "$BEST" 2>/dev/null; printf 'cur_c=%q' "${CTX%k}" )"; fi
  [ "$1" -gt "${cur_c:-0}" ] 2>/dev/null && write_profile "${1}k" "$EXPERTS_FOUND" "$VISION" "$2" 1
}
# only real results are cached (a killed trial must never leave a bogus entry)
record() { case "$2" in fail|[0-9]*) echo "$1 -> $2" >> "$RESULTS" ;; esac; }

probe_experts() {
  local key="experts=$1 ctx=8k vision=$VISION cache=$SSD_CACHE cores=$POOL_NAME" r; r=$(cached "$key")
  if [ -n "$r" ]; then echo "$r"; return; fi
  set_flags "$1"; say "${ACC}▸${R}  experts ${B}$1${R} ${D}·${R} context ${B}8k${R} ${D}·${R} vision ${B}$VISION${R}"
  r=$(trial 8k "$VISION")
  report_trial "$r"; printf '\n' >&2
  record "$key" "$r"; pass "$r" && partial_experts "$1" "$r"; echo "$r"
}
probe_ctx() {
  local key="experts=$EXPERTS_FOUND ctx=${1}k vision=$VISION cache=$SSD_CACHE cores=$POOL_NAME" r; r=$(cached "$key")
  if [ -n "$r" ]; then echo "$r"; return; fi
  set_flags "$EXPERTS_FOUND"; say "${ACC}▸${R}  context ${B}${1}k${R} ${D}·${R} experts ${B}$EXPERTS_FOUND${R} ${D}·${R} vision ${B}$VISION${R}"
  r=$(trial "${1}k" "$VISION")
  report_trial "$r"; printf '\n' >&2
  record "$key" "$r"; pass "$r" && partial_ctx "$1" "$r"; echo "$r"
}

search() {
  local fn=$1 v=$2 lo=0 hi=0
  if pass "$($fn "$v")"; then
    lo=$v
    while :; do
      v=$((lo * 2))
      if [ "$v" -gt "$RUNAWAY_LIMIT" ]; then echo "$lo"; return; fi
      if pass "$($fn "$v")"; then lo=$v; else hi=$v; break; fi
    done
  else
    hi=$v; v=$((v / 2))
    while [ "$v" -ge 1 ]; do
      if pass "$($fn "$v")"; then lo=$v; break; fi
      hi=$v; v=$((v / 2))
    done
    [ "$lo" -eq 0 ] && return
  fi
  while [ $((hi - lo)) -gt 1 ]; do
    local mid=$(((lo + hi) / 2))
    if pass "$($fn "$mid")"; then lo=$mid; else hi=$mid; fi
  done
  echo "$lo"
}

probe_cache() {  # ssd-cache size to test
  local key="experts=$EXPERTS_FOUND ctx=8k vision=$VISION cache=$1 cores=$POOL_NAME" r; r=$(cached "$key")
  if [ -n "$r" ]; then echo "$r"; return; fi
  SSD_CACHE=$1; set_flags "$EXPERTS_FOUND"
  say "${ACC}▸${R}  ssd cache ${B}$1${R} ${D}·${R} experts ${B}$EXPERTS_FOUND${R} ${D}·${R} context ${B}8k${R} ${D}·${R} vision ${B}$VISION${R}"
  r=$(trial 8k "$VISION")
  report_trial "$r"; printf '\n' >&2
  record "$key" "$r"; echo "$r"
}
# try bigger RAM caches until the speed stops improving, keep the best
sweep_cache() {
  local free cap c r best_c=$SSD_CACHE best_t=0 cands=()
  stats; free=${RAM_FREE%.*}; free=${free:-0}
  cap=${SSD_CACHE_MAX:-$(( free - 3 ))}          # leave ~3 GB for the system
  for c in "${CACHE_STEPS[@]}"; do [ "$c" -le "$cap" ] && cands+=("$c"); done
  if [ "${#cands[@]}" -eq 0 ]; then
    warn "only ${free}G RAM free, skipping the cache sweep (keeping ${SSD_CACHE})" ; echo "$SSD_CACHE"; return
  fi
  info "RAM free ${free}G ${D}· testing caches ${cands[*]} (keeps ~3G for the system)${R}" >&2
  for c in "${cands[@]}"; do
    r=$(probe_cache "$c")
    if ! pass "$r"; then warn "cache $c failed, stopping the sweep" >&2; break; fi
    if [ "$best_t" = 0 ] || awk "BEGIN{exit !($r > $best_t * 1.03)}"; then
      best_t=$r; best_c=$c; save_kv SSD_CACHE "$c"
    else
      break    # a bigger cache gave no real gain
    fi
  done
  echo "$best_c"
}
cpu_lists() {
  local n; n=$(nproc --all 2>/dev/null || echo 1)
  POOL_ALL=$(seq -s, 0 $(( n - 1 )))
  POOL_PHYS=$(lscpu -p=CPU,CORE,SOCKET 2>/dev/null | grep -v '^#' | awk -F, '!seen[$2","$3]++{printf "%s%s", (c++ ? "," : ""), $1}')
}
set_pool() {  # none | phys | all | explicit list like 0,1,2 or 0-5
  POOL_NAME=$1
  case $1 in
    all)  POOL_FLAG="--pool-cores $POOL_ALL" ;;
    phys) POOL_FLAG="--pool-cores $POOL_PHYS" ;;
    none|'') POOL_NAME=none; POOL_FLAG="" ;;
    *)    POOL_FLAG="--pool-cores $1" ;;
  esac
}
probe_cores() {  # none | phys | all
  local key="experts=$EXPERTS_FOUND ctx=8k vision=$VISION cache=$SSD_CACHE cores=$1" r; r=$(cached "$key")
  if [ -n "$r" ]; then echo "$r"; return; fi
  set_pool "$1"; set_flags "$EXPERTS_FOUND"
  say "${ACC}▸${R}  cpu pinning ${B}$1${R} ${D}·${R} experts ${B}$EXPERTS_FOUND${R} ${D}·${R} cache ${B}$SSD_CACHE${R} ${D}·${R} context ${B}8k${R}"
  r=$(trial 8k "$VISION")
  report_trial "$r"; printf '\n' >&2
  record "$key" "$r"; echo "$r"
}
# compare no pinning against one-thread-per-core and all-threads, keep the fastest
sweep_cores() {
  local base r n best=none best_t cands=()
  base=$(probe_cores none)
  pass "$base" || { echo none; return; }
  best_t=$base
  [ -n "$POOL_PHYS" ] && [ "$POOL_PHYS" != "$POOL_ALL" ] && cands+=(phys)
  cands+=(all)
  for n in "${cands[@]}"; do
    r=$(probe_cores "$n")
    if ! pass "$r"; then warn "cpu pinning '$n' failed, skipping it" >&2; continue; fi
    if awk "BEGIN{exit !($r > $best_t * 1.03)}"; then best_t=$r; best=$n; fi
  done
  echo "$best"
}
# largest context whose speed is within 3% of the best speed seen (for this experts/cache/cores)
fast_ctx() {
  awk -v e="experts=$EXPERTS_FOUND" -v v="vision=$VISION" -v c="cache=$SSD_CACHE" -v p="cores=$POOL_NAME" '
    /^#/ || index($0, " -> ") == 0 { next }
    { split($0, a, " -> "); k = " " a[1] " "
      if (!index(k, " " e " ") || !index(k, " " v " ") || !index(k, " " c " ") || !index(k, " " p " ")) next
      if (a[2] == "fail") next
      if (!match(a[1], /ctx=[0-9]+k/)) next
      cx = substr(a[1], RSTART + 4, RLENGTH - 5) + 0
      t[cx] = a[2] + 0; if (a[2] + 0 > best) best = a[2] + 0 }
    END { for (cx in t) { n = cx + 0; if (t[cx] >= best * 0.97 && n > m) m = n } if (m) print m "k" }' "$RESULTS"
}
stepper() {
  local n=$1 i g c line1='   ' line2='   ' lab=(vision experts cache cores context) w
  for i in 1 2 3 4 5; do
    if [ "$i" -lt "$n" ]; then c=$GRN; g='●'; elif [ "$i" -eq "$n" ]; then c=$ACC; g='◉'; else c=$D; g='○'; fi
    line1+="${c}${g}${R}"
    [ "$i" -lt 5 ] && line1+="${BRD}$(rep ━ 10)${R}"
    w=11; [ "$i" -eq 5 ] && w=7
    if [ "$i" -eq "$n" ]; then line2+="${B}${c}$(padr "${lab[$((i - 1))]}" "$w")${R}"; else line2+="${c}$(padr "${lab[$((i - 1))]}" "$w")${R}"; fi
  done
  printf '%s\n%s\n\n' "$line1" "$line2" >&2
}
results_table() {
  local pick_e=$1 pick_c=$2 pick_ca=$3 pick_pl=$4 maxv
  maxv=$(grep -oE -- '-> [0-9.]+' "$RESULTS" | awk '{print $2}' | sort -g | tail -1)
  section "Results"
  printf '      %s%-9s%-9s%-8s%-8s%-8s%s%s\n' "$D" experts context vision cache cores speed "$R" >&2
  local line key val e cx v ca pl kv text n gut
  while IFS= read -r line; do
    case $line in '#'*|'') continue ;; esac
    key=${line% -> *}; val=${line##* -> }
    e='' cx='' v='' ca='' pl=''
    for kv in $key; do
      case $kv in experts=*) e=${kv#*=};; ctx=*) cx=${kv#*=};; vision=*) v=${kv#*=};; cache=*) ca=${kv#*=};; cores=*) pl=${kv#*=};; esac
    done
    gut='   '
    if [ "$val" = fail ]; then
      text="${D}$(rep ▱ 8)${R} ${RED}failed${R}"
    else
      n=$(awk "BEGIN{n=int($val/$maxv*8+0.5); if(n<1)n=1; print n}")
      text="${GRN}$(rep ▰ "$n")${D}$(rep ▱ $((8 - n)))${R} ${val} tok/s"
      if [ "$e" = "$pick_e" ] && [ "$cx" = "$pick_c" ] && [ "$ca" = "$pick_ca" ] && [ "$pl" = "$pick_pl" ]; then gut=" ${YLW}★${R} "; text+="  ${YLW}chosen${R}"; fi
    fi
    printf '  %s %-9s%-9s%-8s%-8s%-8s%s\n' "$gut" "$e" "$cx" "$v" "$ca" "$pl" "$text" >&2
  done < "$RESULTS"
}

tune() {
  ENGINE_STARTED=1
  local hdr="# model=$MODEL_PATH" done_n
  [ "$(head -n1 "$RESULTS" 2>/dev/null)" = "$hdr" ] || printf '%s\n' "$hdr" > "$RESULTS"
  done_n=$(grep -c ' -> ' "$RESULTS" 2>/dev/null)
  if [ "${done_n:-0}" -gt 0 ]; then
    section "Resuming tune" "continuing where it left off · ${done_n} results cached"
    info "finished trials are skipped, progress is saved after every trial"
  else
    section "First run" "one-time tune, the result is saved as a profile"
    info "loads the model many times, so this takes a while"
    info "progress is saved after every trial, so you can stop and resume any time"
  fi
  info "engine log  ${D}$TLOG${R}"
  local base; base=$(native_experts)
  if [ -z "$base" ]; then warn "couldn't read the model's expert count, starting at 6"; base=6
  elif [ "$EXPERTS_MODE" = max ]; then ok "native expert count ${B}$base${R} ${D}· search starts here, then goes higher${R}"
  else ok "native expert count ${B}$base${R} ${D}· using it, more experts are slower and not what the model was trained for${R}"; fi
  EXPERTS_FOUND=$base

  printf '\n' >&2
  stepper 1
  if ! pass "$(probe_experts "$base")"; then
    warn "vision on didn't load, retrying with vision off"
    VISION=off
    pass "$(probe_experts "$base")" || { bad "nothing loads even at 8k, see $TLOG"; exit 1; }
  fi
  ok "vision ${B}$VISION${R}"

  printf '\n' >&2
  stepper 2
  if [ -n "${EXPERTS:-}" ]; then EXPERTS_FOUND=$EXPERTS; info "forced to $EXPERTS_FOUND"
  elif [ "$EXPERTS_MODE" = max ]; then EXPERTS_FOUND=$(search probe_experts "$base")
  else EXPERTS_FOUND=$base; fi
  [ -n "$EXPERTS_FOUND" ] || { bad "no expert count worked"; exit 1; }
  ok "experts ${B}$EXPERTS_FOUND${R}"

  printf '\n' >&2
  stepper 3
  if [ -n "$SSD_CACHE_FORCED" ]; then info "ssd cache forced to $SSD_CACHE_FORCED"
  else SSD_CACHE=$(sweep_cache); fi
  save_kv SSD_CACHE "$SSD_CACHE"
  ok "ssd cache ${B}$SSD_CACHE${R}"

  printf '\n' >&2
  stepper 4
  cpu_lists
  if [ -n "$POOL_FORCED" ]; then set_pool "$POOL_FORCED"; info "cpu pinning forced to $POOL_NAME"
  else set_pool "$(sweep_cores)"; fi
  save_kv CPU_POOL "$POOL_NAME"
  ok "cpu pinning ${B}$POOL_NAME${R}"

  printf '\n' >&2
  stepper 5
  local ctxk; ctxk=$(search probe_ctx 8)
  [ -n "$ctxk" ] || { bad "no context loaded"; exit 1; }
  ok "max context ${B}${ctxk}k${R}"

  local tps; tps=$(probe_ctx "$ctxk")
  write_profile "${ctxk}k" "$EXPERTS_FOUND" "$VISION" "$tps" 0
  save_kv SSD_CACHE "$SSD_CACHE"; save_kv CPU_POOL "$POOL_NAME"
  local fastk; fastk=$(fast_ctx)
  if [ -n "$fastk" ] && [ "${fastk%k}" -lt "$ctxk" ]; then save_kv SPEED_CTX "$fastk"; ok "fastest context ${B}${fastk}${R} ${D}· largest within 3% of the best speed${R}"; fi

  results_table "$EXPERTS_FOUND" "${ctxk}k" "$SSD_CACHE" "$POOL_NAME"
  section "Ready" "tuned in $(elapsed)"
  tiles "${ctxk}k" "$EXPERTS_FOUND" "$VISION" "${tps} tok/s"
  printf '\n' >&2
  prow info saved "$BEST"
  printf '\n' >&2
  info "next launches skip tuning ${D}· run ./start.sh --retune to test again${R}"
}

# ── profile settings (context / reasoning / output) ─────────────────────────
# experts + vision are fixed by the tune. Only these three can be changed.
save_kv() {
  local k=$1 v=$2
  if grep -q "^${k}=" "$BEST" 2>/dev/null; then sed -i "s|^${k}=.*|${k}=${v}|" "$BEST"
  else printf '%s=%s\n' "$k" "$v" >> "$BEST"; fi
}

ctx_eff_k() { case "$1" in best) echo "${2%k}" ;; fast) echo "${SPEED_CTX:-$2}" | tr -d k ;; *) echo "${1%k}" ;; esac; }

# opt_values <row> <tuned-ctx> <selected-ctx>  → fills OV (values) and OL (labels)
opt_values() {
  OV=(); OL=()
  local v max=${2%k} eff; eff=$(ctx_eff_k "$3" "$2")
  case $1 in
    0) OV+=(best); OL+=("best · ${2} (tuned max)")
       if [ -n "${SPEED_CTX:-}" ] && [ "${SPEED_CTX%k}" -lt "$max" ]; then OV+=(fast); OL+=("fast · ${SPEED_CTX} (best speed)"); fi
       for v in 4 8 16 32 64 128 256 512; do
         [ "$v" -lt "$max" ] && { OV+=("${v}k"); OL+=("${v}k"); }
       done ;;
    1) OV=(auto low medium high max)
       OL=("auto · engine default" "low" "medium" "high" "max") ;;
    2) OV+=(auto); OL+=("auto · no cap")
       for v in 1024 2048 4096 8192 16384 32768 65536; do
         [ "$v" -le $(( eff * 1024 / 2 )) ] && { OV+=("$v"); OL+=("$((v / 1024))k  ${D}($v tokens)${R}"); }
       done ;;
  esac
}

settings_screen() {
  if [ ! -f "$BEST" ]; then warn "no profile yet, run a tune first"; return; fi
  if [ "$TTY" -ne 1 ] || [ ! -t 0 ]; then warn "settings needs an interactive terminal"; return; fi
  local CTX EXPERTS VISION TPS USE_CTX USE_REASON USE_OUT SPEED_CTX CPU_POOL SSD_CACHE
  source "$BEST"
  USE_CTX=${USE_CTX:-${SPEED_CTX:+fast}}; USE_CTX=${USE_CTX:-best}
  local SEL=("$USE_CTX" "${USE_REASON:-auto}" "${USE_OUT:-auto}")
  local names=("context" "reasoning" "output")
  local row=0 open=0 cur=0 drawn=0 i j key rest idx
  local OV=() OL=()

  printf '\n  %s▍%s %sSettings%s  %sexperts %s · vision %s stay as tuned%s\n\n' "$ACC" "$R" "$B" "$R" "$D" "$EXPERTS" "$VISION" "$R" >&2
  cursor_hide
  while :; do
    [ "$drawn" -gt 0 ] && printf '\033[%dA' "$drawn" >&2
    drawn=0
    for i in 0 1 2; do
      opt_values "$i" "$CTX" "${SEL[0]}"
      idx=0; for j in "${!OV[@]}"; do [ "${OV[$j]}" = "${SEL[$i]}" ] && idx=$j; done
      if [ "$i" -eq "$row" ]; then
        printf '\r\033[K  %s❯%s %s%-10s%s %s%s%s %s▾%s\n' "$ACC" "$R" "$B" "${names[$i]}" "$R" "$ACC" "${OL[$idx]}" "$R" "$D" "$R" >&2
      else
        printf '\r\033[K    %s%-10s%s %s\n' "$D" "${names[$i]}" "$R" "${OL[$idx]}" >&2
      fi
      drawn=$((drawn + 1))
      if [ "$open" -eq 1 ] && [ "$i" -eq "$row" ]; then
        for j in "${!OV[@]}"; do
          if [ "$j" -eq "$cur" ]; then printf '\r\033[K        %s●%s %s%s%s\n' "$ACC" "$R" "$B" "${OL[$j]}" "$R" >&2
          else printf '\r\033[K        %s○ %s%s\n' "$D" "${OL[$j]}" "$R" >&2; fi
          drawn=$((drawn + 1))
        done
      fi
    done
    if [ "$open" -eq 1 ]; then printf '\r\033[K\n\r\033[K   %s↑↓ choose  ·  enter confirm  ·  esc cancel%s\n' "$D" "$R" >&2
    else printf '\r\033[K\n\r\033[K   %s↑↓ move  ·  enter open list  ·  q save & back%s\n' "$D" "$R" >&2; fi
    drawn=$((drawn + 2))
    printf '\033[J' >&2

    IFS= read -rsn1 key || break
    rest=''
    if [ "$key" = $'\e' ]; then read -rsn2 -t 0.05 rest || rest=''; fi
    opt_values "$row" "$CTX" "${SEL[0]}"
    if [ "$open" -eq 1 ]; then
      case "$key$rest" in
        $'\e[A'|k) cur=$(( (cur + ${#OV[@]} - 1) % ${#OV[@]} )) ;;
        $'\e[B'|j) cur=$(( (cur + 1) % ${#OV[@]} )) ;;
        '') SEL[$row]=${OV[$cur]}; open=0
            if [ "$row" -eq 0 ]; then   # context changed → output cap must still fit
              opt_values 2 "$CTX" "${SEL[0]}"
              local ok=0; for j in "${OV[@]}"; do [ "$j" = "${SEL[2]}" ] && ok=1; done
              [ "$ok" -eq 1 ] || SEL[2]=auto
            fi ;;
        $'\e'|$'\e[D'|q) open=0 ;;
      esac
    else
      case "$key$rest" in
        $'\e[A'|k) row=$(( (row + 2) % 3 )) ;;
        $'\e[B'|j) row=$(( (row + 1) % 3 )) ;;
        ''|' '|$'\e[C')
          open=1; cur=0
          for j in "${!OV[@]}"; do [ "${OV[$j]}" = "${SEL[$row]}" ] && cur=$j; done ;;
        q|Q|$'\e') break ;;
      esac
    fi
  done
  printf '\033[%dA\033[J' "$drawn" >&2
  cursor_show
  save_kv USE_CTX "${SEL[0]}"; save_kv USE_REASON "${SEL[1]}"; save_kv USE_OUT "${SEL[2]}"
  ok "saved  ${D}context${R} ${SEL[0]}  ${D}·  reasoning${R} ${SEL[1]}  ${D}·  output${R} ${SEL[2]}"
}

menu() {
  local sel=0 i key rest n key_map=()
  local opts=() desc=() mode=()
  local part=0
  [ -f "$BEST" ] && part=$( source "$BEST" 2>/dev/null; echo "${PARTIAL:-0}" )
  if [ -f "$BEST" ] && [ "$part" = 1 ]; then
    opts+=("Resume tuning"); desc+=("continue where the last tune stopped"); mode+=(run)
    opts+=("Run now");       desc+=("launch with the best settings found so far"); mode+=(runnow)
    opts+=("Settings");      desc+=("pick context, reasoning and output length"); mode+=(settings)
  elif [ -f "$BEST" ]; then
    opts+=("Run");      desc+=("start with your saved profile");            mode+=(run)
    opts+=("Settings"); desc+=("pick context, reasoning and output length"); mode+=(settings)
  else
    opts+=("Tune and run"); desc+=("no saved profile yet, test your machine first"); mode+=(run)
  fi
  opts+=("Retune"); desc+=("test again and save a new profile"); mode+=(retune)
  opts+=("Quit");   desc+=("exit without starting");             mode+=(quit)
  n=${#opts[@]}
  if [ -f "$BEST" ]; then
    local sv; sv=$( source "$BEST" 2>/dev/null
      printf '%s · %s experts · vision %s · %s tok/s' "${CTX:-?}" "${EXPERTS:-?}" "${VISION:-?}" "${TPS:-?}"
      printf '\n   %sactive%s   context %s · reasoning %s · output %s' "$D" "$R" "${USE_CTX:-best}" "${USE_REASON:-auto}" "${USE_OUT:-auto}" )
    printf '   %ssaved%s    %s\n' "$D" "$R" "$sv" >&2
    [ "$part" = 1 ] && printf '   %s▲%s  %sunfinished tune, best found so far%s\n' "$YLW" "$R" "$D" "$R" >&2
  fi
  printf '\n' >&2
  cursor_hide
  local drawn=0
  while :; do
    [ "$drawn" -eq 1 ] && printf '\033[%dA' $((n + 2)) >&2
    for ((i = 0; i < n; i++)); do
      if [ "$i" -eq "$sel" ]; then
        printf '\r\033[K  %s❯%s %s%-14s%s %s%s%s\n' "$ACC" "$R" "$B" "${opts[$i]}" "$R" "$D" "${desc[$i]}" "$R" >&2
      else
        printf '\r\033[K    %s%-14s %s%s\n' "$D" "${opts[$i]}" "${desc[$i]}" "$R" >&2
      fi
    done
    printf '\r\033[K\n\r\033[K   %s↑↓ move  ·  enter select  ·  q quit%s\n' "$D" "$R" >&2
    drawn=1
    IFS= read -rsn1 key || { sel=$((n - 1)); break; }
    case "$key" in
      $'\e') read -rsn2 -t 0.05 rest || rest=''
        case "$rest" in '[A') sel=$(( (sel + n - 1) % n )) ;; '[B') sel=$(( (sel + 1) % n )) ;; esac ;;
      k) sel=$(( (sel + n - 1) % n )) ;;
      j) sel=$(( (sel + 1) % n )) ;;
      [1-9]) [ "$key" -le "$n" ] && { sel=$((key - 1)); break; } ;;
      q|Q) sel=$((n - 1)); break ;;
      '') break ;;
    esac
  done
  printf '\033[%dA\033[J' $((n + 2)) >&2
  cursor_show
  MODE=${mode[$sel]}
  if [ "$MODE" = quit ]; then printf '\n' >&2; exit 0; fi
}

PROXY_PID=""
PROXYPIDF="$STATE/ds-proxy.pid"
ENGINE_STARTED=0; PHASE=tune
stop_proxy() {
  [ -n "$PROXY_PID" ] && kill "$PROXY_PID" 2>/dev/null; PROXY_PID=""
  if [ -f "$PROXYPIDF" ]; then kill "$(cat "$PROXYPIDF" 2>/dev/null)" 2>/dev/null; rm -f "$PROXYPIDF"; fi
}
# stop everything we may have started: proxy + strata engine (quiet, safe to call twice)
stop_all() { stop_proxy; "$STRATA" launch --stop >/dev/null 2>&1; }

# ── boost: OS-level performance tuning (reversible, resets on reboot) ─────────
# No voltage or clock overclocking. It stops Linux from throttling: CPU governor + energy
# preference, disk read-ahead and scheduler, swap eagerness, GPU persistence. Needs sudo once.
# BOOST=0 turns it off, GPU_LOCK=1 also locks the GPU to its max graphics clock.
BOOST_FS="${BOOST_FS:-}"; BOOST_ORIG="$STATE/boost.orig"; BOOST_APPLIED=0
bset() { printf '%s' "$2" | ${BOOST_SUDO-sudo} tee "$BOOST_FS$1" >/dev/null 2>&1; }
# remember the original value, then write the new one
bsave_set() {  # path value
  local cur
  [ -e "$BOOST_FS$1" ] || return 1
  case "$1" in
    */scheduler) cur=$(sed 's/.*\[\(.*\)\].*/\1/' "$BOOST_FS$1" 2>/dev/null) ;;
    *)           cur=$(cat "$BOOST_FS$1" 2>/dev/null) ;;
  esac
  [ -n "$cur" ] || return 1
  [ "$cur" = "$2" ] && return 1                  # already set, nothing to do or undo
  printf 'file|%s|%s\n' "$1" "$cur" >> "$BOOST_ORIG"
  if ! bset "$1" "$2"; then sed -i '$d' "$BOOST_ORIG"; return 1; fi
  return 0
}
boost_restore() {
  [ "$BOOST_APPLIED" = 1 ] || return 0
  BOOST_APPLIED=0
  local kind a b
  [ -f "$BOOST_ORIG" ] || return 0
  while IFS='|' read -r kind a b; do
    case "$kind" in
      file)     bset "$a" "$b" ;;
      gpu_pm)   [ "$b" = Disabled ] && ${BOOST_SUDO-sudo} nvidia-smi -pm 0 >/dev/null 2>&1 ;;
      gpu_lock) ${BOOST_SUDO-sudo} nvidia-smi -rgc >/dev/null 2>&1 ;;
    esac
  done < "$BOOST_ORIG"
  rm -f "$BOOST_ORIG"
}
boost_apply() {
  [ "${BOOST:-1}" = 0 ] && return 0
  [ "$(id -u)" -eq 0 ] && BOOST_SUDO=""
  if [ -n "${BOOST_SUDO-sudo}" ]; then
    command -v sudo >/dev/null 2>&1 || { warn "no sudo, skipping boost"; return 0; }
    if ! sudo -n true 2>/dev/null; then
      if [ "$TTY" -eq 1 ] && [ -t 0 ]; then
        info "boost needs sudo once ${D}· BOOST=0 skips it${R}"
        sudo -v || { warn "no sudo, skipping boost"; return 0; }
      else warn "boost needs sudo and there's no terminal, skipping"; return 0; fi
    fi
  fi
  # a crash can leave the old settings recorded: put them back first
  if [ -f "$BOOST_ORIG" ]; then BOOST_APPLIED=1; boost_restore; fi
  : > "$BOOST_ORIG"; BOOST_APPLIED=1
  local f par notes=() gov=0 epp=0
  for f in "$BOOST_FS"/sys/devices/system/cpu/cpu[0-9]*/cpufreq; do
    [ -d "$f" ] || continue
    if grep -qw performance "$f/scaling_available_governors" 2>/dev/null; then
      bsave_set "${f#"$BOOST_FS"}/scaling_governor" performance && gov=$((gov + 1))
    fi
    bsave_set "${f#"$BOOST_FS"}/energy_performance_preference" performance && epp=$((epp + 1))
  done
  [ "$gov" -gt 0 ] && notes+=("cpu governor performance (${gov} cores)")
  [ "$epp" -gt 0 ] && notes+=("cpu energy preference performance")
  if [ -n "$DISK_DEV" ]; then
    par=$(lsblk -no pkname "/dev/$DISK_DEV" 2>/dev/null | head -n1); par=${par:-$DISK_DEV}
    bsave_set "/sys/block/$par/queue/read_ahead_kb" 4096 && notes+=("read-ahead 4 MB")
    if grep -qw none "$BOOST_FS/sys/block/$par/queue/scheduler" 2>/dev/null; then
      bsave_set "/sys/block/$par/queue/scheduler" none && notes+=("disk scheduler none")
    fi
  fi
  bsave_set /proc/sys/vm/swappiness 10 && notes+=("swappiness 10")
  bsave_set /proc/sys/vm/vfs_cache_pressure 50 && notes+=("keeps file cache longer")
  if [ "$HAS_NV" -eq 1 ] && [ -z "$BOOST_FS" ]; then
    local pm; pm=$(nvidia-smi --query-gpu=persistence_mode --format=csv,noheader 2>/dev/null | head -n1)
    if [ "$pm" = Disabled ] && ${BOOST_SUDO-sudo} nvidia-smi -pm 1 >/dev/null 2>&1; then
      printf 'gpu_pm|x|Disabled\n' >> "$BOOST_ORIG"; notes+=("gpu persistence")
    fi
    if [ "${GPU_LOCK:-0}" = 1 ]; then
      local mx; mx=$(nvidia-smi --query-gpu=clocks.max.graphics --format=csv,noheader,nounits 2>/dev/null | head -n1)
      if [ -n "$mx" ] && ${BOOST_SUDO-sudo} nvidia-smi -lgc "$mx,$mx" >/dev/null 2>&1; then
        printf 'gpu_lock|x|x\n' >> "$BOOST_ORIG"; notes+=("gpu clock locked at ${mx} MHz")
      fi
    fi
  fi
  if [ "${#notes[@]}" -gt 0 ]; then
    local joined="" n
    for n in "${notes[@]}"; do joined+="${joined:+ · }$n"; done
    info "boost on ${D}· ${joined}${R}"
    info "${D}all of it is undone when the script exits${R}"
  else
    rm -f "$BOOST_ORIG"; BOOST_APPLIED=0; info "boost: system already tuned, nothing to change"
  fi
}

listener_pid() { ss -H -ltnp "sport = :$1" 2>/dev/null | grep -o 'pid=[0-9]*' | head -n1 | cut -d= -f2; }
# on start: if a strata server (or our proxy) from an earlier run is still up, stop it first
free_ports() {
  local p pid cmd up=0 i ports=("$PORT" "$((PORT + 1))")
  for p in "${ports[@]}"; do
    curl -sf --max-time 2 "http://127.0.0.1:$p/v1/models" >/dev/null 2>&1 && up=1
    [ -n "$(listener_pid "$p")" ] && up=1
  done
  [ -f "$PROXYPIDF" ] && kill -0 "$(cat "$PROXYPIDF" 2>/dev/null)" 2>/dev/null && up=1
  [ "$up" -eq 0 ] && return 0
  info "a previous strata server is still running, stopping it"
  stop_all
  for i in 1 2 3 4 5 6 7 8 9 10; do
    up=0; for p in "${ports[@]}"; do [ -n "$(listener_pid "$p")" ] && up=1; done
    [ "$up" -eq 0 ] && break
    sleep 1
  done
  if [ "$up" -eq 1 ]; then
    for p in "${ports[@]}"; do
      pid=$(listener_pid "$p"); [ -n "$pid" ] || continue
      cmd=$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null)
      case "$cmd" in
        *strata*|*ds-proxy.py*) kill "$pid" 2>/dev/null; sleep 2; kill -9 "$pid" 2>/dev/null ;;
        *) bad "port $p is used by something else (pid $pid: ${cmd:0:60}), free it or set PORT=..."; exit 1 ;;
      esac
    done
  fi
  ok "old server stopped"
}

# Small local proxy: applies the reasoning default and output-token cap to API requests.
# Only started when reasoning or output is not "auto".
write_proxy() {
  cat > "$STATE/ds-proxy.py" <<'PROXYEOF'
import sys, json, http.client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
LP, UP, EFF, OUT = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3], sys.argv[4]

class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"
    def log_message(self, *a): pass
    def _go(self):
        n = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(n) if n else b""
        if self.command == "POST" and self.path.split("?")[0].endswith("/chat/completions"):
            try:
                d = json.loads(body)
                if EFF != "auto" and "reasoning_effort" not in d:
                    d["reasoning_effort"] = EFF
                if OUT != "auto":
                    cap, seen = int(OUT), False
                    for k in ("max_tokens", "max_completion_tokens"):
                        if d.get(k) is not None:
                            d[k] = min(int(d[k]), cap); seen = True
                    if not seen: d["max_tokens"] = cap
                body = json.dumps(d).encode()
            except Exception:
                pass
        hdrs = {k: v for k, v in self.headers.items()
                if k.lower() not in ("host", "content-length", "connection", "accept-encoding")}
        hdrs["Content-Length"] = str(len(body))
        try:
            c = http.client.HTTPConnection("127.0.0.1", UP, timeout=3600)
            c.request(self.command, self.path, body, hdrs)
            r = c.getresponse()
        except Exception:
            self.send_error(502); return
        self.send_response(r.status)
        for k, v in r.getheaders():
            if k.lower() not in ("transfer-encoding", "connection", "content-length"):
                self.send_header(k, v)
        self.end_headers()
        try:
            while True:
                chunk = r.read1(65536)
                if not chunk: break
                self.wfile.write(chunk); self.wfile.flush()
        except Exception:
            pass
        c.close()
    do_GET = do_POST = do_PUT = do_DELETE = do_OPTIONS = _go

ThreadingHTTPServer(("127.0.0.1", LP), H).serve_forever()
PROXYEOF
}

cleanup() {
  stop_all; boost_restore; live_clear; [ "$TTY" -eq 1 ] && printf '\0337\033[r\0338' >&2
  printf '\n' >&2
  if [ "$PHASE" = run ]; then info "server stopped"; else warn "interrupted, progress is saved, run ./start.sh to continue"; fi
  exit 130
}
trap cleanup INT TERM HUP
trap 'cursor_show; [ "$ENGINE_STARTED" = 1 ] && stop_all; boost_restore' EXIT

MODE=menu
case "${1:-}" in --retune) MODE=retune ;; --run) MODE=run ;; --settings) MODE=settings ;; esac
banner
while :; do
  if [ "$MODE" = menu ]; then
    if [ "$TTY" -eq 1 ] && [ -t 0 ]; then menu; else MODE=run; fi
  fi
  if [ "$MODE" = settings ]; then
    settings_screen
    if [ "$TTY" -eq 1 ] && [ -t 0 ]; then MODE=menu; continue; else exit 0; fi
  fi
  break
done
free_ports
if [ "$MODE" = retune ]; then rm -f "$BEST" "$RESULTS"; info "retuning from scratch"; fi
preflight
boost_apply
if [ -f "$BEST" ]; then
  unset MODEL CTX EXPERTS VISION
  source "$BEST"
  if [ "${MODEL:-}" != "$MODEL_PATH" ] || [ -z "${CTX:-}" ] || [ -z "${EXPERTS:-}" ] || [ -z "${VISION:-}" ]; then
    warn "profile doesn't match this model, re-tuning"
    rm -f "$BEST"
  fi
fi
unset PARTIAL
[ -f "$BEST" ] && source "$BEST"
SSD_CACHE=${SSD_CACHE_FORCED:-6}     # tuning always starts from the default so cached trials match
POOL_NAME=none; POOL_FLAG=""
if [ ! -f "$BEST" ]; then
  tune
elif [ "${PARTIAL:-0}" = 1 ] && [ "$MODE" != runnow ]; then
  info "found an unfinished tune, resuming it"
  tune
elif [ "${PARTIAL:-0}" = 1 ]; then
  warn "launching with a partial profile, run ./start.sh again to finish tuning"
fi
unset USE_CTX USE_REASON USE_OUT SSD_CACHE CPU_POOL SPEED_CTX
source "$BEST"
SSD_CACHE=${SSD_CACHE_FORCED:-${SSD_CACHE:-6}}
cpu_lists; set_pool "${POOL_FORCED:-${CPU_POOL:-none}}"
USE_CTX=${USE_CTX:-${SPEED_CTX:+fast}}; USE_CTX=${USE_CTX:-best}; USE_REASON=${USE_REASON:-auto}; USE_OUT=${USE_OUT:-auto}

RUN_CTX=$CTX
[ "$USE_CTX" = fast ] && USE_CTX=${SPEED_CTX:-best}
if [ "$USE_CTX" != best ]; then
  if [ "${USE_CTX%k}" -le "${CTX%k}" ] 2>/dev/null; then RUN_CTX=$USE_CTX
  else warn "saved context $USE_CTX is above the tuned max ${CTX}, using ${CTX}"; fi
fi
if [ "$USE_OUT" != auto ] && [ "$USE_OUT" -gt $(( ${RUN_CTX%k} * 1024 / 2 )) ] 2>/dev/null; then
  warn "output $USE_OUT doesn't fit in ${RUN_CTX} context, using auto"; USE_OUT=auto
fi

set_flags "$EXPERTS"
section "Launching" "from your saved profile"
tiles "$RUN_CTX" "$EXPERTS" "$VISION" "${TPS:-?} tok/s"
printf '\n' >&2
prow info cache "ssd-cache $SSD_CACHE"
prow info cpu "pinning $POOL_NAME"
prow info reason "$USE_REASON"
prow info output "$([ "$USE_OUT" = auto ] && echo 'auto (no cap)' || echo "$USE_OUT tokens")"
prow info api "$URL/v1"
prow info saved "$BEST"
printf '\n' >&2
info "engine output follows ${D}· it's ready when the engine says so${R}"
rule

ENGINE_STARTED=1; PHASE=run
if [ "$USE_REASON" = auto ] && [ "$USE_OUT" = auto ]; then
  "$STRATA" launch \
    --model-path "$MODEL_PATH" --model ista-das-coder --family deepseek41 \
    --context "$RUN_CTX" --vision "$VISION" --mtp on --parallel 1 --speed faster \
    --tune skip --access local --no-api-key --port "$PORT" --no-tui --yes --restart
  exit $?
fi

# reasoning / output are request-level, so the engine runs on PORT+1 behind the proxy on PORT
UP_PORT=$((PORT + 1))
write_proxy
python3 "$STATE/ds-proxy.py" "$PORT" "$UP_PORT" "$USE_REASON" "$USE_OUT" &
PROXY_PID=$!
echo "$PROXY_PID" > "$PROXYPIDF"
"$STRATA" launch \
  --model-path "$MODEL_PATH" --model ista-das-coder --family deepseek41 \
  --context "$RUN_CTX" --vision "$VISION" --mtp on --parallel 1 --speed faster \
  --tune skip --access local --no-api-key --port "$UP_PORT" --no-tui --yes --restart
