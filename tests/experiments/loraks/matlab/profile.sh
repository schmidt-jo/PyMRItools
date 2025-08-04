#!/usr/bin/env bash
set -eo pipefail

if [ "$#" -lt 1 ]; then
  echo "Usage: profile command [args...]" >&2
  exit 2
fi

# Launch the target command
"$@" &
ROOT=$!
LOGFILE="profile-${ROOT}.log"
echo "# timestamp_ms pid combined_VmHWM_kB" > "$LOGFILE"

# Function to recursively find all live PIDs in tree
get_tree_pids() {
  local pid=$1
  echo "$pid"
  for child in $(cat /proc/$pid/task/$pid/children 2>/dev/null); do
    get_tree_pids "$child"
  done
}

max_combined=0

# Poll every 50ms until root exits
while kill -0 "$ROOT" 2>/dev/null; do
  now_ms=$(( $(date +%s%3N) ))
  sum=0
  for pid in $(get_tree_pids "$ROOT"); do
    if [ -r /proc/$pid/status ]; then
      val=$(awk '/^VmHWM:/ {print $2}' /proc/$pid/status)
      sum=$((sum + val))
    fi
  done
  (( sum > max_combined )) && max_combined=$sum
  echo "${now_ms} ${ROOT} ${max_combined}" >> "$LOGFILE"
  sleep 0.05
done

# One final read after exit (just in case)
now_ms=$(( $(date +%s%3N) ))
sum=0
for pid in $(get_tree_pids "$ROOT"); do
  if [ -r /proc/$pid/status ]; then
    val=$(awk '/^VmHWM:/ {print $2}' /proc/$pid/status)
    sum=$((sum + val))
  fi
done
(( sum > max_combined )) && max_combined=$sum
echo "${now_ms} ${ROOT} ${max_combined}" >> "$LOGFILE"

echo "# done. Combined peak VmHWM = ${max_combined} kB"
exit 0
