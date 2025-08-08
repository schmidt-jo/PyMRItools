#!/usr/bin/env bash
set -eo pipefail

if [ "$#" -lt 1 ]; then
  echo "Usage: profile command [args...]" >&2
  exit 2
fi

# Launch the target command
"$@" &
ROOT=$!
#LOGFILE="profile-${ROOT}.log"
#echo "# timestamp_ms pid combined_VmHWM_kB" > "$LOGFILE"

# Function to recursively find all live PIDs in tree
get_tree_pids() {
  local pid=$1
  if [ -z "$pid" ]; then return; fi

  echo "$pid"
  # Safely check for children, ignore errors
  if [ -f "/proc/$pid/task/$pid/children" ]; then
    for child in $(cat "/proc/$pid/task/$pid/children" 2>/dev/null); do
      # Prevent potential infinite recursion or empty pid
      if [ -n "$child" ] && [ "$child" != "$pid" ]; then
        get_tree_pids "$child"
      fi
    done
  fi
}

max_combined=0

# Poll every 50ms until root exits
while kill -0 "$ROOT" 2>/dev/null; do
  now_ms=$(( $(date +%s%3N) ))
  sum=0

  # Wrap PID retrieval in error handling
  tree_pids=$(get_tree_pids "$ROOT" 2>/dev/null)

  for pid in $tree_pids; do
    # Additional safety checks
    if [ -n "$pid" ] && [ -r "/proc/$pid/status" ]; then
      # Use safer awk parsing with error suppression
      val=$(awk '/^VmHWM:/ {print $2}' "/proc/$pid/status" 2>/dev/null || echo 0)
      # Ensure val is a number
      if [[ "$val" =~ ^[0-9]+$ ]]; then
        sum=$((sum + val))
      fi
    fi
  done

  (( sum > max_combined )) && max_combined=$sum
#  echo "${now_ms} ${ROOT} ${max_combined}" >> "$LOGFILE"
  sleep 0.05
done

# One final read after exit (just in case)
now_ms=$(( $(date +%s%3N) ))
sum=0
tree_pids=$(get_tree_pids "$ROOT" 2>/dev/null)
for pid in $tree_pids; do
  if [ -n "$pid" ] && [ -r "/proc/$pid/status" ]; then
    val=$(awk '/^VmHWM:/ {print $2}' "/proc/$pid/status" 2>/dev/null || echo 0)
    if [[ "$val" =~ ^[0-9]+$ ]]; then
      sum=$((sum + val))
    fi
  fi
done

(( sum > max_combined )) && max_combined=$sum
#echo "${now_ms} ${ROOT} ${max_combined}" >> "$LOGFILE"

echo "# done. Combined peak VmHWM = ${max_combined} kB"
exit 0