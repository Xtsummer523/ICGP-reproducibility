#!/usr/bin/env bash
set -euo pipefail

USAGE="usage: capture_rviz_gui_from_guest.sh CONFIG RUN_DIR [WAIT_S]"
CONFIG="${1:?$USAGE}"
RUN_DIR="${2:?$USAGE}"
WAIT_S="${3:-18}"
SCREENSHOT_DIR="$RUN_DIR/experiment_screenshots"
LOG="$RUN_DIR/rviz_gui_capture_attempt.log"
MARKER_STDOUT="$RUN_DIR/rviz_marker_node_stdout.log"
MARKER_STDERR="$RUN_DIR/rviz_marker_node_stderr.log"
RVIZ_STDOUT="$RUN_DIR/rviz2_stdout.log"
RVIZ_STDERR="$RUN_DIR/rviz2_stderr.log"
OUTPUT="$SCREENSHOT_DIR/real_rviz_gui.png"

mkdir -p "$SCREENSHOT_DIR"
: >"$LOG"

log() {
  printf '[%s] %s\n' "$(date -Iseconds)" "$*" | tee -a "$LOG"
}

find_display() {
  if [ -n "${DISPLAY:-}" ]; then
    printf '%s\n' "$DISPLAY"
    return 0
  fi
  if [ -S /tmp/.X11-unix/X0 ]; then
    printf ':0\n'
    return 0
  fi
  if [ -S /tmp/.X11-unix/X1 ]; then
    printf ':1\n'
    return 0
  fi
  return 1
}

find_xauthority() {
  local display_name="$1"
  local display_number="${display_name#:}"
  local auth_path=""
  auth_path="$(ps -ef | sed -n "s/.*Xwayland :${display_number} .* -auth \([^ ]*\).*/\1/p" | head -n 1 || true)"
  if [ -n "$auth_path" ] && [ -f "$auth_path" ]; then
    printf '%s\n' "$auth_path"
    return 0
  fi
  for candidate in \
    "$XAUTHORITY" \
    "$HOME/.Xauthority" \
    /run/user/"$(id -u)"/.mutter-Xwaylandauth.*; do
    if [ -n "${candidate:-}" ]; then
      for expanded in $candidate; do
        if [ -f "$expanded" ]; then
          printf '%s\n' "$expanded"
          return 0
        fi
      done
    fi
  done
  return 1
}

DISPLAY_VALUE="$(find_display || true)"
if [ -z "$DISPLAY_VALUE" ]; then
  log "ERROR: no X display socket found under /tmp/.X11-unix."
  exit 2
fi
XAUTHORITY_VALUE="$(find_xauthority "$DISPLAY_VALUE" || true)"
if [ -n "$XAUTHORITY_VALUE" ]; then
  export XAUTHORITY="$XAUTHORITY_VALUE"
fi
export DISPLAY="$DISPLAY_VALUE"
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
export DBUS_SESSION_BUS_ADDRESS="${DBUS_SESSION_BUS_ADDRESS:-unix:path=$XDG_RUNTIME_DIR/bus}"

RVIZ_CONFIG="${ICGP_RVIZ_CONFIG:-}"
if [ -z "$RVIZ_CONFIG" ]; then
  RVIZ_CONFIG="$(ros2 pkg prefix icgp_experiments)/share/icgp_experiments/rviz/icgp_scene.rviz"
fi

log "DISPLAY=$DISPLAY"
log "XAUTHORITY=${XAUTHORITY:-}"
log "RUN_DIR=$RUN_DIR"
log "CONFIG=$CONFIG"
log "WAIT_S=$WAIT_S"
log "rviz2=$(command -v rviz2 || true)"
log "RVIZ_CONFIG=$RVIZ_CONFIG"
log "xwininfo=$(command -v xwininfo || true)"

if [ ! -f "$RVIZ_CONFIG" ]; then
  log "ERROR: RViz config is missing: $RVIZ_CONFIG"
  exit 3
fi
if ! command -v gnome-screenshot >/dev/null 2>&1 && ! command -v import >/dev/null 2>&1; then
  log "ERROR: neither gnome-screenshot nor ImageMagick import is available."
  exit 4
fi

pkill -TERM -f "apport-gtk|update-notifier|rviz2|rviz_scene_marker_node" >/dev/null 2>&1 || true
sleep 1

ros2 run icgp_experiments rviz_scene_marker_node --ros-args -p config_path:="$CONFIG" \
  >"$MARKER_STDOUT" 2>"$MARKER_STDERR" &
MARKER_PID="$!"
echo "$MARKER_PID" >"$RUN_DIR/rviz_marker_node_pid.txt"

rviz2 -d "$RVIZ_CONFIG" >"$RVIZ_STDOUT" 2>"$RVIZ_STDERR" &
RVIZ_PID="$!"
echo "$RVIZ_PID" >"$RUN_DIR/rviz2_pid.txt"

sleep "$WAIT_S"
pkill -TERM -f "apport-gtk|update-notifier" >/dev/null 2>&1 || true
if command -v wmctrl >/dev/null 2>&1; then
  wmctrl -a RViz >/dev/null 2>&1 || true
elif command -v xdotool >/dev/null 2>&1; then
  xdotool search --name RViz windowactivate >/dev/null 2>&1 || true
fi
sleep 1

set +e
CAPTURE_RC="127"
WINDOW_ID=""
if command -v xwininfo >/dev/null 2>&1 && command -v import >/dev/null 2>&1; then
  xwininfo -root -tree >"$RUN_DIR/xwininfo_rviz_tree.txt" 2>>"$LOG" || true
  WINDOW_ID="$(python3 - "$RUN_DIR/xwininfo_rviz_tree.txt" <<'PY' || true
import re
import sys
from pathlib import Path

path = Path(sys.argv[1])
best = ("", -1)
for line in path.read_text(errors="ignore").splitlines():
    lower = line.lower()
    if "rviz" not in lower:
        continue
    match = re.search(r"(0x[0-9a-fA-F]+).*?(\d+)x(\d+)[+-]\d+[+-]\d+", line)
    if not match:
        continue
    area = int(match.group(2)) * int(match.group(3))
    if area > best[1]:
        best = (match.group(1), area)
if best[0]:
    print(best[0])
PY
)"
  if [ -n "$WINDOW_ID" ]; then
    log "capturing RViz window id=$WINDOW_ID with ImageMagick import to $OUTPUT"
    import -window "$WINDOW_ID" "$OUTPUT" >>"$LOG" 2>&1
    CAPTURE_RC="$?"
  fi
fi
if [ "$CAPTURE_RC" != "0" ] && command -v gnome-screenshot >/dev/null 2>&1; then
  log "capturing desktop with gnome-screenshot to $OUTPUT"
  if [ "${ICGP_RVIZ_WINDOW_SCREENSHOT:-false}" = "true" ]; then
    log "window screenshot requested; this is opt-in because some desktop sessions select the wrong active window"
    gnome-screenshot -w -f "$OUTPUT" >>"$LOG" 2>&1
  else
    gnome-screenshot -f "$OUTPUT" >>"$LOG" 2>&1
  fi
  CAPTURE_RC="$?"
fi
if [ "$CAPTURE_RC" != "0" ] && command -v import >/dev/null 2>&1; then
  log "gnome-screenshot failed or unavailable (rc=$CAPTURE_RC); trying ImageMagick import"
  import -window root "$OUTPUT" >>"$LOG" 2>&1
  CAPTURE_RC="$?"
fi
set -e
log "capture_rc=$CAPTURE_RC"

kill -TERM "$RVIZ_PID" "$MARKER_PID" >/dev/null 2>&1 || true
sleep 2
kill -KILL "$RVIZ_PID" "$MARKER_PID" >/dev/null 2>&1 || true
pkill -TERM -f "rviz2|rviz_scene_marker_node" >/dev/null 2>&1 || true

if [ "$CAPTURE_RC" != "0" ]; then
  exit "$CAPTURE_RC"
fi
if [ ! -s "$OUTPUT" ]; then
  log "ERROR: screenshot file is missing or empty."
  exit 5
fi
log "OK: screenshot_bytes=$(stat -c '%s' "$OUTPUT")"

python3 - "$RUN_DIR" <<'PY'
import json
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
png = run_dir / "experiment_screenshots" / "real_rviz_gui.png"
sidecar = png.with_suffix(".json")
metadata = {
    "schema": "icgp_real_rviz_gui_screenshot_v1",
    "claim_boundary": (
        "Raw desktop screenshot of RViz during a dedicated visual-support run. "
        "RViz markers are generated online from ROS pose topics for operator explanation only. "
        "This file must not be used to compute progress, collision, safety violation, "
        "final distance, or inter-robot distance."
    ),
    "output_png": str(png),
    "run_dir": str(run_dir),
    "source": "rviz2 + icgp rviz_scene_marker_node + gnome-screenshot/import",
    "metric_source": "CSV/JSON logs and rosbag where available, not screenshot pixels",
    "required_for_metric_claim": False,
    "exists": png.exists(),
    "size_bytes": png.stat().st_size if png.exists() else 0,
}
sidecar.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
PY
