#!/usr/bin/env bash
# Install ptt-dictate as a login daemon (LaunchAgent): warm from login, with no
# terminal pane to babysit.
#
#   ./install.sh                                          # defaults to --key right_option
#   ./install.sh --key f13 --context "Kubernetes, Postgres, Terraform"
#
# The plist is generated from this repo's own location and $HOME, so no
# machine-specific paths are ever committed.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LABEL=local.ptt-dictate
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
LOG="$HOME/Library/Logs/ptt-dictate/daemon.log"
PYTHON="${PTT_PYTHON:-$HOME/models/venv-mlx-audio/bin/python}"

if [[ ! -x "$PYTHON" ]]; then
  echo "interpreter not found: $PYTHON" >&2
  echo "point PTT_PYTHON at a python with mlx-audio + pyobjc-framework-Quartz installed" >&2
  exit 1
fi

# `launchctl bootstrap` cannot expand ~, so the plist needs real paths; the
# defaults live here rather than in a committed file.
if [[ $# -eq 0 ]]; then
  set -- --key right_option
fi

# Hotwords live in a file the script re-reads on every press, so adding a term
# takes effect on the next dictation instead of needing a reinstall.
HOTWORDS="${PTT_HOTWORDS:-$HOME/.config/ptt-dictate/hotwords.txt}"
mkdir -p "$(dirname "$HOTWORDS")"
if [[ ! -f "$HOTWORDS" ]]; then
  {
    echo '# ptt-dictate hotwords — names and jargon the model tends to mis-hear.'
    echo '# One per line (or a comma-separated list); # starts a comment.'
    echo '# Re-read on every press: edits apply to your next dictation, no restart.'
    ctx=; prev=
    for a in "$@"; do
      [[ "$prev" == "--context" ]] && ctx="$a"
      prev="$a"
    done
    if [[ -n "$ctx" ]]; then
      printf '%s\n' "$ctx" | tr ',' '\n' | sed 's/^ *//; s/ *$//' | grep -v '^$' || true
    fi
  } > "$HOTWORDS"
  echo "seeded $HOTWORDS"
fi

mkdir -p "$(dirname "$LOG")"
{
  echo '<?xml version="1.0" encoding="UTF-8"?>'
  echo '<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">'
  echo '<plist version="1.0"><dict>'
  echo "  <key>Label</key><string>$LABEL</string>"
  echo '  <key>ProgramArguments</key><array>'
  echo "    <string>$PYTHON</string>"
  echo "    <string>$REPO/ptt_dictate.py</string>"
  for arg in "$@" --context-file "$HOTWORDS"; do
    echo "    <string>$arg</string>"
  done
  echo '  </array>'
  echo '  <key>RunAtLoad</key><true/>'
  echo '  <key>KeepAlive</key><true/>'
  # latency-sensitive: keep it out of App Nap / background QoS throttling
  echo '  <key>ProcessType</key><string>Interactive</string>'
  echo '  <key>LimitLoadToSessionType</key><string>Aqua</string>'
  echo "  <key>StandardOutPath</key><string>$LOG</string>"
  echo "  <key>StandardErrorPath</key><string>$LOG</string>"
  echo '</dict></plist>'
} > "$PLIST"

plutil -lint "$PLIST" >/dev/null
launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true

# bootout is asynchronous, and bootstrapping the label too soon fails with
# "Bootstrap failed: 5: Input/output error" — which leaves dictation down.
# Retry until launchd has actually let go of the label.
booted=
for _ in $(seq 10); do
  if launchctl bootstrap "gui/$(id -u)" "$PLIST" 2>/dev/null; then
    booted=yes
    break
  fi
  sleep 1
done
if [[ -z "$booted" ]]; then
  echo "launchctl bootstrap failed 10x — is the label stuck?" >&2
  echo "try: launchctl bootout gui/$(id -u)/$LABEL && launchctl bootstrap gui/$(id -u) $PLIST" >&2
  exit 1
fi

echo "installed $LABEL"
echo "  plist: $PLIST"
echo "  log:   $LOG"
echo "  hotwords: $HOTWORDS  (edit any time; applies to the next dictation)"
echo "  first load takes ~15s; then hold your hotkey and speak"
