#!/usr/bin/env bash
# Stop and remove the ptt-dictate LaunchAgent. The logs and the model are left
# alone; delete ~/Library/Logs/ptt-dictate yourself if you want them gone.
set -euo pipefail

LABEL=local.ptt-dictate
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"

launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
if [[ -f "$PLIST" ]]; then
  rm -f "$PLIST"
  echo "removed $PLIST"
else
  echo "nothing to do ($PLIST not present)"
fi
