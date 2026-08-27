#!/bin/sh
# Starts `poly03 make live run` in the background (dry-run: no orders placed,
# no network writes). Output is mirrored to paper_trade.log automatically by
# TelegramReporter; stdout is also captured here for convenience.
#
# Usage: ./run_dry_run.sh [bankroll_cap_usd]

set -eu
cd "$(dirname "$0")"

BANKROLL_CAP="${1:-500}"
STDOUT_LOG="make_run.log"
PID_FILE="make_run.pid"

if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    echo "already running (pid $(cat "$PID_FILE")) -- stop it first or check $PID_FILE" >&2
    exit 1
fi

nohup .venv/bin/poly03 make live run --bankroll-cap "$BANKROLL_CAP" > "$STDOUT_LOG" 2>&1 &
echo $! > "$PID_FILE"
disown

echo "started dry-run (pid $(cat "$PID_FILE")), bankroll_cap=\$$BANKROLL_CAP"
echo "stdout: $STDOUT_LOG"
echo "mirrored decisions: paper_trade.log"
echo "stop with: kill -9 \$(cat $PID_FILE)"
