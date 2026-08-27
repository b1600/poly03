#!/bin/sh
# Stops the loop started by run_dry_run.sh.
# Tries a graceful SIGINT first (runs the loop's cancel-tracked-orders +
# final report), falls back to SIGKILL if it hasn't exited after a few
# seconds.

set -eu
cd "$(dirname "$0")"

PID_FILE="make_run.pid"

if [ ! -f "$PID_FILE" ]; then
    echo "no $PID_FILE -- nothing to stop" >&2
    exit 1
fi

PID="$(cat "$PID_FILE")"

if ! kill -0 "$PID" 2>/dev/null; then
    echo "pid $PID not running"
    rm -f "$PID_FILE"
    exit 0
fi

kill -INT "$PID" 2>/dev/null || true
for _ in 1 2 3 4 5; do
    kill -0 "$PID" 2>/dev/null || { echo "stopped (pid $PID)"; rm -f "$PID_FILE"; exit 0; }
    sleep 1
done

echo "graceful stop didn't take -- sending SIGKILL" >&2
kill -9 "$PID" 2>/dev/null || true
rm -f "$PID_FILE"
echo "killed (pid $PID)"
