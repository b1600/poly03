#!/bin/sh
# Talk to the box the live loop actually runs on.
#
# The bot does not run here. It runs in a tmux session on a Lightsail
# instance, and the state/log files in this repo are scp'd snapshots of it.
# That split is why both the `make live report` output and the local state
# file read as "stopped at 03:58" on 2026-09-01 while the wallet kept
# trading until 09:03 -- the copies were 5h stale and nothing said so.
#
# Everything here is read-only against the remote box except `stop`.
#
# Usage:
#   ./sync_remote.sh pull      # fetch state/logs into ./remote_snapshot/
#   ./sync_remote.sh status    # run `make live status` ON the server
#   ./sync_remote.sh report    # run `make live report` ON the server
#   ./sync_remote.sh tail      # follow the live stdout log
#   ./sync_remote.sh ps        # is the loop actually running?
#   ./sync_remote.sh stop      # graceful Ctrl-C into the tmux session
#
# Override the target with env vars if the host moves:
#   REMOTE_HOST=ubuntu@1.2.3.4 REMOTE_KEY=~/.ssh/other.pem ./sync_remote.sh pull

set -eu
cd "$(dirname "$0")"

REMOTE_HOST="${REMOTE_HOST:-ubuntu@56.69.8.154}"
REMOTE_KEY="${REMOTE_KEY:-$HOME/.ssh/Lightsail-SEA-5.pem}"
REMOTE_DIR="${REMOTE_DIR:-poly03}"
REMOTE_TMUX="${REMOTE_TMUX:-poly03}"
SNAPSHOT_DIR="${SNAPSHOT_DIR:-remote_snapshot}"

SSH="ssh -i $REMOTE_KEY -o ConnectTimeout=10 $REMOTE_HOST"

# Run a poly03 subcommand on the server, in the repo, under its venv.
remote_poly03() {
    # shellcheck disable=SC2086
    $SSH "cd $REMOTE_DIR && .venv/bin/poly03 $*"
}

usage() {
    sed -n '3,20p' "$0" | sed 's/^# \{0,1\}//'
    exit 1
}

[ $# -ge 1 ] || usage

case "$1" in
pull)
    mkdir -p "$SNAPSHOT_DIR"
    # rsync, not scp: only moves what changed, and -t preserves the remote
    # mtimes so a stale snapshot is visible in `ls -l` instead of looking
    # freshly written every time you sync.
    rsync -avt --no-perms \
        -e "ssh -i $REMOTE_KEY -o ConnectTimeout=10" \
        "$REMOTE_HOST:$REMOTE_DIR/making_live_state.json" \
        "$REMOTE_HOST:$REMOTE_DIR/making_live_decisions.jsonl" \
        "$REMOTE_HOST:$REMOTE_DIR/make_run.log" \
        "$REMOTE_HOST:$REMOTE_DIR/paper_trade.log" \
        "$SNAPSHOT_DIR/"
    echo
    echo "snapshot in $SNAPSHOT_DIR/ -- remote mtimes preserved:"
    ls -l "$SNAPSHOT_DIR/"
    echo
    echo "NOTE: these are copies. Read them with the timestamps above in mind,"
    echo "      and prefer './sync_remote.sh report' for anything decision-grade."
    ;;
status)
    remote_poly03 make live status
    ;;
report)
    remote_poly03 make live report
    ;;
tail)
    # shellcheck disable=SC2086
    $SSH "tail -f $REMOTE_DIR/make_run.log"
    ;;
ps)
    # shellcheck disable=SC2086
    if $SSH "pgrep -af 'poly03 make live run'"; then
        echo "-- loop is running"
    else
        echo "-- NO live loop running on $REMOTE_HOST"
    fi
    # shellcheck disable=SC2086
    $SSH "tmux ls 2>/dev/null || echo '(no tmux sessions)'"
    ;;
stop)
    # SIGINT into the tmux session, not `kill -9`: the loop's Ctrl-C path
    # cancels every tracked resting order and prints a final report. Killing
    # it outright leaves orders resting with nothing watching them.
    # shellcheck disable=SC2086
    $SSH "tmux send-keys -t $REMOTE_TMUX C-c" \
        && echo "sent Ctrl-C to tmux session '$REMOTE_TMUX' on $REMOTE_HOST" \
        && echo "check it landed: ./sync_remote.sh ps"
    ;;
*)
    usage
    ;;
esac
