#!/bin/bash
# OllApeX control for the systemd service "ollapex":  ./ollapex.sh start | stop | restart | status | log [lines]
HERE="$(cd "$(dirname "$0")" && pwd)"
PORT="${OLLAPEX_PORT:-3000}"
case "${1:-status}" in
start|stop|restart)
    sudo systemctl "$1" ollapex
    systemctl is-active ollapex
    ;;
status)
    echo "service: $(systemctl is-active ollapex) (at boot: $(systemctl is-enabled ollapex 2>/dev/null))"
    curl -s -m 3 "http://127.0.0.1:$PORT/api/status" | python3 -c "
import json, sys
try:
    s = json.load(sys.stdin)
except Exception:
    print('the interface is not responding'); sys.exit(0)
p = s.get('progress') or {}
print('state:', s['state'], '| model:', s.get('model'), '| context:', s.get('num_ctx'),
      ('| %s/%s cards, %d s' % (p.get('ready'), p.get('cards'), p.get('elapsed_s', 0))) if p else '',
      ('| error: ' + s['error'][:300]) if s.get('error') else '')
"
    ;;
log)
    tail -n "${2:-40}" "$HERE/logs/server.log"
    ;;
esac
