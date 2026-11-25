#!/bin/bash
# Long-context retrieval check on Qwen3.5-9B: one fact, many filler sentences, then the question.
# usage: tools/needle.sh <n_filler_sentences> <max_context> <tag>
set -eu
cd "$(dirname "$0")/.."
export Q_MODEL=q35_9b
python_bin=${PYTHON:-python3}
log="$HOME/needle_${3}.log"
PROMPT=$("$python_bin" - "$1" <<'PY'
import random, sys
random.seed(3)
words = "river stone market morning window garden letter bridge forest candle harbor winter village music lantern road".split()
filler = ["The %s near the %s was %s, and people walked past the %s without stopping." % tuple(random.choice(words) for _ in range(4))
          for _ in range(int(sys.argv[1]))]
print("Note: the secret code of the warehouse is PURPLE-4721. " + " ".join(filler)
      + " Question: what is the secret code of the warehouse? Answer with the code only.")
PY
)
status=0
timeout 7200 "$python_bin" run_tp.py --tp 10fp4 --engines 2 --nap-us 30 --max-context "$2" \
    --max-new-tokens 16 --prompt "$PROMPT" > "$log" 2>&1 || status=$?
echo "exit=$status"
grep -E 'ERROR|Error|PROMPT|PREFILL|DECODE|SEGMENT\|full att' "$log" | cut -c1-160
sed -n '/TEXT_BEGIN/,/TEXT_END/p' "$log"
exit "$status"
