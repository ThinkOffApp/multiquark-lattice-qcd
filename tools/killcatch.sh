#!/bin/bash
# Stream macOS unified-log kill/EXC_RESOURCE/jetsam events to a file so we can
# identify what SIGKILLs the su2 measurement processes (exit 137, no crash
# report). Run inside tmux; output: /tmp/killcatch_9101.log
exec log stream --style compact \
  --predicate 'eventMessage CONTAINS[c] "kill" OR eventMessage CONTAINS[c] "EXC_RESOURCE" OR eventMessage CONTAINS[c] "memorystatus" OR eventMessage CONTAINS[c] "jetsam"' \
  > /tmp/killcatch_9101.log 2>&1
