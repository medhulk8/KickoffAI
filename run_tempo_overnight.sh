#!/bin/bash

# Script to run Tempo and keep it running until task completes
# This will automatically resume at credit reset time

cd /Users/medhul/asil_project

echo "=========================================="
echo "Starting Tempo - Overnight Task Runner"
echo "=========================================="
echo "Session will auto-resume at credit reset"
echo "Press Ctrl+C to stop (not recommended)"
echo "=========================================="
echo ""

# Resume existing session or start new one
if tempo status | grep -q "Session ID"; then
    echo "Resuming existing session..."
    tempo resume --dir /Users/medhul/asil_project
else
    echo "Starting new session..."
    tempo run --file ./prompts/current_prompt.md --dir /Users/medhul/asil_project
fi

# If it exits, try to resume again (in case of temporary issues)
while true; do
    echo ""
    echo "Tempo process ended. Checking status..."
    STATUS=$(tempo status 2>&1)
    
    if echo "$STATUS" | grep -q "uncertain\|waiting"; then
        echo "Session still active. Resuming..."
        tempo resume --dir /Users/medhul/asil_project
        sleep 60  # Wait a minute before checking again
    else
        echo "Task completed or session cleared. Exiting."
        break
    fi
done

