#!/bin/bash
# Wrapper to execute AppleScript without triggering python3.14 TCC prompts.
# macOS TCC attributes Automation permission to the calling binary.
# /bin/bash already has broad automation permissions.
exec /usr/bin/osascript "$@"
