#!/bin/zsh
set -euo pipefail

ARCH=$(uname -m)
case "$ARCH" in
  arm64) ASSET='cloudflared-darwin-arm64.tgz' ;;
  x86_64) ASSET='cloudflared-darwin-amd64.tgz' ;;
  *) echo "Unsupported arch: $ARCH" >&2; exit 1 ;;
esac

TARGET_DIR="$HOME/.local/bin"
TMP_DIR=$(mktemp -d)
trap 'rm -rf "$TMP_DIR"' EXIT
mkdir -p "$TARGET_DIR"

URL="https://github.com/cloudflare/cloudflared/releases/latest/download/${ASSET}"
echo "Downloading $URL"
curl -L "$URL" -o "$TMP_DIR/cloudflared.tgz"
tar -xzf "$TMP_DIR/cloudflared.tgz" -C "$TMP_DIR"
install -m 755 "$TMP_DIR/cloudflared" "$TARGET_DIR/cloudflared"

echo "Installed: $TARGET_DIR/cloudflared"
echo "You may want to add to PATH: export PATH=\"$TARGET_DIR:\$PATH\""
