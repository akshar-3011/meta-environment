#!/usr/bin/env bash
# ─── Rollback Script ─────────────────────────────────────────────────────────
# Usage:
#   ./scripts/rollback.sh              # Roll back 1 commit
#   ./scripts/rollback.sh <commit-sha> # Roll back to specific commit
#   ./scripts/rollback.sh --dry-run    # Show what would happen
# ──────────────────────────────────────────────────────────────────────────────

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

DRY_RUN=false
TARGET=""

for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=true ;;
        *) TARGET="$arg" ;;
    esac
done

echo -e "${YELLOW}🔄 Workplace Env Rollback${NC}"
echo "─────────────────────────────────"

# Current state
CURRENT_SHA=$(git rev-parse --short HEAD)
CURRENT_MSG=$(git log -1 --format="%s")
echo -e "Current: ${GREEN}${CURRENT_SHA}${NC} — ${CURRENT_MSG}"

# Determine target
if [ -z "$TARGET" ]; then
    TARGET=$(git rev-parse --short HEAD~1)
fi
TARGET_MSG=$(git log -1 --format="%s" "$TARGET" 2>/dev/null || echo "unknown")
echo -e "Target:  ${YELLOW}${TARGET}${NC} — ${TARGET_MSG}"
echo ""

if [ "$DRY_RUN" = true ]; then
    echo -e "${YELLOW}[DRY RUN]${NC} Would reset to ${TARGET}"
    echo ""
    echo "Changes that would be reverted:"
    git log --oneline "${TARGET}..HEAD"
    exit 0
fi

# Confirm
echo -e "${RED}⚠️  This will reset HEAD to ${TARGET}.${NC}"
read -rp "Type 'yes' to confirm: " CONFIRM
if [ "$CONFIRM" != "yes" ]; then
    echo "Aborted."
    exit 1
fi

# Create safety tag
SAFETY_TAG="pre-rollback-$(date +%Y%m%d-%H%M%S)"
git tag "$SAFETY_TAG"
echo -e "Safety tag created: ${GREEN}${SAFETY_TAG}${NC}"

# Reset
git reset --hard "$TARGET"
echo -e "${GREEN}✅ Reset to ${TARGET}${NC}"

# Rebuild if Docker is available
if command -v docker &>/dev/null; then
    echo ""
    read -rp "Rebuild Docker image? (y/N): " REBUILD
    if [ "$REBUILD" = "y" ] || [ "$REBUILD" = "Y" ]; then
        SHA=$(git rev-parse --short HEAD)
        docker build -t "meta-environment:sha-${SHA}" -t meta-environment:latest .
        echo -e "${GREEN}✅ Docker image rebuilt: meta-environment:sha-${SHA}${NC}"
    fi
fi

echo ""
echo "To undo this rollback:"
echo "  git reset --hard ${SAFETY_TAG}"
