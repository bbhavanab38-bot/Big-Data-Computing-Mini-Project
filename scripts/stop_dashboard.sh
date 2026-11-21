#!/usr/bin/env bash
# ==============================================================================
# Stop all BDT Dashboard & Service Processes
# ==============================================================================

set -eo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

GREEN="\033[92m"
YELLOW="\033[93m"
CYAN="\033[96m"
BOLD="\033[1m"
RESET="\033[0m"

echo -e "${BOLD}${CYAN}Stopping BDT Platform Services...${RESET}"

# Stop Streamlit on port 8501
STREAMLIT_PIDS="$(lsof -ti :8501 2>/dev/null || true)"
if [ -n "$STREAMLIT_PIDS" ]; then
    echo -e "  ${YELLOW}Stopping Streamlit dashboard on port 8501 (PID: $STREAMLIT_PIDS)...${RESET}"
    kill -TERM $STREAMLIT_PIDS 2>/dev/null || kill -9 $STREAMLIT_PIDS 2>/dev/null || true
    echo -e "  ${GREEN}✔ Streamlit dashboard stopped.${RESET}"
else
    echo -e "  ${GREEN}✔ No process active on port 8501.${RESET}"
fi

# Stop FastAPI on port 8000
API_PIDS="$(lsof -ti :8000 2>/dev/null || true)"
if [ -n "$API_PIDS" ]; then
    echo -e "  ${YELLOW}Stopping FastAPI microservice on port 8000 (PID: $API_PIDS)...${RESET}"
    kill -TERM $API_PIDS 2>/dev/null || kill -9 $API_PIDS 2>/dev/null || true
    echo -e "  ${GREEN}✔ FastAPI microservice stopped.${RESET}"
else
    echo -e "  ${GREEN}✔ No process active on port 8000.${RESET}"
fi

echo -e "${BOLD}${GREEN}All BDT platform processes stopped cleanly.${RESET}"
