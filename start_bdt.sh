#!/usr/bin/env bash
# ==============================================================================
# BDT (Big Data Computing) Mini Project — Production Startup Script
# ==============================================================================
# Ensures complete reproducibility, virtual environment isolation, dependency
# pinning (scikit-learn 1.9.1), system health validation, and dashboard launch.
#
# Usage:
#   ./start_bdt.sh
# ==============================================================================

set -eo pipefail

# 1. Navigate to Project Directory
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

# ANSI Colors
GREEN="\033[92m"
RED="\033[91m"
YELLOW="\033[93m"
CYAN="\033[96m"
BOLD="\033[1m"
RESET="\033[0m"

echo ""
echo -e "${BOLD}${CYAN}=================================================================${RESET}"
echo -e "${BOLD}${CYAN}   BANKING RETENTION INTELLIGENCE PLATFORM (BDT 2.0)            ${RESET}"
echo -e "${BOLD}${CYAN}   Automated Environment Verification & Dashboard Startup        ${RESET}"
echo -e "${BOLD}${CYAN}=================================================================${RESET}"
echo ""

# 2. Locate Best Compatible Python Interpreter
find_python_base() {
    if command -v python3.12 >/dev/null 2>&1; then
        echo "python3.12"
    elif command -v python3.11 >/dev/null 2>&1; then
        echo "python3.11"
    elif command -v python3 >/dev/null 2>&1; then
        echo "python3"
    else
        echo ""
    fi
}

BASE_PYTHON="$(find_python_base)"
if [ -z "$BASE_PYTHON" ]; then
    echo -e "${RED}[ERROR]${RESET} No Python 3 interpreter found in PATH."
    echo -e "        Please install Python 3.12 via Homebrew: ${YELLOW}brew install python@3.12${RESET}"
    exit 1
fi

# 3. Virtual Environment Check & Creation
VENV_DIR="$PROJECT_DIR/.venv"
VENV_PYTHON="$VENV_DIR/bin/python"
VENV_ACTIVATE="$VENV_DIR/bin/activate"

if [ ! -d "$VENV_DIR" ] || [ ! -f "$VENV_PYTHON" ] || [ ! -f "$VENV_ACTIVATE" ]; then
    echo -e "${YELLOW}[SETUP]${RESET} Project virtual environment (.venv) not found. Creating fresh .venv using ${BASE_PYTHON}..."
    "$BASE_PYTHON" -m venv "$VENV_DIR"
    echo -e "${GREEN}[SETUP]${RESET} Virtual environment created successfully at ${VENV_DIR}."
fi

# 4. Activate Virtual Environment
# shellcheck disable=SC1090
source "$VENV_ACTIVATE"

# Verify active python belongs strictly to .venv
ACTIVE_PYTHON="$(which python)"
if [[ "$ACTIVE_PYTHON" != "$VENV_DIR"* ]]; then
    echo -e "${RED}[ERROR]${RESET} Failed to activate virtual environment. Current python: ${ACTIVE_PYTHON}"
    echo -e "        Expected path: ${VENV_DIR}/bin/python"
    exit 1
fi

echo -e "  ${GREEN}✔${RESET} Virtual Environment Active: ${CYAN}${ACTIVE_PYTHON}${RESET} ($(python --version))"

# 5. Dependency Verification & Installation (Smart Cache)
REQUIREMENTS_FILE="$PROJECT_DIR/requirements.txt"
HASH_FILE="$VENV_DIR/.requirements_installed_hash"

compute_hash() {
    if command -v shasum >/dev/null 2>&1; then
        shasum -a 256 "$REQUIREMENTS_FILE" | awk '{print $1}'
    elif command -v md5 >/dev/null 2>&1; then
        md5 -q "$REQUIREMENTS_FILE"
    else
        cat "$REQUIREMENTS_FILE" | wc -c
    fi
}

CURRENT_HASH="$(compute_hash)"
NEEDS_INSTALL=false

if [ ! -f "$HASH_FILE" ]; then
    NEEDS_INSTALL=true
else
    STORED_HASH="$(cat "$HASH_FILE")"
    if [ "$CURRENT_HASH" != "$STORED_HASH" ]; then
        NEEDS_INSTALL=true
    fi
fi

# Also check if critical dependencies can be imported
if ! python -c "import sklearn, pandas, numpy, streamlit, pyspark" >/dev/null 2>&1; then
    NEEDS_INSTALL=true
fi

if [ "$NEEDS_INSTALL" = true ]; then
    echo -e "${YELLOW}[SETUP]${RESET} Installing / updating dependencies from requirements.txt into .venv..."
    python -m pip install --upgrade pip
    python -m pip install -r "$REQUIREMENTS_FILE"
    python -m pip install -e .
    echo "$CURRENT_HASH" > "$HASH_FILE"
    echo -e "${GREEN}[SETUP]${RESET} All requirements successfully installed in .venv."
else
    echo -e "  ${GREEN}✔${RESET} Dependencies up-to-date (matching requirements.txt lockfile)."
fi

# 6. Verify Scikit-Learn Version Compatibility
SKLEARN_VERSION="$(python -c "import sklearn; print(sklearn.__version__)" 2>/dev/null || echo "MISSING")"
if [ "$SKLEARN_VERSION" != "1.9.1" ]; then
    echo -e "${RED}[ERROR]${RESET} Scikit-learn version mismatch!"
    echo -e "        Installed in .venv: ${SKLEARN_VERSION}"
    echo -e "        Required by models: 1.9.1"
    echo -e "        Fixing version mismatch now..."
    python -m pip install scikit-learn==1.9.1
fi
echo -e "  ${GREEN}✔${RESET} Scikit-Learn Version Verified: ${CYAN}v1.9.1${RESET} (exact match with saved model artifacts)"

# 7. Check Essential Data & Model Files
REQUIRED_FILES=(
    "data/raw/bank_churn.csv"
    "data/processed/processed_bank_data.csv"
    "data/processed/test_split.csv"
    "data/processed/scored_customers.csv"
    "models/best_model.pkl"
    "models/registry.json"
    "dashboard/platform_app.py"
)

MISSING_COUNT=0
for f in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$PROJECT_DIR/$f" ]; then
        echo -e "  ${RED}✖ MISSING FILE:${RESET} $f"
        MISSING_COUNT=$((MISSING_COUNT + 1))
    fi
done

if [ "$MISSING_COUNT" -gt 0 ]; then
    echo -e "${RED}[ERROR]${RESET} $MISSING_COUNT required project file(s) are missing."
    echo -e "        Run preprocessing & training to regenerate: ${YELLOW}python src/train.py${RESET}"
    exit 1
fi
echo -e "  ${GREEN}✔${RESET} Essential project datasets and model weights verified."

# 8. Run Full System Health Check
echo ""
echo -e "${BOLD}Running Diagnostic Health Check...${RESET}"
if ! python scripts/health_check.py --quick; then
    echo -e "${RED}[FATAL]${RESET} System health check failed. Review the diagnostics above and resolve errors before starting."
    exit 1
fi

# 9. Check Port Availability & Stop Stale Processes on 8501
PORT=8501
EXISTING_PID="$(lsof -ti :$PORT 2>/dev/null || true)"
if [ -n "$EXISTING_PID" ]; then
    echo -e "${YELLOW}[NOTICE]${RESET} Port $PORT is currently occupied by PID: $EXISTING_PID"
    echo -e "         Stopping existing process on port $PORT to ensure clean startup..."
    kill -9 $EXISTING_PID 2>/dev/null || true
    sleep 1
fi

# 10. Start Streamlit Platform Dashboard
echo -e "${BOLD}${GREEN}=================================================================${RESET}"
echo -e "${BOLD}${GREEN}  STARTING RETENTION IQ ENTERPRISE DASHBOARD                    ${RESET}"
echo -e "${BOLD}${GREEN}=================================================================${RESET}"
echo -e "  Platform URL: ${BOLD}${CYAN}http://localhost:${PORT}${RESET}"
echo -e "  Platform App: ${CYAN}dashboard/platform_app.py${RESET}"
echo -e "  Press ${YELLOW}Ctrl+C${RESET} in this terminal to stop the dashboard anytime."
echo ""

# Launch Streamlit using python -m streamlit to guarantee .venv python interpreter
exec python -m streamlit run dashboard/platform_app.py --server.port "$PORT" --server.headless false
