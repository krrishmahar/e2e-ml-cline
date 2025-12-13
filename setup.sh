#!/bin/bash

# ============================================================================
# SETUP SCRIPT FOR NODE 24, CLINE, KESTRA, AND CODERABBIT
# Comprehensive installation with logging, error handling, and validation
# ============================================================================

set -euo pipefail

# Configuration
DRY_RUN=false
VERBOSE=false
SKIP_EXISTING=false
FORCE_REINSTALL=false
LOG_FILE="/tmp/setup_${USER}_$(date +%Y%m%d_%H%M%S).log"
TEMP_DIR="/tmp/setup_$$"
INSTALL_DIR="$HOME/.setup_tools"

# Exit codes
EXIT_SUCCESS=0
EXIT_DEPENDENCY_MISSING=1
EXIT_INSTALLATION_FAILED=2
EXIT_VALIDATION_FAILED=3
EXIT_PERMISSION_DENIED=4
EXIT_SYSTEM_REQUIREMENT=5

# Colors
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly CYAN='\033[0;36m'
readonly NC='\033[0m' # No Color

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

# Enhanced logging with colors and file output
log() {
    local level=$1
    shift
    local message="$*"
    local timestamp=$(date +"%Y-%m-%d %H:%M:%S")
    local color=""

    case "$level" in
        INFO)    color="$CYAN" ;;
        SUCCESS) color="$GREEN" ;;
        WARNING) color="$YELLOW" ;;
        ERROR)   color="$RED" ;;
        DEBUG)   color="$BLUE" ;;
        *)       color="$NC" ;;
    esac

    # Console output with color
    echo -e "${color}[$timestamp] [$level]${NC} $message" >&2

    # File output without color
    echo "[$timestamp] [$level] $message" >> "$LOG_FILE" 2>/dev/null || true
}

# Progress spinner
spinner() {
    local pid=$1
    local delay=0.1
    local spinstr='⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏'
    local temp

    while ps -p "$pid" > /dev/null 2>&1; do
        temp=${spinstr#?}
        printf " [%c]  " "$spinstr"
        spinstr=$temp${spinstr%"$temp"}
        sleep $delay
        printf "\b\b\b\b\b\b"
    done
    printf "    \b\b\b\b"
}

# Enhanced command runner with progress indication and output capture
run_command() {
    local description="$1"
    local log_output="${2:-true}"
    shift 2

    local cmd_output_file="$TEMP_DIR/cmd_output_$$.log"
    local cmd_error_file="$TEMP_DIR/cmd_error_$$.log"

    if [ "$DRY_RUN" = true ]; then
        log "INFO" "[DRY-RUN] Would run: $description"
        log "DEBUG" "[DRY-RUN] Command: $*"
        return 0
    fi

    log "INFO" "Running: $description..."

    if [ "$VERBOSE" = true ]; then
        # Verbose mode: show output in real-time
        "$@" 2>&1 | tee -a "$LOG_FILE"
        local exit_code=${PIPESTATUS[0]}
    else
        # Non-verbose: capture output and show spinner
        "$@" > "$cmd_output_file" 2> "$cmd_error_file" &
        local cmd_pid=$!
        spinner $cmd_pid
        wait $cmd_pid
        local exit_code=$?

        # Log output to file
        if [ "$log_output" = "true" ]; then
            cat "$cmd_output_file" >> "$LOG_FILE" 2>/dev/null || true
            cat "$cmd_error_file" >> "$LOG_FILE" 2>/dev/null || true
        fi
    fi

    if [ $exit_code -ne 0 ]; then
        log "ERROR" "$description failed with exit code $exit_code"

        # Show error output even in non-verbose mode
        if [ "$VERBOSE" = false ] && [ -s "$cmd_error_file" ]; then
            log "ERROR" "Error output:"
            cat "$cmd_error_file" >&2
        fi

        # Cleanup temp files
        rm -f "$cmd_output_file" "$cmd_error_file"
        return $exit_code
    fi

    log "SUCCESS" "$description completed"

    # Cleanup temp files
    rm -f "$cmd_output_file" "$cmd_error_file"
    return 0
}

# ============================================================================
# SETUP AND VALIDATION FUNCTIONS
# ============================================================================

# Parse command line arguments
parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --dry-run|-n)
                DRY_RUN=true
                shift
                ;;
            --verbose|-v)
                VERBOSE=true
                shift
                ;;
            --skip-existing|-s)
                SKIP_EXISTING=true
                shift
                ;;
            --force|-f)
                FORCE_REINSTALL=true
                shift
                ;;
            --help|-h)
                show_help
                exit 0
                ;;
            *)
                log "ERROR" "Unknown option: $1"
                echo "Use --help for usage information"
                exit 1
                ;;
        esac
    done
}

# Show help message
show_help() {
    cat << EOF
Usage: $0 [options]

Setup script for Node 24, Cline, Kestra, and Coderabbit installation.

Options:
  -n, --dry-run            Dry run mode (no actual installation)
  -v, --verbose            Verbose output (show all command output)
  -s, --skip-existing      Skip installation if tools are already installed
  -f, --force              Force reinstallation of all tools
  -h, --help               Show this help message

Tools to be installed:
  - Node.js 24.x (via NVM)
  - Cline (AI-powered code improvement tool)
  - Kestra (workflow orchestration)
  - Coderabbit (code review automation)

Requirements:
  - Linux/macOS system
  - curl (for downloading)
  - git (for version control)
  - Docker (for Kestra installation)
  - sudo privileges (for system-wide installations)

Output:
  - Log file: $LOG_FILE
  - Installation directory: $INSTALL_DIR

Exit Codes:
  0 - Success
  1 - Dependency missing
  2 - Installation failed
  3 - Validation failed
  4 - Permission denied
  5 - System requirement not met

EOF
}

# Setup temporary directory
setup_temp_dir() {
    mkdir -p "$TEMP_DIR" || {
        log "ERROR" "Failed to create temporary directory: $TEMP_DIR"
        exit 1
    }

    # Ensure cleanup on exit
    trap cleanup EXIT INT TERM

    log "DEBUG" "Temporary directory: $TEMP_DIR"
    log "DEBUG" "Log file: $LOG_FILE"
}

# Cleanup function
cleanup() {
    local exit_code=$?

    log "DEBUG" "Cleaning up..."

    # Remove temporary files
    if [ -d "$TEMP_DIR" ]; then
        rm -rf "$TEMP_DIR"
        log "DEBUG" "Removed temporary directory"
    fi

    if [ $exit_code -eq 0 ]; then
        log "SUCCESS" "Cleanup completed"
    else
        log "WARNING" "Cleanup completed with errors (exit code: $exit_code)"
    fi
}

# Validate system requirements
validate_system() {
    log "INFO" "Validating system requirements..."

    # Check OS
    local os=$(uname -s)
    if [[ "$os" != "Linux" && "$os" != "Darwin" ]]; then
        log "ERROR" "Unsupported operating system: $os"
        log "INFO" "This script supports Linux and macOS only"
        exit $EXIT_SYSTEM_REQUIREMENT
    fi

    # Check for required commands
    local required_commands=("curl" "git" "bash")
    local missing_commands=()

    for cmd in "${required_commands[@]}"; do
        if ! command -v "$cmd" &> /dev/null; then
            missing_commands+=("$cmd")
        fi
    done

    if [ ${#missing_commands[@]} -gt 0 ]; then
        log "ERROR" "Missing required commands: ${missing_commands[*]}"
        log "INFO" "Please install these commands before running the setup"
        exit $EXIT_DEPENDENCY_MISSING
    fi

    # Check for Docker (needed for Kestra)
    if ! command -v docker &> /dev/null; then
        log "WARNING" "Docker not found - Kestra installation will be skipped"
        log "INFO" "Install Docker to enable Kestra setup"
    fi

    log "SUCCESS" "System validation passed"
}

# Check if tool is already installed
is_installed() {
    local tool_name=$1
    local version_check=${2:-}

    if command -v "$tool_name" &> /dev/null; then
        if [ -n "$version_check" ]; then
            local version=$($tool_name --version 2>/dev/null || echo "")
            if [[ "$version" =~ $version_check ]]; then
                log "INFO" "$tool_name is already installed: $version"
                return 0
            else
                log "INFO" "$tool_name is installed but version doesn't match: $version"
                return 1
            fi
        else
            log "INFO" "$tool_name is already installed"
            return 0
        fi
    fi

    return 1
}

# ============================================================================
# INSTALLATION FUNCTIONS
# ============================================================================

# Install NVM (Node Version Manager)
install_nvm() {
    if [ "$SKIP_EXISTING" = true ] && is_installed "nvm"; then
        log "INFO" "Skipping NVM installation (already installed)"
        return 0
    fi

    if [ "$FORCE_REINSTALL" = false ] && is_installed "nvm"; then
        log "INFO" "NVM is already installed"
        return 0
    fi

    log "INFO" "🔧 Installing NVM (Node Version Manager)..."

    # Install NVM
    run_command "NVM installation" true curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash || {
        log "ERROR" "Failed to install NVM"
        return $EXIT_INSTALLATION_FAILED
    }

    # Source NVM in current shell
    export NVM_DIR="$HOME/.nvm"
    [ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh" || {
        log "ERROR" "Failed to source NVM"
        return $EXIT_INSTALLATION_FAILED
    }

    log "SUCCESS" "NVM installed successfully"
    return 0
}

# Install Node.js 24.x
install_node_24() {
    if [ "$SKIP_EXISTING" = true ] && is_installed "node" "v24\."; then
        log "INFO" "Skipping Node 24 installation (already installed)"
        return 0
    fi

    if [ "$FORCE_REINSTALL" = false ] && is_installed "node" "v24\."; then
        log "INFO" "Node 24 is already installed"
        return 0
    fi

    log "INFO" "🔧 Installing Node.js 24.x..."

    # Source NVM
    export NVM_DIR="$HOME/.nvm"
    [ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh" || {
        log "ERROR" "NVM not found - install NVM first"
        return $EXIT_DEPENDENCY_MISSING
    }

    # Install Node 24
    run_command "Node 24 installation" true nvm install 24 || {
        log "ERROR" "Failed to install Node 24"
        return $EXIT_INSTALLATION_FAILED
    }

    # Use Node 24
    run_command "Node 24 activation" true nvm use 24 || {
        log "ERROR" "Failed to activate Node 24"
        return $EXIT_INSTALLATION_FAILED
    }

    # Set default Node version
    run_command "Node 24 default" true nvm alias default 24 || {
        log "WARNING" "Failed to set Node 24 as default"
    }

    log "SUCCESS" "Node 24 installed and activated"
    return 0
}

# Install Cline
install_cline() {
    if [ "$SKIP_EXISTING" = true ] && is_installed "cline"; then
        log "INFO" "Skipping Cline installation (already installed)"
        return 0
    fi

    if [ "$FORCE_REINSTALL" = false ] && is_installed "cline"; then
        log "INFO" "Cline is already installed"
        return 0
    fi

    log "INFO" "🔧 Installing Cline..."

    # Install Cline globally
    run_command "Cline installation" true npm install -g cline || {
        log "ERROR" "Failed to install Cline"
        return $EXIT_INSTALLATION_FAILED
    }

    log "SUCCESS" "Cline installed successfully"
    return 0
}

# Install Kestra (Docker-based)
install_kestra() {
    if ! command -v docker &> /dev/null; then
        log "WARNING" "Docker not found - skipping Kestra installation"
        return 0
    fi

    if [ "$SKIP_EXISTING" = true ] && docker ps -a | grep -q kestra; then
        log "INFO" "Skipping Kestra installation (already installed)"
        return 0
    fi

    if [ "$FORCE_REINSTALL" = false ] && docker ps -a | grep -q kestra; then
        log "INFO" "Kestra is already installed"
        return 0
    fi

    log "INFO" "🔧 Installing Kestra (Docker-based)..."

    # Create Kestra directory
    mkdir -p "$HOME/kestra" || {
        log "ERROR" "Failed to create Kestra directory"
        return $EXIT_INSTALLATION_FAILED
    }

    # Create docker-compose.yml for Kestra
    local compose_file="$HOME/kestra/docker-compose.yml"
    cat > "$compose_file" << 'EOF'
version: '3.8'

services:
  kestra:
    image: kestra/kestra:latest
    container_name: kestra
    ports:
      - "8080:8080"
    volumes:
      - ./kestra-data:/app/storage
    environment:
      - KESTRA_CONFIGURATION=file:///app/application.yml
    restart: unless-stopped

  postgres:
    image: postgres:13
    container_name: kestra-postgres
    environment:
      POSTGRES_USER: kestra
      POSTGRES_PASSWORD: kestra
      POSTGRES_DB: kestra
    volumes:
      - ./postgres-data:/var/lib/postgresql/data
    restart: unless-stopped
EOF

    # Start Kestra
    run_command "Kestra startup" true docker-compose -f "$compose_file" up -d || {
        log "ERROR" "Failed to start Kestra"
        return $EXIT_INSTALLATION_FAILED
    }

    log "SUCCESS" "Kestra installed and started"
    log "INFO" "Kestra UI will be available at http://localhost:8080"
    return 0
}

# Install Coderabbit
install_coderabbit() {
    if [ "$SKIP_EXISTING" = true ] && is_installed "coderabbit"; then
        log "INFO" "Skipping Coderabbit installation (already installed)"
        return 0
    fi

    if [ "$FORCE_REINSTALL" = false ] && is_installed "coderabbit"; then
        log "INFO" "Coderabbit is already installed"
        return 0
    fi

    log "INFO" "🔧 Installing Coderabbit..."

    # Install Coderabbit globally
    run_command "Coderabbit installation" true npm install -g coderabbit || {
        log "ERROR" "Failed to install Coderabbit"
        return $EXIT_INSTALLATION_FAILED
    }

    log "SUCCESS" "Coderabbit installed successfully"
    return 0
}

# ============================================================================
# VALIDATION FUNCTIONS
# ============================================================================

# Validate Node installation
validate_node() {
    log "INFO" "🔍 Validating Node installation..."

    if ! command -v node &> /dev/null; then
        log "ERROR" "Node.js is not installed"
        return $EXIT_VALIDATION_FAILED
    fi

    local version=$(node --version)
    if [[ ! "$version" =~ v24\..* ]]; then
        log "ERROR" "Node version is not 24.x: $version"
        return $EXIT_VALIDATION_FAILED
    fi

    # Test npm
    if ! command -v npm &> /dev/null; then
        log "ERROR" "npm is not available"
        return $EXIT_VALIDATION_FAILED
    fi

    log "SUCCESS" "Node 24 validation passed: $version"
    return 0
}

# Validate Cline installation
validate_cline() {
    log "INFO" "🔍 Validating Cline installation..."

    if ! command -v cline &> /dev/null; then
        log "ERROR" "Cline is not installed"
        return $EXIT_VALIDATION_FAILED
    fi

    local version=$(cline --version 2>/dev/null || echo "unknown")
    log "SUCCESS" "Cline validation passed: $version"

    # Test basic functionality
    if [ "$DRY_RUN" = false ]; then
        local test_output=$(cline --help 2>&1 | head -1 || echo "")
        if [[ -z "$test_output" ]]; then
            log "WARNING" "Cline help command returned empty output"
        else
            log "DEBUG" "Cline help output: $test_output"
        fi
    fi

    return 0
}

# Validate Kestra installation
validate_kestra() {
    if ! command -v docker &> /dev/null; then
        log "INFO" "Docker not available - skipping Kestra validation"
        return 0
    fi

    log "INFO" "🔍 Validating Kestra installation..."

    if ! docker ps -a | grep -q kestra; then
        log "ERROR" "Kestra container not found"
        return $EXIT_VALIDATION_FAILED
    fi

    local container_status=$(docker inspect -f '{{.State.Status}}' kestra 2>/dev/null || echo "unknown")
    if [ "$container_status" != "running" ]; then
        log "WARNING" "Kestra container is not running: $container_status"
        return 0
    fi

    log "SUCCESS" "Kestra validation passed"
    log "INFO" "Kestra container status: $container_status"
    return 0
}

# Validate Coderabbit installation
validate_coderabbit() {
    log "INFO" "🔍 Validating Coderabbit installation..."

    if ! command -v coderabbit &> /dev/null; then
        log "ERROR" "Coderabbit is not installed"
        return $EXIT_VALIDATION_FAILED
    fi

    local version=$(coderabbit --version 2>/dev/null || echo "unknown")
    log "SUCCESS" "Coderabbit validation passed: $version"

    # Test basic functionality
    if [ "$DRY_RUN" = false ]; then
        local test_output=$(coderabbit --help 2>&1 | head -1 || echo "")
        if [[ -z "$test_output" ]]; then
            log "WARNING" "Coderabbit help command returned empty output"
        else
            log "DEBUG" "Coderabbit help output: $test_output"
        fi
    fi

    return 0
}

# ============================================================================
# MAIN WORKFLOW
# ============================================================================

main() {
    # Print banner
    echo -e "${CYAN}"
    echo "╔════════════════════════════════════════════════════════════╗"
    echo "║                                                            ║"
    echo "║        Setup Script for Node 24, Cline, Kestra,          ║"
    echo "║              and Coderabbit Installation                ║"
    echo "║                                                            ║"
    echo "╚════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
    echo

    # Setup
    setup_temp_dir
    parse_arguments "$@"

    log "INFO" "Setup started (PID: $$)"
    log "INFO" "Mode: $([ "$DRY_RUN" = true ] && echo "DRY-RUN" || echo "LIVE")"
    echo

    # Pre-flight checks
    validate_system

    echo
    log "INFO" "Starting installation workflow..."
    echo

    # Execute installation steps
    local installation_failed=false

    # Step 1: Install NVM
    install_nvm || {
        log "ERROR" "NVM installation failed"
        installation_failed=true
    }

    # Step 2: Install Node 24
    if [ "$installation_failed" = false ]; then
        install_node_24 || {
            log "ERROR" "Node 24 installation failed"
            installation_failed=true
        }
    fi

    # Step 3: Install Cline
    if [ "$installation_failed" = false ]; then
        install_cline || {
            log "ERROR" "Cline installation failed"
            installation_failed=true
        }
    fi

    # Step 4: Install Kestra
    if [ "$installation_failed" = false ]; then
        install_kestra || {
            log "WARNING" "Kestra installation had issues, but continuing"
            # Don't fail workflow for Kestra
        }
    fi

    # Step 5: Install Coderabbit
    if [ "$installation_failed" = false ]; then
        install_coderabbit || {
            log "ERROR" "Coderabbit installation failed"
            installation_failed=true
        }
    fi

    echo
    log "INFO" "Starting validation workflow..."
    echo

    # Execute validation steps
    local validation_failed=false

    # Step 1: Validate Node
    validate_node || {
        log "ERROR" "Node validation failed"
        validation_failed=true
    }

    # Step 2: Validate Cline
    if [ "$validation_failed" = false ]; then
        validate_cline || {
            log "ERROR" "Cline validation failed"
            validation_failed=true
        }
    fi

    # Step 3: Validate Kestra
    if [ "$validation_failed" = false ]; then
        validate_kestra || {
            log "WARNING" "Kestra validation had issues, but continuing"
            # Don't fail workflow for Kestra
        }
    fi

    # Step 4: Validate Coderabbit
    if [ "$validation_failed" = false ]; then
        validate_coderabbit || {
            log "ERROR" "Coderabbit validation failed"
            validation_failed=true
        }
    fi

    echo

    # Final status
    if [ "$installation_failed" = true ] || [ "$validation_failed" = true ]; then
        log "ERROR" "Setup failed!"
        echo
        log "ERROR" "Check the log file for details: $LOG_FILE"
        exit 1
    else
        echo
        log "SUCCESS" "🎉 Setup completed successfully!"
        echo
        log "INFO" "Installed tools:"
        log "INFO" "  - Node.js 24.x: $(node --version 2>/dev/null || echo 'installed')"
        log "INFO" "  - Cline: $(cline --version 2>/dev/null || echo 'installed')"
        log "INFO" "  - Coderabbit: $(coderabbit --version 2>/dev/null || echo 'installed')"
        if command -v docker &> /dev/null && docker ps -a | grep -q kestra; then
            log "INFO" "  - Kestra: Docker container running"
        else
            log "INFO" "  - Kestra: Not installed (Docker required)"
        fi
        echo
        log "INFO" "Log file: $LOG_FILE"
        log "INFO" "Next steps:"
        log "INFO" "  1. Add NVM to your shell configuration:"
        log "INFO" "     echo 'export NVM_DIR=\"$HOME/.nvm\"' >> ~/.bashrc"
        log "INFO" "     echo '[ -s \"$NVM_DIR/nvm.sh\" ] && \\. \"$NVM_DIR/nvm.sh\"' >> ~/.bashrc"
        log "INFO" "  2. Source your shell configuration:"
        log "INFO" "     source ~/.bashrc"
        log "INFO" "  3. Verify installations:"
        log "INFO" "     node --version"
        log "INFO" "     cline --version"
        log "INFO" "     coderabbit --version"
        exit 0
    fi
}

main "$@"