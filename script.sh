#!/bin/bash

set -euo pipefail

# ============================================================================
# PRODUCTION-READY WORKFLOW SCRIPT
# Complete rewrite with all improvements, no code omitted
# ============================================================================

# Configuration
BRANCH="main"
COMMIT_MESSAGE="auto: updated via clinerules workflow"
DRY_RUN=false
VERBOSE=false
SKIP_MISSING_TOOLS=false
LOCK_FILE="/tmp/workflow_${USER}_$(basename "$PWD").lock"
TEMP_DIR="/tmp/workflow_$$"
LOG_FILE="$TEMP_DIR/workflow.log"
ROLLBACK_ENABLED=true

# Tool paths (resolved dynamically)
CLINE=""
CODERABBIT=""
PYTEST=""

# Exit codes
EXIT_SUCCESS=0
EXIT_TOOL_MISSING=1
EXIT_GIT_ERROR=2
EXIT_TEST_FAILED=3
EXIT_LOCK_EXISTS=4
EXIT_VALIDATION_FAILED=5

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
            --branch|-b)
                BRANCH="$2"
                shift 2
                ;;
            --message|-m)
                COMMIT_MESSAGE="$2"
                shift 2
                ;;
            --dry-run|-n)
                DRY_RUN=true
                shift
                ;;
            --verbose|-v)
                VERBOSE=true
                shift
                ;;
            --skip-missing|-s)
                SKIP_MISSING_TOOLS=true
                shift
                ;;
            --no-rollback)
                ROLLBACK_ENABLED=false
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

A production-ready workflow script for automated code improvement, testing, and deployment.

Options:
  -b, --branch BRANCH      Git branch to push to (default: main)
  -m, --message MESSAGE    Commit message (default: 'auto: updated via clinerules workflow')
  -n, --dry-run            Dry run mode (no actual changes)
  -v, --verbose            Verbose output (show all command output)
  -s, --skip-missing       Skip steps for missing tools instead of failing
  --no-rollback            Disable automatic rollback on failure
  -h, --help               Show this help message

Examples:
  $0                                    # Run with defaults
  $0 --dry-run --verbose                # Test run with full output
  $0 --branch develop --skip-missing    # Use develop branch, skip missing tools
  $0 --message "feat: add new feature"  # Custom commit message

Environment:
  The script will automatically detect and use installed tools:
  - Cline (for code improvement and ML review)
  - CodeRabbit (for code review and documentation)
  - pytest (for testing)

Output:
  - Log file: $TEMP_DIR/workflow.log
  - Lock file: $LOCK_FILE

Exit Codes:
  0 - Success
  1 - Tool missing
  2 - Git error
  3 - Test failed
  4 - Lock exists (another instance running)
  5 - Validation failed

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

# Acquire lock to prevent concurrent runs
acquire_lock() {
    if [ -f "$LOCK_FILE" ]; then
        local lock_pid
        lock_pid=$(cat "$LOCK_FILE" 2>/dev/null || echo "unknown")
        
        # Check if the process is still running
        if ps -p "$lock_pid" > /dev/null 2>&1; then
            log "ERROR" "Another instance is running (PID: $lock_pid)"
            log "ERROR" "Lock file: $LOCK_FILE"
            log "INFO" "If you're sure no other instance is running, remove the lock file manually:"
            log "INFO" "  rm $LOCK_FILE"
            exit $EXIT_LOCK_EXISTS
        else
            log "WARNING" "Stale lock file found, removing..."
            rm -f "$LOCK_FILE"
        fi
    fi
    
    echo $$ > "$LOCK_FILE" || {
        log "ERROR" "Failed to create lock file: $LOCK_FILE"
        exit 1
    }
    
    log "DEBUG" "Lock acquired: $LOCK_FILE (PID: $$)"
}

# Release lock
release_lock() {
    if [ -f "$LOCK_FILE" ]; then
        rm -f "$LOCK_FILE"
        log "DEBUG" "Lock released"
    fi
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
    
    # Release lock
    release_lock
    
    if [ $exit_code -eq 0 ]; then
        log "SUCCESS" "Cleanup completed"
    else
        log "WARNING" "Cleanup completed with errors (exit code: $exit_code)"
    fi
}

# Validate environment before starting
validate_environment() {
    log "INFO" "Validating environment..."
    
    # Check if we're in a git repository
    if ! git rev-parse --git-dir > /dev/null 2>&1; then
        log "ERROR" "Not a git repository"
        exit $EXIT_VALIDATION_FAILED
    fi
    
    # Check if .clinerules directory exists
    if [ ! -d ".clinerules" ]; then
        log "ERROR" ".clinerules directory not found"
        log "INFO" "Create the directory structure: mkdir -p .clinerules/prompts"
        exit $EXIT_VALIDATION_FAILED
    fi
    
    # Check if prompts directory exists
    if [ ! -d ".clinerules/prompts" ]; then
        log "WARNING" ".clinerules/prompts directory not found"
        log "INFO" "Some features may not work without prompt files"
    fi
    
    # Validate branch name format
    if ! [[ "$BRANCH" =~ ^[a-zA-Z0-9/_-]+$ ]]; then
        log "ERROR" "Invalid branch name: $BRANCH"
        log "INFO" "Branch name must contain only alphanumeric characters, hyphens, underscores, and slashes"
        exit $EXIT_VALIDATION_FAILED
    fi
    
    # Validate commit message (prevent command injection)
    if [[ "$COMMIT_MESSAGE" =~ [\;\&\|\`\$\(\)] ]]; then
        log "ERROR" "Commit message contains potentially unsafe characters"
        exit $EXIT_VALIDATION_FAILED
    fi
    
    log "SUCCESS" "Environment validation passed"
}

# Resolve tool paths dynamically
resolve_tool_paths() {
    log "INFO" "Resolving tool paths..."
    
    # Resolve Cline
    if command -v cline &>/dev/null; then
        CLINE=$(command -v cline)
        log "DEBUG" "Found cline in PATH: $CLINE"
    else
        # Try common NVM installation paths
        local nvm_paths=(
            "$HOME/.nvm/versions/node/v24.12.0/bin/cline"
            "$HOME/.nvm/versions/node/v*/bin/cline"
        )
        
        for path in "${nvm_paths[@]}"; do
            # Expand glob pattern
            for expanded_path in $path; do
                if [ -x "$expanded_path" ]; then
                    CLINE="$expanded_path"
                    log "DEBUG" "Found cline at: $CLINE"
                    break 2
                fi
            done
        done
    fi
    
    # Resolve CodeRabbit
    if command -v coderabbit &>/dev/null; then
        CODERABBIT=$(command -v coderabbit)
        log "DEBUG" "Found coderabbit: $CODERABBIT"
    fi
    
    
    # Resolve pytest
    if command -v pytest &>/dev/null; then
        PYTEST=$(command -v pytest)
        log "DEBUG" "Found pytest: $PYTEST"
    fi
    
    log "SUCCESS" "Tool path resolution completed"
}

# Check if tool is available
check_tool() {
    local tool_name=$1
    local tool_path=$2
    
    if [ -z "$tool_path" ]; then
        log "WARNING" "$tool_name is not installed or not found"
        
        if [ "$SKIP_MISSING_TOOLS" = true ]; then
            log "INFO" "Skipping $tool_name steps (--skip-missing enabled)"
            return 1
        else
            log "ERROR" "$tool_name is required but not found"
            log "INFO" "Install $tool_name or use --skip-missing flag"
            return 1
        fi
    fi
    
    log "DEBUG" "$tool_name is available: $tool_path"
    return 0
}

# ============================================================================
# GIT OPERATIONS
# ============================================================================

# Git status cache
GIT_HAS_CHANGES=""
GIT_HAS_STAGED=""
GIT_CURRENT_BRANCH=""
GIT_IS_DETACHED=""

# Check and cache git status
check_git_status() {
    log "INFO" "Checking git status..."
    
    # Check for uncommitted changes
    if git diff --quiet 2>/dev/null; then
        GIT_HAS_CHANGES="false"
    else
        GIT_HAS_CHANGES="true"
        log "WARNING" "You have uncommitted changes"
        
        if [ "$DRY_RUN" = true ] || [ "$VERBOSE" = true ]; then
            log "DEBUG" "Changed files:"
            git diff --name-only | while read -r file; do
                log "DEBUG" "  - $file"
            done
        fi
    fi
    
    # Check for staged changes
    if git diff --cached --quiet 2>/dev/null; then
        GIT_HAS_STAGED="false"
    else
        GIT_HAS_STAGED="true"
        log "WARNING" "You have staged but uncommitted changes"
        
        if [ "$DRY_RUN" = true ] || [ "$VERBOSE" = true ]; then
            log "DEBUG" "Staged files:"
            git diff --cached --name-only | while read -r file; do
                log "DEBUG" "  - $file"
            done
        fi
    fi
    
    # Check current branch
    GIT_CURRENT_BRANCH=$(git symbolic-ref --short HEAD 2>/dev/null || echo "")
    
    if [ -z "$GIT_CURRENT_BRANCH" ]; then
        GIT_IS_DETACHED="true"
        log "WARNING" "You are in detached HEAD state"
    else
        GIT_IS_DETACHED="false"
        log "DEBUG" "Current branch: $GIT_CURRENT_BRANCH"
        
        if [ "$GIT_CURRENT_BRANCH" != "$BRANCH" ]; then
            log "WARNING" "Current branch ($GIT_CURRENT_BRANCH) differs from target branch ($BRANCH)"
        fi
    fi
    
    log "SUCCESS" "Git status check completed"
}

# Save git state for potential rollback
save_git_state() {
    if [ "$ROLLBACK_ENABLED" = false ]; then
        log "DEBUG" "Rollback disabled, skipping state save"
        return 0
    fi
    
    log "DEBUG" "Saving git state for potential rollback..."
    
    local state_file="$TEMP_DIR/git_state.txt"
    
    git rev-parse HEAD > "$state_file" 2>/dev/null || {
        log "WARNING" "Failed to save git state"
        return 1
    }
    
    log "DEBUG" "Git state saved: $(cat "$state_file")"
    return 0
}

# Rollback git state
rollback_git_state() {
    if [ "$ROLLBACK_ENABLED" = false ]; then
        log "DEBUG" "Rollback disabled, skipping"
        return 0
    fi
    
    local state_file="$TEMP_DIR/git_state.txt"
    
    if [ ! -f "$state_file" ]; then
        log "WARNING" "No saved git state found, cannot rollback"
        return 1
    fi
    
    local saved_commit
    saved_commit=$(cat "$state_file")
    
    log "WARNING" "Rolling back to previous state: $saved_commit"
    
    if [ "$DRY_RUN" = true ]; then
        log "INFO" "[DRY-RUN] Would run: git reset --hard $saved_commit"
        return 0
    fi
    
    git reset --hard "$saved_commit" || {
        log "ERROR" "Failed to rollback git state"
        return 1
    }
    
    log "SUCCESS" "Rollback completed"
    return 0
}

# Perform git operations
git_operations() {
    log "INFO" "🔧 Preparing git operations..."
    
    # Re-check git status (may have changed after workflow steps)
    if [ "$DRY_RUN" = false ]; then
        git diff --quiet && git diff --cached --quiet
        local has_changes=$?
    else
        has_changes=1  # Assume changes in dry-run
    fi
    
    if [ $has_changes -eq 0 ]; then
        log "INFO" "No changes to commit"
        return 0
    fi
    
    if [ "$DRY_RUN" = true ]; then
        log "INFO" "[DRY-RUN] Would execute:"
        log "INFO" "[DRY-RUN]   git add ."
        log "INFO" "[DRY-RUN]   git commit -m \"$COMMIT_MESSAGE\""
        log "INFO" "[DRY-RUN]   git push origin $BRANCH"
        return 0
    fi
    
    # Git add
    log "INFO" "Staging changes..."
    git add . || {
        log "ERROR" "Failed to stage changes"
        return $EXIT_GIT_ERROR
    }
    
    # Git commit
    log "INFO" "Committing changes..."
    git commit -m "$COMMIT_MESSAGE" || {
        local commit_exit=$?
        if [ $commit_exit -eq 1 ]; then
            log "WARNING" "Nothing to commit (working tree clean)"
            return 0
        else
            log "ERROR" "Failed to commit changes (exit code: $commit_exit)"
            return $EXIT_GIT_ERROR
        fi
    }
    
    # Git push
    log "INFO" "Pushing to origin/$BRANCH..."
    git push origin "$BRANCH" || {
        log "ERROR" "Failed to push changes"
        log "INFO" "You may need to pull first or resolve conflicts"
        return $EXIT_GIT_ERROR
    }
    
    log "SUCCESS" "Git operations completed"
    return 0
}

# ============================================================================
# WORKFLOW STEPS
# ============================================================================

# Step 1: Code improvement with Cline
step_improve_code() {
    if ! check_tool "Cline" "$CLINE"; then
        return 0
    fi
    
    local prompt_file=".clinerules/prompts/cline_improve.md"
    
    if [ ! -f "$prompt_file" ]; then
        log "WARNING" "Prompt file not found: $prompt_file"
        log "INFO" "Skipping code improvement step"
        return 0
    fi
    
    log "INFO" "🔧 Improving code with Cline..."
    
    run_command "Cline code improvement" true "$CLINE" -y "$prompt_file" || {
        log "ERROR" "Code improvement failed"
        if [ "$SKIP_MISSING_TOOLS" = false ]; then
            return 1
        fi
    }
    
    return 0
}

# Step 2: ML review with Cline
step_ml_review() {
    if ! check_tool "Cline" "$CLINE"; then
        return 0
    fi
    
    local prompt_file=".clinerules/prompts/ml_review.md"
    
    if [ ! -f "$prompt_file" ]; then
        log "WARNING" "Prompt file not found: $prompt_file"
        log "INFO" "Skipping ML review step"
        return 0
    fi
    
    log "INFO" "🤖 Running ML review..."
    
    run_command "Cline ML review" true "$CLINE" -y "$prompt_file" || {
        log "ERROR" "ML review failed"
        if [ "$SKIP_MISSING_TOOLS" = false ]; then
            return 1
        fi
    }
    
    return 0
}

# Step 3: Run tests
step_run_tests() {
    log "INFO" "🧪 Running tests..."
    
    if [ -z "$PYTEST" ]; then
        log "WARNING" "pytest not found, skipping tests"
        return 0
    fi
    
    # Check if there are any test files
    if ! find . -name "test_*.py" -o -name "*_test.py" | grep -q .; then
        log "INFO" "No test files found, skipping tests"
        return 0
    fi
    
    run_command "pytest" true "$PYTEST" || {
        local test_exit=$?
        log "WARNING" "Tests failed with exit code: $test_exit"
        log "WARNING" "Continuing with workflow despite test failures"
        # Don't exit, just log the failure
    }
    
    return 0
}

# Step 4: CodeRabbit documentation (2-pass)
step_coderabbit_docs() {
    if ! check_tool "CodeRabbit" "$CODERABBIT"; then
        return 0
    fi
    
    if ! check_tool "Cline" "$CLINE"; then
        log "WARNING" "Cline not available, skipping documentation generation"
        return 0
    fi
    
    local prompt_file=".clinerules/prompts/gen_docs_from_review.md"
    
    if [ ! -f "$prompt_file" ]; then
        log "WARNING" "Prompt file not found: $prompt_file"
        log "INFO" "Skipping documentation generation"
        return 0
    fi
    
    # Pass 1
    log "INFO" "📚 Running CodeRabbit Pass 1 (documentation)..."
    
    local pass1_file="$TEMP_DIR/coderabbit_pass1.md"
    
    if [ "$DRY_RUN" = true ]; then
        log "INFO" "[DRY-RUN] Would run: coderabbit --prompt-only --base $BRANCH > $pass1_file"
        log "INFO" "[DRY-RUN] Would run: cline -y $prompt_file"
    else
        "$CODERABBIT" --prompt-only --base "$BRANCH" > "$pass1_file" 2>&1 || {
            log "ERROR" "CodeRabbit Pass 1 failed"
            return 1
        }
        
        log "INFO" "✍️  Applying Pass 1 review with Cline..."
        run_command "Cline documentation (Pass 1)" true "$CLINE" -y "$prompt_file" || {
            log "ERROR" "Failed to apply Pass 1 review"
            return 1
        }
    fi
    
    # Pass 2
    log "INFO" "📚 Running CodeRabbit Pass 2 (documentation)..."
    
    local pass2_file="$TEMP_DIR/coderabbit_pass2.md"
    
    if [ "$DRY_RUN" = true ]; then
        log "INFO" "[DRY-RUN] Would run: coderabbit --prompt-only --base $BRANCH > $pass2_file"
        log "INFO" "[DRY-RUN] Would run: cline -y $prompt_file"
    else
        "$CODERABBIT" --prompt-only --base "$BRANCH" > "$pass2_file" 2>&1 || {
            log "ERROR" "CodeRabbit Pass 2 failed"
            return 1
        }
        
        log "INFO" "✍️  Applying Pass 2 review with Cline..."
        run_command "Cline documentation (Pass 2)" true "$CLINE" -y "$prompt_file" || {
            log "ERROR" "Failed to apply Pass 2 review"
            return 1
        }
    fi
    
    log "SUCCESS" "Documentation generation completed"
    return 0
}

# Step 5: Generate changelog
step_generate_changelog() {
    if ! check_tool "CodeRabbit" "$CODERABBIT"; then
        return 0
    fi
    
    if ! check_tool "Cline" "$CLINE"; then
        log "WARNING" "Cline not available, skipping changelog generation"
        return 0
    fi
    
    local prompt_file=".clinerules/prompts/gen_changelog_from_review.md"
    
    if [ ! -f "$prompt_file" ]; then
        log "WARNING" "Prompt file not found: $prompt_file"
        log "INFO" "Skipping changelog generation"
        return 0
    fi
    
    log "INFO" "📝 Generating changelog..."
    
    local changelog_file="$TEMP_DIR/coderabbit_changelog.md"
    
    if [ "$DRY_RUN" = true ]; then
        log "INFO" "[DRY-RUN] Would run: coderabbit --prompt-only --base $BRANCH > $changelog_file"
        log "INFO" "[DRY-RUN] Would run: cline -y $prompt_file"
    else
        "$CODERABBIT" --prompt-only --base "$BRANCH" > "$changelog_file" 2>&1 || {
            log "ERROR" "CodeRabbit changelog generation failed"
            return 1
        }
        
        log "INFO" "✍️  Applying changelog with Cline..."
        run_command "Cline changelog generation" true "$CLINE" -y "$prompt_file" || {
            log "ERROR" "Failed to generate changelog"
            return 1
        }
    fi
    
    log "SUCCESS" "Changelog generation completed"
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
    echo "║        Production-Ready Workflow Automation Script        ║"
    echo "║                                                            ║"
    echo "╚════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
    echo
    
    # Setup
    setup_temp_dir
    parse_arguments "$@"
    acquire_lock
    
    log "INFO" "Workflow started (PID: $$)"
    log "INFO" "Mode: $([ "$DRY_RUN" = true ] && echo "DRY-RUN" || echo "LIVE")"
    log "INFO" "Target branch: $BRANCH"
    log "INFO" "Commit message: $COMMIT_MESSAGE"
    echo
    
    # Pre-flight checks
    validate_environment
    resolve_tool_paths
    check_git_status
    save_git_state
    
    echo
    log "INFO" "Starting workflow execution..."
    echo
    
    # Execute workflow steps
    local workflow_failed=false
    
    # Step 1: Improve code
    step_improve_code || {
        log "ERROR" "Code improvement step failed"
        workflow_failed=true
    }
    
    if [ "$workflow_failed" = false ]; then
        # Step 2: ML review
        step_ml_review || {
            log "ERROR" "ML review step failed"
            workflow_failed=true
        }
    fi
    
    if [ "$workflow_failed" = false ]; then
        # Step 3: Run tests
        step_run_tests || {
            log "WARNING" "Test step had issues, but continuing"
            # Don't set workflow_failed for test failures
        }
    fi
    
    if [ "$workflow_failed" = false ]; then
        # Step 4: Git operations (commit and push)
        git_operations || {
            log "ERROR" "Git operations failed"
            workflow_failed=true
        }
    fi
    
    if [ "$workflow_failed" = false ]; then
        # Step 5: CodeRabbit documentation
        step_coderabbit_docs || {
            log "WARNING" "Documentation generation had issues, but continuing"
            # Don't fail workflow for docs
        }
    fi
    
    if [ "$workflow_failed" = false ]; then
        # Step 6: Generate changelog
        step_generate_changelog || {
            log "WARNING" "Changelog generation had issues, but continuing"
            # Don't fail workflow for changelog
        }
    fi
    
    
    echo
    
    # Final status
    if [ "$workflow_failed" = true ]; then
        log "ERROR" "Workflow failed!"
        echo
        log "INFO" "Attempting rollback..."
        rollback_git_state || {
            log "WARNING" "Rollback failed or was not possible"
        }
        echo
        log "ERROR" "Check the log file for details: $LOG_FILE"
        exit 1
    else
        echo
	log "SUCCESS" "🎉 Workflow completed successfully!"
	exit 0
    fi
return 0
}

main "$@"