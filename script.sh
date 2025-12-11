#!/bin/bash

set -e
CLINE="$HOME/.nvm/versions/node/v24.12.0/bin/cline"

# Enhanced workflow script with installation checks, error handling, and edge cases

# Configuration
BRANCH="main"
COMMIT_MESSAGE="auto: updated via clinerules workflow"
DRY_RUN=false
VERBOSE=false
SKIP_MISSING_TOOLS=false

# Parse command line arguments
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
        --help|-h)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  -b, --branch BRANCH      Git branch to push to (default: main)"
            echo "  -m, --message MESSAGE    Commit message (default: 'auto: updated via clinerules workflow')"
            echo "  -n, --dry-run            Dry run mode (no actual changes)"
            echo "  -v, --verbose            Verbose output"
            echo "  -s, --skip-missing       Skip steps for missing tools instead of failing"
            echo "  -h, --help               Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Logging function
log() {
    local level=$1
    local message=$2
    local timestamp=$(date +"%Y-%m-%d %H:%M:%S")
    echo "[$timestamp] [$level] $message"
}

# Error handling function
handle_error() {
    local step=$1
    local exit_code=$2
    log "ERROR" "$step failed with exit code $exit_code"
    if [ "$DRY_RUN" = true ]; then
        log "INFO" "Dry run mode - would have exited here"
        return 0
    fi
    return $exit_code
}

# Check if tool is installed
check_tool() {
    local tool=$1
    if ! command -v "$tool" &> /dev/null; then
        log "WARNING" "$tool is not installed"
        if [ "$SKIP_MISSING_TOOLS" = true ]; then
            log "INFO" "Skipping $tool steps due to --skip-missing flag"
            return 1
        else
            log "ERROR" "$tool is required but not installed. Install it or use --skip-missing flag."
            return 1
        fi
    fi
    return 0
}

# Check git status
check_git_status() {
    if ! git diff --quiet; then
        log "WARNING" "You have uncommitted changes"
        if [ "$DRY_RUN" = true ]; then
            log "INFO" "Dry run mode - would have shown git diff"
            git diff --name-only
        fi
    fi

    if ! git diff --cached --quiet; then
        log "WARNING" "You have staged but uncommitted changes"
        if [ "$DRY_RUN" = true ]; then
            log "INFO" "Dry run mode - would have shown staged changes"
            git diff --cached --name-only
        fi
    fi

    local current_branch=$(git symbolic-ref --short HEAD 2>/dev/null)
    if [ -z "$current_branch" ]; then
        log "WARNING" "You are in detached HEAD state"
    elif [ "$current_branch" != "$BRANCH" ]; then
        log "WARNING" "Current branch is $current_branch, but target branch is $BRANCH"
    fi
}

# Main workflow
main() {
    log "INFO" "Starting workflow"

    # 1. Check tool installations
    log "INFO" "Checking tool installations..."
    local cline_installed=true
    local coderabbit_installed=true
    local oumi_installed=true

    if ! check_tool "cline"; then
        cline_installed=false
    fi

    if ! check_tool "coderabbit"; then
        coderabbit_installed=false
    fi

    if ! check_tool "oumi"; then
        oumi_installed=false
    fi

    # 2. Check git status
    if [ "$DRY_RUN" = false ]; then
        check_git_status
    fi

    # 3. Run Cline improvement (if installed)
    if [ "$cline_installed" = true ]; then
        log "INFO" "🔧 Improving code with Cline..."
        if [ "$DRY_RUN" = true ]; then
            log "INFO" "Dry run mode - would run: cline -y .clinerules/prompts/cline_improve.md"
        else
            if [ "$VERBOSE" = true ]; then
                "$CLINE" -y .clinerules/prompts/cline_improve.md
            else
                "$CLINE" -y .clinerules/prompts/cline_improve.md > /dev/null 2>&1
            fi
            if [ $? -ne 0 ]; then
                handle_error "Cline improvement" $?
                if [ "$SKIP_MISSING_TOOLS" = false ]; then
                    return 1
                fi
            fi
        fi
    else
        log "INFO" "Skipping Cline improvement (tool not installed)"
    fi

    # 4. Run ML review (if installed)
    if [ "$cline_installed" = true ]; then
        log "INFO" "🤖 Running ML Review..."
        if [ "$DRY_RUN" = true ]; then
            log "INFO" "Dry run mode - would run: cline -y .clinerules/prompts/ml_review.md"
        else
            if [ "$VERBOSE" = true ]; then
                "$CLINE" -y .clinerules/prompts/ml_review.md
            else
                "$CLINE" -y .clinerules/prompts/ml_review.md > /dev/null 2>&1
            fi
            if [ $? -ne 0 ]; then
                handle_error "ML Review" $?
                if [ "$SKIP_MISSING_TOOLS" = false ]; then
                    return 1
                fi
            fi
        fi
    else
        log "INFO" "Skipping ML Review (tool not installed)"
    fi

    # 5. Run tests
    log "INFO" "🧪 Running tests..."
    if [ "$DRY_RUN" = true ]; then
        log "INFO" "Dry run mode - would run: pytest"
    else
        if command -v pytest &> /dev/null; then
            if [ "$VERBOSE" = true ]; then
                pytest
            else
                pytest > /dev/null 2>&1
            fi
            local test_exit_code=$?
            if [ $test_exit_code -ne 0 ]; then
                handle_error "Tests" $test_exit_code
                log "WARNING" "Tests failed, but continuing with workflow"
                # Don't exit here, just log the failure
            fi
        else
            log "WARNING" "pytest not found, skipping tests"
        fi
    fi

    # 6. Git operations
    log "INFO" "📤 Preparing Git operations..."
    if [ "$DRY_RUN" = true ]; then
        log "INFO" "Dry run mode - would run:"
        log "INFO" "  git add ."
        log "INFO" "  git commit -m \"$COMMIT_MESSAGE\""
        log "INFO" "  git push origin $BRANCH"
    else
        # Check if there are any changes to commit
        if git diff --quiet && git diff --cached --quiet; then
            log "INFO" "No changes to commit"
        else
            git add . || handle_error "Git add" $?
            git commit -m "$COMMIT_MESSAGE" || handle_error "Git commit" $?
            git push origin "$BRANCH" || handle_error "Git push" $?
        fi
    fi

    # 7. Trigger Oumi deployment (if installed)
    if [ "$oumi_installed" = true ]; then
        log "INFO" "🚀 Deploying with Oumi..."
        if [ "$DRY_RUN" = true ]; then
            log "INFO" "Dry run mode - would run: oumi run .clinerules/prompts/oumi_deploy.md"
        else
            if [ "$VERBOSE" = true ]; then
                oumi run .clinerules/prompts/oumi_deploy.md
            else
                oumi run .clinerules/prompts/oumi_deploy.md > /dev/null 2>&1
            fi
            if [ $? -ne 0 ]; then
                handle_error "Oumi deployment" $?
                log "WARNING" "Oumi deployment failed, but continuing with workflow"
            fi
        fi
    else
        log "INFO" "Skipping Oumi deployment (tool not installed)"
    fi

# 8. Two-pass Coderabbit review + documentation generation
if [ "$coderabbit_installed" = true ]; then
    log "INFO" "📚 Running Coderabbit Pass 1 (prompt-only)..."
    
    coderabbit --prompt-only --base "$BRANCH" > coderabbit_pass1.md
    if [ $? -ne 0 ]; then
        handle_error "Coderabbit Pass 1" $?
    else
        log "INFO" "✍️ Cline applying Pass 1 review..."
        $CLINE -y .clinerules/prompts/gen_docs_from_review.md
    fi

    log "INFO" "📚 Running Coderabbit Pass 2 (prompt-only)..."

    coderabbit --prompt-only --base "$BRANCH" > coderabbit_pass2.md
    if [ $? -ne 0 ]; then
        handle_error "Coderabbit Pass 2" $?
    else
        log "INFO" "✍️ Cline applying Pass 2 review..."
        $CLINE -y .clinerules/prompts/gen_docs_from_review.md
    fi
else
    log "INFO" "Skipping documentation generation (Coderabbit not installed)"
fi


# 9. Generate Changelog (1-pass only)
if [ "$coderabbit_installed" = true ]; then
    log "INFO" "📝 Running Coderabbit for Changelog..."
    
    coderabbit --prompt-only --base "$BRANCH" > coderabbit_changelog.md
    $CLINE -y .clinerules/prompts/gen_changelog_from_review.md
else
    log "INFO" "Skipping Changelog update (Coderabbit not installed)"
fi

    # 10. Final message
    log "INFO" "🎉 Workflow Complete!"
    return 0
}

# Run main function
main
exit $?
