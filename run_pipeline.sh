#!/bin/bash

# YOLO OBB + Label Studio Pipeline Runner
# This script runs the complete pipeline with proper environment setup

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_message() {
    local color=$1
    local message=$2
    echo -e "${color}${message}${NC}"
}

print_header() {
    echo
    echo "==========================================="
    echo "🚀 YOLO OBB + Label Studio Pipeline"
    echo "==========================================="
    echo
}

print_section() {
    echo
    print_message $BLUE "📋 $1"
    echo "-------------------------------------------"
}

# Check if Python is available
check_python() {
    print_section "Checking Python Environment"
    
    if ! command -v python3 &> /dev/null; then
        print_message $RED "❌ Python 3 is required but not installed"
        exit 1
    fi
    
    python_version=$(python3 --version 2>&1 | awk '{print $2}')
    print_message $GREEN "✅ Python $python_version found"
}

# Install dependencies
install_dependencies() {
    print_section "Installing Dependencies"
    
    if [ -f "requirements.txt" ]; then
        print_message $YELLOW "📦 Installing Python packages..."
        pip3 install -r requirements.txt
        print_message $GREEN "✅ Dependencies installed"
    else
        print_message $RED "❌ requirements.txt not found"
        exit 1
    fi
}

# Check configuration
check_config() {
    print_section "Checking Configuration"
    
    if [ ! -f "config/config.yaml" ]; then
        print_message $RED "❌ Configuration file not found: config/config.yaml"
        print_message $YELLOW "Please create the configuration file based on config/config.yaml.template"
        exit 1
    fi
    
    print_message $GREEN "✅ Configuration file found"
    
    # Check for required environment variables for Label Studio local files
    if [ -z "$LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED" ]; then
        print_message $YELLOW "⚠️  Setting LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true"
        export LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true
    fi
    
    if [ -z "$LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT" ]; then
        DOCUMENT_ROOT=$(realpath ./data/raw_images)
        print_message $YELLOW "⚠️  Setting LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=$DOCUMENT_ROOT"
        export LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT="$DOCUMENT_ROOT"
    fi
}

# Create necessary directories
setup_directories() {
    print_section "Setting up Directories"
    
    directories=("data/raw_images" "data/yolo_results" "data/predictions" "logs" "models")
    
    for dir in "${directories[@]}"; do
        if [ ! -d "$dir" ]; then
            mkdir -p "$dir"
            print_message $GREEN "✅ Created directory: $dir"
        fi
    done
}

# Show usage information
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo
    echo "Options:"
    echo "  -h, --help              Show this help message"
    echo "  -s, --stage STAGE       Run only specific stage (download|inference|convert|upload)"
    echo "  -c, --config FILE       Use specific configuration file (default: config/config.yaml)"
    echo "  --install-deps          Install dependencies before running"
    echo "  --dry-run              Show what would be executed without running"
    echo "  --skip-validation      Skip prerequisite validation"
    echo
    echo "Examples:"
    echo "  $0                      # Run complete pipeline"
    echo "  $0 --install-deps       # Install dependencies and run pipeline"
    echo "  $0 -s inference         # Run only YOLO inference stage"
    echo "  $0 --dry-run           # Show pipeline execution plan"
    echo
}

# Parse command line arguments
STAGE=""
CONFIG_FILE="config/config.yaml"
INSTALL_DEPS=false
DRY_RUN=false
SKIP_VALIDATION=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_usage
            exit 0
            ;;
        -s|--stage)
            STAGE="$2"
            shift 2
            ;;
        -c|--config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --install-deps)
            INSTALL_DEPS=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --skip-validation)
            SKIP_VALIDATION=true
            shift
            ;;
        *)
            print_message $RED "❌ Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Main execution
main() {
    print_header
    
    # Check Python
    check_python
    
    # Install dependencies if requested
    if [ "$INSTALL_DEPS" = true ]; then
        install_dependencies
    fi
    
    # Check configuration
    check_config
    
    # Setup directories
    setup_directories
    
    # Build Python command
    python_cmd="python3 scripts/main_pipeline.py"
    
    if [ ! -z "$STAGE" ]; then
        python_cmd="$python_cmd --stage $STAGE"
    fi
    
    if [ ! -z "$CONFIG_FILE" ]; then
        python_cmd="$python_cmd --config $CONFIG_FILE"
    fi
    
    if [ "$DRY_RUN" = true ]; then
        python_cmd="$python_cmd --dry-run"
    fi
    
    if [ "$SKIP_VALIDATION" = true ]; then
        python_cmd="$python_cmd --skip-validation"
    fi
    
    # Run the pipeline
    print_section "Starting Pipeline Execution"
    print_message $BLUE "Command: $python_cmd"
    echo
    
    # Execute the Python pipeline
    $python_cmd
    
    exit_code=$?
    
    echo
    if [ $exit_code -eq 0 ]; then
        print_message $GREEN "🎉 Pipeline completed successfully!"
    elif [ $exit_code -eq 130 ]; then
        print_message $YELLOW "⏹️  Pipeline interrupted by user"
    else
        print_message $RED "❌ Pipeline failed with exit code $exit_code"
    fi
    
    exit $exit_code
}

# Run main function
main "$@" 