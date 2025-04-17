#!/usr/bin/env bash

# Load necessary modules
module purge
module load rocm/6.4

# Default behavior: perform configuration and use all available cores for parallelism
DO_CONFIGURE=true
PARALLEL_CORES=$(nproc) # Default to the number of available cores

# Parse optional arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --no-configure)
            DO_CONFIGURE=false
            shift
            ;;
        --parallel)
            if [[ -n $2 && $2 =~ ^[0-9]+$ ]]; then
                PARALLEL_CORES=$2
                shift 2
            else
                echo "Error: --parallel requires a numeric argument."
                exit 1
            fi
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Perform configuration if enabled
if $DO_CONFIGURE; then
    echo "Configuring the project with CMake using $PARALLEL_CORES cores..."
    mkdir build
    cmake -B build -S . -DCMAKE_BUILD_TYPE=Release
fi

# Always build the project
echo "Building the project using $PARALLEL_CORES cores..."
cmake --build build --parallel $PARALLEL_CORES
