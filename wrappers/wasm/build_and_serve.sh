#!/bin/bash
# Build and Serve ZXing WASM Demo Script
# This script compiles zxing-cpp to WebAssembly and starts a local server

set -e

# Configuration
SKIP_BUILD=false
PORT=8080

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-build)
            SKIP_BUILD=true
            shift
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--skip-build] [--port PORT]"
            exit 1
            ;;
    esac
done

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WASM_DIR="$SCRIPT_DIR"
BUILD_DIR="$WASM_DIR/build"

echo "=== ZXing-cpp WASM Build and Serve Script ==="
echo ""
echo "WASM Directory: $WASM_DIR"
echo "Build Directory: $BUILD_DIR"
echo ""

# Check if Emscripten is available
check_emscripten() {
    if command -v emcmake &> /dev/null; then
        echo "[OK] Emscripten found: $(which emcmake)"
        return 0
    else
        echo "[ERROR] Emscripten (emcmake) not found in PATH"
        echo "Please install Emscripten and ensure it's in your PATH."
        echo "See: https://emscripten.org/docs/getting_started/downloads.html"
        return 1
    fi
}

# Build the WASM files
build_wasm() {
    echo ""
    echo "=== Building WASM ==="
    
    # Create build directory
    mkdir -p "$BUILD_DIR"
    
    pushd "$BUILD_DIR" > /dev/null
    
    # Run emcmake cmake
    echo "Running emcmake cmake..."
    emcmake cmake \
        -DCMAKE_BUILD_TYPE=Release \
        -DZXING_READERS=ON \
        -DZXING_WRITERS=ON \
        "$WASM_DIR"
    
    # Build
    echo ""
    echo "Running cmake --build..."
    cmake --build . --config Release
    
    popd > /dev/null
    
    echo ""
    echo "[OK] Build completed successfully!"
}

# Copy WASM files to the wasm directory
copy_wasm_files() {
    echo ""
    echo "=== Copying WASM files ==="
    
    FILES=(
        "zxing.js"
        "zxing.wasm"
        "zxing_reader.js"
        "zxing_reader.wasm"
        "zxing_writer.js"
        "zxing_writer.wasm"
    )
    
    for file in "${FILES[@]}"; do
        src_path="$BUILD_DIR/$file"
        dest_path="$WASM_DIR/$file"
        
        if [ -f "$src_path" ]; then
            cp "$src_path" "$dest_path"
            echo "  Copied: $file"
        else
            echo "  Not found (optional): $file"
        fi
    done
    
    echo ""
    echo "[OK] Files copied to: $WASM_DIR"
}

# Start a local web server
start_web_server() {
    echo ""
    echo "=== Starting Web Server ==="
    echo ""
    
    pushd "$WASM_DIR" > /dev/null
    
    echo "========================================"
    echo "  Demo URLs (HTTPS with COI headers):"
    echo "  Camera Reader: https://localhost:$PORT/demo_cam_reader.html"
    echo "  File Reader:   https://localhost:$PORT/demo_reader.html"
    echo "  Writer:        https://localhost:$PORT/demo_writer.html"
    echo "========================================"
    echo ""
    echo "NOTE: Accept the self-signed certificate warning in your browser."
    echo "      HTTPS with Cross-Origin Isolation is required for multi-threading."
    echo ""
    echo "Press Ctrl+C to stop the server"
    echo ""
    
    # Try to open browser (use https since we need COI headers)
    if command -v xdg-open &> /dev/null; then
        xdg-open "https://localhost:$PORT/demo_cam_reader.html" &
    elif command -v open &> /dev/null; then
        open "https://localhost:$PORT/demo_cam_reader.html" &
    fi
    
    # Start HTTPS server with Cross-Origin Isolation headers (required for multi-threading)
    if [ -f "$WASM_DIR/https_server.py" ]; then
        echo "Starting HTTPS server with Cross-Origin Isolation headers..."
        if command -v python3 &> /dev/null; then
            python3 "$WASM_DIR/https_server.py"
        elif command -v python &> /dev/null; then
            python "$WASM_DIR/https_server.py"
        fi
    elif command -v emrun &> /dev/null; then
        emrun --serve_after_close --port "$PORT" demo_cam_reader.html
    else
        echo "[ERROR] No suitable web server found."
        echo "Please install Python or use emrun from Emscripten."
    fi
    
    popd > /dev/null
}

# Main execution
if [ "$SKIP_BUILD" = false ]; then
    if ! check_emscripten; then
        exit 1
    fi
    
    build_wasm
    copy_wasm_files
else
    echo "Skipping build (using existing WASM files)"
    
    # Verify WASM files exist
    if [ ! -f "$WASM_DIR/zxing_reader.js" ]; then
        echo "[ERROR] WASM files not found. Run without --skip-build first."
        exit 1
    fi
fi

start_web_server
