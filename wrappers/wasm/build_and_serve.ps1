# Build and Serve ZXing WASM Demo Script
# This script compiles zxing-cpp to WebAssembly and starts a local server

param(
    [switch]$SkipBuild,
    [int]$Port = 8080,
    [string]$EmsdkPath = ""
)

$ErrorActionPreference = "Stop"

# Get script directory and set paths
$ScriptDir = $PSScriptRoot
$WasmDir = $ScriptDir
$RootDir = Resolve-Path (Join-Path $ScriptDir "..\..") 
$BuildDir = Join-Path $WasmDir "build"

Write-Host "=== ZXing-cpp WASM Build and Serve Script ===" -ForegroundColor Cyan
Write-Host ""

# Common Emscripten installation locations
$EmsdkSearchPaths = @(
    $EmsdkPath,
    "$env:USERPROFILE\emsdk",
    "C:\Users\Rex\emsdk",
    "$env:LOCALAPPDATA\emsdk",
    "C:\emsdk",
    "D:\emsdk",
    "$env:USERPROFILE\Documents\emsdk",
    "$env:USERPROFILE\Desktop\emsdk",
    "$env:ProgramFiles\emsdk",
    "${env:ProgramFiles(x86)}\emsdk"
)

# Find and activate Emscripten SDK
function Initialize-Emscripten {
    # First check if already in PATH
    $emcc = Get-Command emcmake -ErrorAction SilentlyContinue
    if ($emcc) {
        Write-Host "[OK] Emscripten already in PATH: $($emcc.Source)" -ForegroundColor Green
        return $true
    }
    
    Write-Host "Searching for Emscripten SDK..." -ForegroundColor Yellow
    
    # Search common locations
    foreach ($searchPath in $EmsdkSearchPaths) {
        if ([string]::IsNullOrEmpty($searchPath)) { continue }
        
        $envScript = Join-Path $searchPath "emsdk_env.ps1"
        $envBat = Join-Path $searchPath "emsdk_env.bat"
        
        if (Test-Path $envScript) {
            Write-Host "Found emsdk at: $searchPath" -ForegroundColor Green
            Write-Host "Activating Emscripten environment..." -ForegroundColor Yellow
            
            # Source the environment script
            Push-Location $searchPath
            try {
                # Run emsdk_env.ps1 and capture environment changes
                & $envScript
                Pop-Location
                
                # Verify it worked
                $emcc = Get-Command emcmake -ErrorAction SilentlyContinue
                if ($emcc) {
                    Write-Host "[OK] Emscripten activated: $($emcc.Source)" -ForegroundColor Green
                    return $true
                }
            } catch {
                Pop-Location
                Write-Host "  Failed to activate via ps1: $_" -ForegroundColor Yellow
            }
        }
        
        if (Test-Path $envBat) {
            Write-Host "Found emsdk at: $searchPath" -ForegroundColor Green
            Write-Host "Activating Emscripten environment via batch file..." -ForegroundColor Yellow
            
            # Run batch file and import environment variables
            try {
                $envVars = cmd /c "`"$envBat`" >nul 2>&1 && set"
                foreach ($line in $envVars) {
                    if ($line -match "^([^=]+)=(.*)$") {
                        $varName = $matches[1]
                        $varValue = $matches[2]
                        [Environment]::SetEnvironmentVariable($varName, $varValue, "Process")
                    }
                }
                
                # Verify it worked
                $emcc = Get-Command emcmake -ErrorAction SilentlyContinue
                if ($emcc) {
                    Write-Host "[OK] Emscripten activated: $($emcc.Source)" -ForegroundColor Green
                    return $true
                }
            } catch {
                Write-Host "  Failed to activate via bat: $_" -ForegroundColor Yellow
            }
        }
    }
    
    return $false
}

# Check if Emscripten is available
function Test-Emscripten {
    # Try to initialize/find Emscripten first
    if (Initialize-Emscripten) {
        return $true
    }
    
    Write-Host ""
    Write-Host "[ERROR] Emscripten (emcmake) not found" -ForegroundColor Red
    Write-Host ""
    Write-Host "To install Emscripten:" -ForegroundColor Yellow
    Write-Host "  1. git clone https://github.com/emscripten-core/emsdk.git C:\emsdk" -ForegroundColor White
    Write-Host "  2. cd C:\emsdk" -ForegroundColor White
    Write-Host "  3. .\emsdk.bat install latest" -ForegroundColor White
    Write-Host "  4. .\emsdk.bat activate latest" -ForegroundColor White
    Write-Host ""
    Write-Host "Or specify the path:" -ForegroundColor Yellow
    Write-Host "  .\build_and_serve.ps1 -EmsdkPath 'C:\path\to\emsdk'" -ForegroundColor White
    Write-Host ""
    Write-Host "See: https://emscripten.org/docs/getting_started/downloads.html" -ForegroundColor Cyan
    return $false
}

# Build the WASM files
function Build-Wasm {
    Write-Host ""
    Write-Host "=== Building WASM ===" -ForegroundColor Cyan
    
    # Create build directory
    if (-not (Test-Path $BuildDir)) {
        Write-Host "Creating build directory: $BuildDir"
        New-Item -ItemType Directory -Path $BuildDir -Force | Out-Null
    }
    
    Push-Location $BuildDir
    try {
        # Run emcmake cmake
        Write-Host "Running emcmake cmake..." -ForegroundColor Yellow
        $cmakeArgs = @(
            "cmake",
            "-DCMAKE_BUILD_TYPE=Release",
            "-DZXING_READERS=ON",
            "-DZXING_WRITERS=ON",
            $WasmDir
        )
        & emcmake @cmakeArgs
        if ($LASTEXITCODE -ne 0) {
            throw "emcmake cmake failed with exit code $LASTEXITCODE"
        }
        
        # Build
        Write-Host ""
        Write-Host "Running cmake --build..." -ForegroundColor Yellow
        & cmake --build . --config Release
        if ($LASTEXITCODE -ne 0) {
            throw "cmake build failed with exit code $LASTEXITCODE"
        }
        
        Write-Host ""
        Write-Host "[OK] Build completed successfully!" -ForegroundColor Green
    }
    finally {
        Pop-Location
    }
}

# Copy WASM files to the wasm directory
function Copy-WasmFiles {
    Write-Host ""
    Write-Host "=== Copying WASM files ===" -ForegroundColor Cyan
    
    $filesToCopy = @(
        "zxing.js",
        "zxing.wasm",
        "zxing_reader.js",
        "zxing_reader.wasm",
        "zxing_writer.js",
        "zxing_writer.wasm"
    )
    
    foreach ($file in $filesToCopy) {
        $srcPath = Join-Path $BuildDir $file
        $destPath = Join-Path $WasmDir $file
        
        if (Test-Path $srcPath) {
            Copy-Item -Path $srcPath -Destination $destPath -Force
            Write-Host "  Copied: $file" -ForegroundColor Green
        } else {
            Write-Host "  Not found (optional): $file" -ForegroundColor Yellow
        }
    }
    
    Write-Host ""
    Write-Host "[OK] Files copied to: $WasmDir" -ForegroundColor Green
}

# Start a local web server
function Start-WebServer {
    Write-Host ""
    Write-Host "=== Starting Web Server ===" -ForegroundColor Cyan
    Write-Host ""
    
    Push-Location $WasmDir
    try {
        # Check for Python
        $python = Get-Command python -ErrorAction SilentlyContinue
        if (-not $python) {
            $python = Get-Command python3 -ErrorAction SilentlyContinue
        }
        
        if ($python) {
            # Get local IP address for phone access
            $localIP = (Get-NetIPAddress -AddressFamily IPv4 | Where-Object { $_.InterfaceAlias -notlike "*Loopback*" -and $_.IPAddress -notlike "169.*" -and $_.PrefixOrigin -ne "WellKnown" } | Select-Object -First 1).IPAddress
            if (-not $localIP) { $localIP = "YOUR_LOCAL_IP" }
            
            Write-Host "Starting Python HTTPS server on port $Port..." -ForegroundColor Yellow
            Write-Host ""
            Write-Host "========================================" -ForegroundColor Magenta
            Write-Host "  HTTPS Demo URLs:" -ForegroundColor Magenta
            Write-Host "  Local:  https://localhost:$Port/demo_cam_reader.html" -ForegroundColor White
            Write-Host "  Phone:  https://${localIP}:$Port/demo_cam_reader.html" -ForegroundColor Green
            Write-Host "========================================" -ForegroundColor Magenta
            Write-Host ""
            Write-Host "NOTE: Accept the certificate warning in your browser!" -ForegroundColor Yellow
            Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Yellow
            Write-Host ""
            
            # Create HTTPS server script using Python's cryptography
            $httpsScript = @'
import http.server
import ssl
import os
import sys

# Certificate and key paths
cert_dir = os.path.dirname(os.path.abspath(__file__))
certfile = os.path.join(cert_dir, 'server.pem')

# Generate self-signed certificate using Python
def generate_self_signed_cert():
    try:
        from cryptography import x509
        from cryptography.x509.oid import NameOID
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.backends import default_backend
        from cryptography.hazmat.primitives.asymmetric import rsa
        from cryptography.hazmat.primitives import serialization
        import datetime
        
        # Generate key
        key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
            backend=default_backend()
        )
        
        # Generate certificate
        subject = issuer = x509.Name([
            x509.NameAttribute(NameOID.COMMON_NAME, u"localhost"),
        ])
        
        cert = x509.CertificateBuilder().subject_name(
            subject
        ).issuer_name(
            issuer
        ).public_key(
            key.public_key()
        ).serial_number(
            x509.random_serial_number()
        ).not_valid_before(
            datetime.datetime.utcnow()
        ).not_valid_after(
            datetime.datetime.utcnow() + datetime.timedelta(days=365)
        ).add_extension(
            x509.SubjectAlternativeName([
                x509.DNSName(u"localhost"),
                x509.DNSName(u"*"),
                x509.IPAddress(ipaddress.IPv4Address('127.0.0.1')),
            ]),
            critical=False,
        ).sign(key, hashes.SHA256(), default_backend())
        
        # Write to PEM file (combined cert + key)
        with open(certfile, "wb") as f:
            f.write(cert.public_bytes(serialization.Encoding.PEM))
            f.write(key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.TraditionalOpenSSL,
                encryption_algorithm=serialization.NoEncryption()
            ))
        print(f"Generated certificate: {certfile}")
        return True
    except ImportError:
        return False

# Try to generate cert with cryptography library
import ipaddress
if not os.path.exists(certfile):
    if not generate_self_signed_cert():
        print("Installing cryptography package...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "cryptography", "-q"])
        generate_self_signed_cert()

# Change to script directory to serve files
os.chdir(cert_dir)

# Start HTTPS server
server_address = ('0.0.0.0', PORT_PLACEHOLDER)
httpd = http.server.HTTPServer(server_address, http.server.SimpleHTTPRequestHandler)

context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
context.load_cert_chain(certfile)
httpd.socket = context.wrap_socket(httpd.socket, server_side=True)

print(f"Serving HTTPS on https://0.0.0.0:{PORT_PLACEHOLDER}/")
print("Accept the certificate warning in your browser")
httpd.serve_forever()
'@
            
            # Replace port placeholder
            $httpsScript = $httpsScript -replace 'PORT_PLACEHOLDER', $Port
            
            # Save script to wasm directory
            $scriptPath = Join-Path $WasmDir "https_server.py"
            $httpsScript | Out-File -FilePath $scriptPath -Encoding UTF8 -NoNewline
            
            # Open the browser
            Start-Process "https://localhost:$Port/demo_cam_reader.html"
            
            # Start the HTTPS server
            & $python.Source $scriptPath
        } else {
            Write-Host "[ERROR] Python not found." -ForegroundColor Red
        }
    }
    finally {
        Pop-Location
    }
}

# Main execution
Write-Host "WASM Directory: $WasmDir"
Write-Host "Build Directory: $BuildDir"
Write-Host ""

if (-not $SkipBuild) {
    if (-not (Test-Emscripten)) {
        exit 1
    }
    
    Build-Wasm
    Copy-WasmFiles
} else {
    Write-Host "Skipping build (using existing WASM files)" -ForegroundColor Yellow
    
    # Verify WASM files exist
    $readerJs = Join-Path $WasmDir "zxing_reader.js"
    if (-not (Test-Path $readerJs)) {
        Write-Host "[ERROR] WASM files not found. Run without -SkipBuild first." -ForegroundColor Red
        exit 1
    }
}

Start-WebServer
