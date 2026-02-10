@echo off
setlocal

echo ========================================
echo  whisper-burn - Build Script
echo ========================================
echo.

if "%1"=="lib" goto lib
if "%1"=="run" goto run
if "%1"=="test" goto test
if "%1"=="clean" goto clean

:app
echo [*] Building whisper-native (release)...
cargo build --release --bin whisper-native
if errorlevel 1 (
    echo [!] Build failed.
    exit /b 1
)
echo [+] Build OK: target\release\whisper-native.exe
if "%1"=="run" goto run_after_build
goto done

:lib
echo [*] Building library only (no GUI)...
cargo build --release --no-default-features --features wgpu
if errorlevel 1 (
    echo [!] Build failed.
    exit /b 1
)
echo [+] Build OK
goto done

:run
echo [*] Building and running whisper-native...
:run_after_build
cargo run --release --bin whisper-native
goto done

:test
echo [*] Running tests...
cargo test
goto done

:clean
echo [*] Cleaning build artifacts...
cargo clean
goto done

:done
echo.
echo Done.
endlocal
