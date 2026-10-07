@echo off
rem Project Maya on Windows: not supported yet - this explains why and what to use instead. Nothing is installed.
rem (Strata's own START-HERE.bat installs Strata's Qwen model, not Maya.)
setlocal
title Project Maya
echo.
echo  Project Maya runs on Linux only for now.
echo.
echo  Why: the engine's GLM-5.3-Flash loader maps the model files with Linux calls (mmap, O_DIRECT) and has no
echo  Windows version yet (src\core\glm_model.cu, pack_shard_mmap). WSL2 is not supported either: Strata measured
echo  that WSL2's driver pins only about 1 GB of RAM for the GPU, and Maya keeps tens of GB pinned for its experts.
echo.
echo  What to do: install Linux (Ubuntu 24.04 is a good choice) on this PC (dual boot) or on another machine with
echo  an NVIDIA GPU, copy this folder there and run:
echo.
echo      ./maya.sh
echo.
echo  It checks the PC, compiles the engine, asks before it downloads the model, and starts the dashboard.
echo  Details: README-MAYA.md
echo.
pause
