@echo off
REM Author: Rui Sun
REM One-click trainer for the analyst persona panel (100 per provider, frozen to agents/).
cd /d "%~dp0"
py train_agents.py --n-personas 100
pause
