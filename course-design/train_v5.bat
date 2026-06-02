@echo off
chcp 65001 >nul 2>&1
cd /d "%~dp0"
python scripts/train_coco2017.py --dataset datasets/campus_safety_v5/data.yaml --model yolo12s --name campus_safety_v5 --epochs 50 --patience 7 --workers 0 --batch 4 --amp
pause
