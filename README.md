# RSAI: Remote Sensing and Automated Image-processing

- Currently Windows Only

## RSAI.exe Manual
### Pre-processing
- reproject
    RSAI.exe reproject "C:\ERSML\Data\input" "C:\ERSML\Data\output"

## CloudRemovalUNet.exe (Test) Manual

- Main commnand
    CloudRemovalUNet.exe --mode [mode]

- Mode types
    preprocess --cloudy [cloudy-image-path] --clear [clear-image-path]
    train --epochs [repeat-count]
    test --model [epoch-location] --test-patch [patch]
    merge --model [epoch-location] 

## How to Build?
### RSAI.exe
- Extract binat-packages.zip as 'bin' folder
- Run build.bat
- Check build/Release/RSAI.exe

### CloudRemovalUNet.exe
- Install Miniconda [https://www.anaconda.com/docs/getting-started/miniconda/main]
- Make virtual environment and install below packages
    python -m pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu118
    conda install rasterio PIL numpy joblib
%pip install rasterio PIL numpy joblib
- In virtual environment, at main.py directory, run this command
    python -m PyInstaller main.py --onefile --console --name CloudRemovalUNet
- Check 'dist' folder