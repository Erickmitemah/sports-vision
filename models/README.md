# Models

This repository stores large trained model files in the `models/` folder and tracks them with Git LFS.

Files in this folder:
- `models/action_model.pkl` (~310 MB)
- `models/cluster_model.pkl` (~2 KB)
- `models/xg_model.pkl` (~135 KB)

How to fetch the model files (once per machine):

Windows (PowerShell):
```
git lfs install
git pull
git lfs pull
```

macOS / Linux:
```
git lfs install
git pull
git lfs pull
```

Alternative: regenerate the models locally by running the project setup (this may retrain models and can take time):

Windows:
```
setup.bat
```

macOS / Linux:
```
./setup.sh
```

If you want these model files hosted as release assets or on a cloud storage (to avoid LFS transfers), tell me and I can add a download script and upload instructions.
