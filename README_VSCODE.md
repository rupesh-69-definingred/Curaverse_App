# Curaverse — VS Code Ready

## Project structure
```text
Curaverse_VSCode_Ready/
├── app.py                 # Flask application
├── requirements.txt       # Python dependencies
├── README.md              # Quick start
├── README_VSCODE.md      # VS Code setup notes
├── .gitignore
├── .vscode/
│   ├── launch.json       # F5 / Run & Debug configuration
│   └── settings.json     # VS Code Python settings
├── uploads/
│   └── .keep
└── patients.db           # Created automatically on first run
```

> The current prototype renders its UI directly from `app.py`, so a `templates/` folder is not required.

## Windows + VS Code

1. Install Python 3.10+ and VS Code with the Python extension.
2. Extract this ZIP and open the **Curaverse_VSCode_Ready** folder in VS Code.
3. Open **Terminal → New Terminal**.
4. Create a virtual environment:

```powershell
python -m venv .venv
```

5. Activate it:

```powershell
.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, use Command Prompt and run:

```cmd
.venv\Scripts\activate.bat
```

6. Install dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

7. Run:

```bash
python app.py
```

8. Open:

```text
http://127.0.0.1:5000
```

Demo OTP: `123456`

## VS Code Run button / F5

Press **F5** and select the `Curaverse Flask` configuration if VS Code asks. The included `.vscode/launch.json` runs `app.py` with the selected Python interpreter.

## Image Processing

The prototype supports image upload, grayscale preprocessing and K-Means clustering with K=3. It is a demonstration/visual segmentation workflow and **must not be presented as an automated clinical diagnosis**.

## Stop the server

Press `Ctrl+C` in the terminal.
