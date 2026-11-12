#!/usr/bin/env bash
# macOS / Linux launcher. On Linux you may first need:  sudo apt install python3-venv python3-tk
cd "$(dirname "$0")"
ENVDIR="$HOME/.venvs/seizure"
if [ ! -x "$ENVDIR/bin/python" ] || ! "$ENVDIR/bin/python" -c "import sklearn, mne, PIL, tkinter; assert sklearn.__version__=='1.6.1'" 2>/dev/null; then
  echo "First-time setup (3-6 minutes, needs internet)..."
  python3 -m venv "$ENVDIR" && "$ENVDIR/bin/pip" install -q --upgrade pip && \
  "$ENVDIR/bin/pip" install -q "scikit-learn==1.6.1" numpy pandas scipy joblib matplotlib pillow mne || { echo "Setup failed"; exit 1; }
fi
exec "$ENVDIR/bin/python" gui.py "$@"
