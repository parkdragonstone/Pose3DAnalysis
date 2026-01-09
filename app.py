# app.py
import os
import sys
from pathlib import Path

# Force a non-interactive backend to avoid Qt/Tk conflicts when running from the Tk GUI
os.environ.setdefault("MPLBACKEND", "Agg")

# Add project root to Python path so Pose2Sim can be imported
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from pose3danalysis.gui.main_app import Pose3DAnalysisApp


def main():
    app = Pose3DAnalysisApp()
    app.mainloop()


if __name__ == "__main__":
    main()
