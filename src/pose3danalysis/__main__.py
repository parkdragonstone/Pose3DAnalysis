# -*- coding: utf-8 -*-
"""Entry point for running pose3danalysis as a module.

Usage:
    python -m pose3danalysis
"""

import os
import sys
from pathlib import Path

# Force a non-interactive backend to avoid Qt/Tk conflicts when running from the Tk GUI
os.environ.setdefault("MPLBACKEND", "Agg")

# Add project root to Python path so Pose2Sim can be imported
# This allows Pose2Sim to be found when installed as a package
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from pose3danalysis.gui.main_app import Pose3DAnalysisApp


def main():
    """Main entry point."""
    app = Pose3DAnalysisApp()
    app.mainloop()


if __name__ == "__main__":
    main()

