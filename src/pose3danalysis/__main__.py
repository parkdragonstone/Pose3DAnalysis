# -*- coding: utf-8 -*-
"""Entry point for running pose3danalysis as a module.

Usage:
    python -m pose3danalysis
"""

import os
# Force a non-interactive backend to avoid Qt/Tk conflicts when running from the Tk GUI
os.environ.setdefault("MPLBACKEND", "Agg")

from pose3danalysis.gui.main_app import Pose3DAnalysisApp


def main():
    """Main entry point."""
    app = Pose3DAnalysisApp()
    app.mainloop()


if __name__ == "__main__":
    main()

