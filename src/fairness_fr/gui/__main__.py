"""Enables ``python -m fairness_fr.gui`` as an alternative launch method.

Equivalent to running ``streamlit run src/fairness_fr/gui/app.py``
directly; this just locates the same ``app.py`` and hands it to the
Streamlit CLI so both launch methods behave identically.
"""

from __future__ import annotations

import sys
from pathlib import Path

from streamlit.web import cli as streamlit_cli


def main() -> None:
    """Launch the dashboard via Streamlit's own CLI entry point."""
    app_path = Path(__file__).resolve().parent / "app.py"
    sys.argv = ["streamlit", "run", str(app_path)]
    sys.exit(streamlit_cli.main())


if __name__ == "__main__":
    main()
