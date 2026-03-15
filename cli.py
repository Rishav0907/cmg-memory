"""
cmg — CLI entry point installed by pip.

After `pip install cmg-memory`, users can run:
    cmg chat --model llama3.2
    cmg chat --store pinecone
    cmg inspect
    cmg stats
    cmg forget "python"
    cmg remember "I always use PyTorch"
    cmg consolidate
    cmg reset
"""

from __future__ import annotations
import sys
import os


def main() -> None:
    """Entry point registered in pyproject.toml [project.scripts]."""
    # Delegate to run.py's main() — reuse everything already built
    try:
        from cmg._run import main as _main
        _main()
    except ImportError:
        # Fallback: run.py lives next to the package during development
        script = os.path.join(os.path.dirname(__file__), "..", "run.py")
        if os.path.exists(script):
            import runpy
            sys.argv[0] = script
            runpy.run_path(script, run_name="__main__")
        else:
            print("cmg: run.py not found. Reinstall with: pip install cmg-memory")
            sys.exit(1)


if __name__ == "__main__":
    main()