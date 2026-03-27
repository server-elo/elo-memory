"""CLI for neuro-memory."""

import argparse
import sys


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="neuro-memory", description="Elo Memory: Bio-inspired episodic memory for AI"
    )

    parser.add_argument("--version", action="version", version="%(prog)s 0.1.0")

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Server command
    server_parser = subparsers.add_parser("server", help="Start API server")
    server_parser.add_argument(
        "--port", type=int, default=8000, help="Port to run server on (default: 8000)"
    )
    server_parser.add_argument(
        "--host", default="0.0.0.0", help="Host to bind to (default: 0.0.0.0)"
    )

    # Info command
    info_parser = subparsers.add_parser("info", help="Show system info")

    args = parser.parse_args()

    if args.command == "server":
        print(f"Starting Neuro-Memory server on {args.host}:{args.port}")
        print("Note: Install with 'pip install neuro-memory[api]' for full server functionality")
        # TODO: Implement actual server startup

    elif args.command == "info":
        from . import __version__, __license__

        print(f"Elo Memory {__version__}")
        print(f"License: {__license__}")
        print("Bio-inspired episodic memory system implementing EM-LLM (ICLR 2025)")
        print("\nComponents:")
        print("  - Bayesian Surprise Detection")
        print("  - Event Segmentation")
        print("  - Episodic Storage")
        print("  - Two-Stage Retrieval")
        print("  - Memory Consolidation")
        print("  - Forgetting & Decay")
        print("  - Interference Resolution")
        print("  - Online Learning")

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
