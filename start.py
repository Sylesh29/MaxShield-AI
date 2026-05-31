"""
MaxShield AI — single-command launcher.

Usage:
    python start.py               # loads .env, starts on port 8000
    python start.py --port 9000   # custom port
    python start.py --reload      # hot-reload (dev mode)
"""

from __future__ import annotations

import argparse
import os
import sys

from dotenv import load_dotenv

load_dotenv(override=True)


def _check_env():
    missing = []
    if not os.environ.get("ANTHROPIC_API_KEY", "").strip():
        missing.append("ANTHROPIC_API_KEY  (required - get it at https://console.anthropic.com/)")
    if not os.environ.get("WANDB_API_KEY", "").strip():
        missing.append("WANDB_API_KEY      (optional - enables W&B Weave tracing)")

    if missing:
        print("\n[MaxShield AI] The following env vars are not set:")
        for k in missing:
            print(f"  - {k}")
        print(
            "\n  Copy .env.example to .env and fill in your keys, then restart.\n"
            "  The server will start, but POST /api/v1/scrub-claim will return 503\n"
            "  until ANTHROPIC_API_KEY is set.\n"
        )


def main():
    parser = argparse.ArgumentParser(description="MaxShield AI launcher")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--reload", action="store_true")
    parser.add_argument("--log-level", default="info")
    args = parser.parse_args()

    _check_env()

    import uvicorn

    print(f"\n  MaxShield AI starting on http://{args.host}:{args.port}")
    print(f"  Dashboard   :  http://127.0.0.1:{args.port}/")
    print(f"  Swagger UI  :  http://127.0.0.1:{args.port}/docs")
    print(f"  Mock demo   :  GET  http://127.0.0.1:{args.port}/api/v1/mock-demo")
    print(f"  Scrub claim :  POST http://127.0.0.1:{args.port}/api/v1/scrub-claim\n")

    uvicorn.run(
        "main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=args.log_level,
    )


if __name__ == "__main__":
    main()
