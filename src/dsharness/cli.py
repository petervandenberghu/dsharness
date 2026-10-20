"""Opdrachtregel: `dsharness serve <map>` en `dsharness rooktest <map>`."""
from __future__ import annotations

import argparse
import json
import sys


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="dsharness")
    sub = p.add_subparsers(dest="opdracht", required=True)
    s = sub.add_parser("serve", help="Start de API rond een resultaatmap")
    s.add_argument("map", nargs="?", default="resultaat")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    r = sub.add_parser("rooktest", help="Deploy-check: API voorspelt hetzelfde als het model")
    r.add_argument("map", nargs="?", default="resultaat")
    a = p.parse_args(argv)
    if a.opdracht == "serve":
        import uvicorn

        from .serve import maak_app
        uvicorn.run(maak_app(a.map), host=a.host, port=a.port)
        return 0
    from .serve import rooktest
    uit = rooktest(a.map)
    print(json.dumps(uit, ensure_ascii=False, indent=1))
    return 0 if uit["api_gelijk_aan_model"] and uit["lege_rij_geweigerd"] else 1


if __name__ == "__main__":
    sys.exit(main())
