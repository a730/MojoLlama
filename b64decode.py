#!/usr/bin/env python3
import pathlib, base64, sys
data = base64.b64decode(sys.argv[1]).decode()
pathlib.Path(sys.argv[2]).write_text(data)
print(f"Written {len(data)} bytes to {sys.argv[2]}")
