import pathlib

content = pathlib.Path('/onedev-workspace/work/competitor-design-research.md').read_text()
print(f"Existing: {len(content)} bytes")
print(f"Ends with: ...{content[-80:]}")
