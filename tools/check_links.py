#!/usr/bin/env python3
"""Check that every relative link and image in the markdown files points to a file of this repository.
    python3 tools/check_links.py
"""
import re, os, sys, glob, urllib.parse
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
bad = n = 0
for md in glob.glob(os.path.join(ROOT, "**", "*.md"), recursive=True):
    rel_md = os.path.relpath(md, ROOT)
    if rel_md.startswith("reference_results/"): continue
    text = re.sub(r"```.*?```", "", open(md, encoding="utf-8").read(), flags=re.S)
    for m in re.finditer(r"\]\(([^)\s]+)\)|<img[^>]+src=\"([^\"]+)\"", text):
        t = m.group(1) or m.group(2)
        if t.startswith(("http://", "https://", "#", "mailto:")): continue
        t = urllib.parse.unquote(t.split("#")[0]); n += 1
        if not os.path.exists(os.path.normpath(os.path.join(os.path.dirname(md), t))):
            bad += 1; print(f"MISSING {rel_md}: {t}")
print(f"{n} links, {bad} missing"); sys.exit(1 if bad else 0)
