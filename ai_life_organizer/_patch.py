"""Patch app.py to apply the dark-theme upgrade."""
import re

with open("app.py", "r", encoding="utf-8") as f:
    src = f.read()

print("Original length:", len(src))
print("Lines:", src.count("\n"))
