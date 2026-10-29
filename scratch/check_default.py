# -*- coding: utf-8 -*-
with open("langflow/custom_components/my_components/dialog.py", "r", encoding="utf-8") as f:
    for idx, line in enumerate(f, 1):
        if '["default"]' in line or "['default']" in line:
            print(f"Line {idx}: {line.strip()}")
