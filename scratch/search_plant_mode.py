# -*- coding: utf-8 -*-
with open("langflow/custom_components/my_components/dialog.py", "r", encoding="utf-8") as f:
    code = f.read()

import re
lines = code.splitlines()
for idx, line in enumerate(lines, 1):
    if "plant_mode" in line or "Режим работы" in line or "normalize_lever_value" in line:
        print(f"Line {idx}: {line}")
