# -*- coding: utf-8 -*-
with open("langflow/custom_components/my_components/dialog.py", "r", encoding="utf-8") as f:
    code = f.read()

import re
match = re.search(r'def _get_default_settings\(\) -> dict:(.*?)(?=\ndef\s|\nclass\s)', code, re.S)
if match:
    print(match.group(0))
