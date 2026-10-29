# -*- coding: utf-8 -*-
import py_compile

file_path = "langflow/custom_components/my_components/dialog.py"
with open(file_path, "r", encoding="utf-8") as f:
    code = f.read()

# Replace lev["default"] with lev.get("default")
code = code.replace('lev["default"]', 'lev.get("default")')
code = code.replace("lev['default']", 'lev.get("default")')

with open(file_path, "w", encoding="utf-8") as f:
    f.write(code)

py_compile.compile(file_path, doraise=True)
print("Updated lev['default'] to lev.get('default') safely!")
