"""Собирает Langflow-компонент в один самодостаточный файл (ADR-0004).

На genaiflow.ru в JSON flow хранится только код узла, пакетов из репо там нет, поэтому модель
встраивается в файл компонента целиком. Пересобрать: python -m consequences_model.build_component
"""

from __future__ import annotations

import ast
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent
COMPONENT_PATH = PACKAGE.parent / "langflow/custom_components/my_components/ConsequencesModel.py"
# Порядок важен: каждый модуль опирается только на предыдущие.
MODULES = ("catalog", "protocol", "state", "report", "explain", "model", "langflow_component")
HEADER = '''"""Модель последствий для Langflow. СОБРАНО из пакета consequences_model — руками не править.

Пересобрать: python -m consequences_model.build_component
"""

from __future__ import annotations
'''


def top_level_names(tree: ast.Module) -> set[str]:
    names = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            names.update(target.id for target in node.targets if isinstance(target, ast.Name))
    return names


def render_component() -> str:
    imports: list[str] = []
    sections: list[str] = []
    defined: dict[str, str] = {}
    for module in MODULES:
        source = (PACKAGE / f"{module}.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        for name in top_level_names(tree):
            if name in defined:
                raise ValueError(f"Имя {name} определено и в {defined[name]}, и в {module}: сборка их смешает")
            defined[name] = module
        dropped: set[int] = set()
        docstring = ast.get_docstring(tree) or ""
        for node in tree.body:
            is_docstring = isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and node is tree.body[0]
            is_import = isinstance(node, (ast.Import, ast.ImportFrom))
            if not (is_docstring or is_import):
                continue
            assert node.end_lineno is not None
            dropped.update(range(node.lineno - 1, node.end_lineno))
            relative = isinstance(node, ast.ImportFrom) and (node.level or node.module == "__future__")
            if is_import and not relative:
                imports.append(ast.get_source_segment(source, node) or "")
        body = "\n".join(line for number, line in enumerate(source.splitlines()) if number not in dropped).strip()
        sections.append(f"# --- consequences_model/{module}.py: {docstring.splitlines()[0]}\n\n{body}")
    return HEADER + "\n" + "\n".join(dict.fromkeys(imports)) + "\n\n\n" + "\n\n\n".join(sections) + "\n"


def main() -> None:
    COMPONENT_PATH.write_text(render_component(), encoding="utf-8")
    print(f"Собрано: {COMPONENT_PATH}")


if __name__ == "__main__":
    main()
