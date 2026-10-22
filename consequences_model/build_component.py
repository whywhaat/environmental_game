"""Собирает Langflow-компонент в один самодостаточный файл (ADR-0004).

На genaiflow.ru в JSON flow хранится только код узла, пакетов из репо там нет, поэтому модель
встраивается в файл компонента целиком. Пересобрать: python -m consequences_model.build_component

Сборка отказывается работать, если файл вышел бы молча неправильным: модуль пакета не попал в
порядок сборки, модуль импортирует имя из модуля, который собирается позже, или имена из разных
модулей и общих импортов перекрывают друг друга.
"""

from __future__ import annotations

import ast
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent
COMPONENT_PATH = PACKAGE.parent / "langflow/custom_components/my_components/ConsequencesModel.py"
# Порядок сборки: каждый модуль импортирует только из модулей выше.
MODULES = ("catalog", "protocol", "state", "report", "explain", "model", "langflow_component")
NOT_BUILT = {"__init__", "__main__", "build_component"}
HEADER = '''"""Модель последствий для Langflow. СОБРАНО из пакета consequences_model — руками не править.

Пересобрать: python -m consequences_model.build_component
"""

from __future__ import annotations
'''


def defined_names(node: ast.stmt) -> set[str]:
    """Имена, которые оператор верхнего уровня определяет в модуле."""
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return {node.name}
    if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        return {name.id for target in targets for name in ast.walk(target) if isinstance(name, ast.Name)}
    return set()


def render_component(package: Path = PACKAGE) -> str:
    modules = {path.stem for path in package.glob("*.py")} - NOT_BUILT
    if modules != set(MODULES):
        raise ValueError(f"Модули пакета и порядок сборки MODULES расходятся: {sorted(modules ^ set(MODULES))}")

    definitions: dict[str, str] = {}  # имя → модуль, где оно определено
    shared_imports: dict[str, str] = {}  # имя → откуда импортировано (общие импорты поднимаются наверх файла)
    import_lines: list[str] = []
    sections: list[str] = []
    for module in MODULES:
        source = (package / f"{module}.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        docstring = ast.get_docstring(tree)
        if not docstring:
            raise ValueError(f"У модуля {module} нет докстринга: из него берётся заголовок раздела")
        dropped: set[int] = set()
        for index, node in enumerate(tree.body):
            if index == 0:  # докстринг
                dropped.update(range(node.lineno - 1, node.end_lineno or node.lineno))
            elif isinstance(node, ast.ImportFrom) and node.module == "__future__":
                dropped.update(range(node.lineno - 1, node.end_lineno or node.lineno))
            elif isinstance(node, ast.ImportFrom) and node.level:
                # Внутри пакета: имя должно быть уже определено в модуле выше, иначе в файле будет NameError.
                for alias in node.names:
                    if alias.asname or definitions.get(alias.name) != node.module:
                        raise ValueError(f"{module} импортирует {alias.name} из .{node.module}: имя должно быть "
                                         "определено в этом модуле, без переименования, и модуль — стоять выше "
                                         "в MODULES")
                dropped.update(range(node.lineno - 1, node.end_lineno or node.lineno))
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                origin = node.module if isinstance(node, ast.ImportFrom) else ""
                for alias in node.names:
                    name = alias.asname or alias.name.split(".")[0]
                    source_name = f"{origin}.{alias.name}".lstrip(".")
                    if name in definitions or shared_imports.get(name, source_name) != source_name:
                        raise ValueError(f"Имя {name} из импорта в {module} перекрывается с другим определением")
                    shared_imports[name] = source_name
                import_lines.append(ast.get_source_segment(source, node) or "")
                dropped.update(range(node.lineno - 1, node.end_lineno or node.lineno))
            else:
                if node.lineno - 1 in dropped:
                    raise ValueError(f"В {module} оператор стоит в одной строке с импортом: разнесите их")
                for name in defined_names(node):
                    if name in shared_imports or definitions.get(name, module) != module:
                        raise ValueError(f"Имя {name} в {module} перекрывает {definitions.get(name, 'общий импорт')}")
                    definitions[name] = module
        body = "\n".join(line for number, line in enumerate(source.splitlines()) if number not in dropped).strip()
        sections.append(f"# --- consequences_model/{module}.py: {docstring.splitlines()[0]}\n\n{body}")
    return HEADER + "\n" + "\n".join(dict.fromkeys(import_lines)) + "\n\n\n" + "\n\n\n".join(sections) + "\n"


def main() -> None:
    COMPONENT_PATH.write_text(render_component(), encoding="utf-8")
    print(f"Собрано: {COMPONENT_PATH}")


if __name__ == "__main__":
    main()
