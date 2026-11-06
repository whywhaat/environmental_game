"""Сборка переносимого компонента переговоров и обновление кода узла в обоих flow.

Запуск из корня: python -m negotiation_game.build_component
Не требует Langflow, LangChain или обращений к API. Настройки и связи графов сохраняются.
"""

import ast
import copy
import json
from pathlib import Path


PACKAGE = Path(__file__).resolve().parent
ROOT = PACKAGE.parent
COMPONENT_PATH = ROOT / "langflow/custom_components/my_components/dialog.py"
FLOW_PATHS = (
    ROOT / "langflow/Negotiation_MVP_5_players.json",
    ROOT / "langflow/Negotiation_MVP_with_consequences.json",
)
MODULES = ("schemas", "rules", "report", "run_store", "langflow_component")
HEADER = '''"""СОБРАНО из negotiation_game — руками не править.

Пересобрать компонент и код в обоих flow: python -m negotiation_game.build_component
"""
'''


def render_component(package: Path = PACKAGE) -> str:
    """Встраивает модули в порядке зависимостей, убирая только локальные импорты."""
    definitions = {}
    imports = {}
    sections = []
    for module in MODULES:
        source = (package / f"{module}.py").read_text(encoding="utf-8")
        lines = source.splitlines(keepends=True)
        dropped = set()
        for node in ast.parse(source).body:
            if isinstance(node, ast.ImportFrom) and node.level:
                if node.level != 1 or node.module not in MODULES:
                    raise ValueError(f"Неизвестная локальная зависимость в {module}")
                for alias in node.names:
                    if alias.asname or definitions.get(alias.name) != node.module:
                        raise ValueError(f"{module}: {alias.name} должен быть определён в {node.module} раньше")
                dropped.update(range(node.lineno - 1, node.end_lineno))
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                for alias in node.names:
                    name = alias.asname or alias.name.split(".")[0]
                    origin = (node.module if isinstance(node, ast.ImportFrom) else None, alias.name)
                    if name in definitions or (name in imports and imports[name] != origin):
                        raise ValueError(f"Конфликт импорта {name} в {module}")
                    imports[name] = origin
            else:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    names = [node.name]
                elif isinstance(node, ast.Assign):
                    names = [t.id for t in node.targets if isinstance(t, ast.Name)]
                else:
                    names = []
                for name in names:
                    if name in definitions or name in imports:
                        raise ValueError(f"Конфликт определения {name} в {module}")
                    definitions[name] = module
        body = "".join(line for i, line in enumerate(lines) if i not in dropped).strip()
        sections.append(f"# --- negotiation_game/{module}.py ---\n\n{body}\n")
    result = HEADER + "\n\n" + "\n\n".join(sections)
    ast.parse(result)
    return result


def main() -> None:
    code = render_component()
    # Сначала проверяем оба графа и готовим изменения, затем записываем файлы.
    updated_flows = []
    for path in FLOW_PATHS:
        flow = json.loads(path.read_text(encoding="utf-8"))
        nodes = [node for node in flow["data"]["nodes"]
                 if node["data"]["type"] == "NegotiationGame"]
        if len(nodes) != 1:
            raise ValueError(f"{path}: ожидался ровно один узел NegotiationGame")
        nodes[0]["data"]["node"]["template"]["code"]["value"] = code
        template = nodes[0]["data"]["node"]["template"]
        for name, label, default in (
            ("request_timeout_seconds", "Лимит запроса, секунд", 60),
            ("game_timeout_seconds", "Лимит игры, секунд", 900),
            ("request_attempts", "Попыток при сетевой ошибке", 2),
        ):
            if name not in template:
                field = copy.deepcopy(template["turns"])
                field.update(name=name, display_name=label, value=default, advanced=True)
                template[name] = field
            order = nodes[0]["data"]["node"].get("field_order")
            if isinstance(order, list) and name not in order:
                order.append(name)
        updated_flows.append((path, json.dumps(flow, ensure_ascii=False, indent=2) + "\n"))
    COMPONENT_PATH.write_text(code, encoding="utf-8")
    for path, content in updated_flows:
        path.write_text(content, encoding="utf-8")
    print("Собран dialog.py и обновлён код узла в обоих flow.")


if __name__ == "__main__":
    main()
