# -*- coding: utf-8 -*-
import py_compile

file_path = "langflow/custom_components/my_components/dialog.py"
with open(file_path, "r", encoding="utf-8") as f:
    code = f.read()

# Check blocks 1-5 already applied
print("Block 1 applied:", "чанков RAG" in code)
print("Block 2 applied:", "штатн" in code)
print("Block 3 applied:", "max_budget" in code)
print("Block 4 applied:", "_validate_and_sanitize_actions" in code)
print("Block 5 applied:", "sp_max_b" in code)

# 6. Update parser human prompt to explicitly show speaker's budget
old_parser_human = """        human_prompt = (
            f"ТЕКУЩИЙ СПИКЕР: @{speaker}\\n"
            f"Рычаги в ведении спикера: {owned_str}\\n"
            f"Текущий зафиксированный взнос спикера: {cur_contrib_str}\\n\\n" """

new_parser_human = """        human_prompt = (
            f"ТЕКУЩИЙ СПИКЕР: @{speaker}\\n"
            f"Лимит бюджета спикера: {sp_budget_str}\\n"
            f"Рычаги в ведении спикера: {owned_str}\\n"
            f"Текущий зафиксированный взнос спикера: {cur_contrib_str}\\n\\n" """

old_ph = """        human_prompt = (
            f"ТЕКУЩИЙ СПИКЕР: @{speaker}\\n"
            f"Рычаги в ведении спикера: {owned_str}\\n"
            f"Текущий зафиксированный взнос спикера: {cur_contrib_str}\\n\\n"
            f"ВСЕ СТОРОНЫ: {all_players_str}\\n\\n" """

new_ph = """        human_prompt = (
            f"ТЕКУЩИЙ СПИКЕР: @{speaker}\\n"
            f"Лимит бюджета спикера: {sp_budget_str}\\n"
            f"Рычаги в ведении спикера: {owned_str}\\n"
            f"Текущий зафиксированный взнос спикера: {cur_contrib_str}\\n\\n"
            f"ВСЕ СТОРОНЫ: {all_players_str}\\n\\n" """

if old_ph.strip() in code:
    code = code.replace(old_ph.strip(), new_ph.strip(), 1)
    print("OK: Replaced parser human prompt budget")
else:
    print("Trying substring for human prompt...")
    target = f'ТЕКУЩИЙ СПИКЕР: @{{speaker}}\\n"\n            f"Рычаги в ведении спикера:'
    repl = f'ТЕКУЩИЙ СПИКЕР: @{{speaker}}\\n"\n            f"Лимит бюджета спикера: {{sp_budget_str}}\\n"\n            f"Рычаги в ведении спикера:'
    if target in code:
        code = code.replace(target, repl, 1)
        print("OK: Replaced parser human prompt via target")

# 7. Update _build_prompt_phase2 with focus on real table parameters
target_p2 = """            f"ИНСТРУКЦИЯ К ЭТАПУ 2:\\n"
            f"Веди торг, отстаивай интересы своей стороны, предлагай компромиссы или выдвигай требования и ультиматумы.\\n"
            f"Секретарь слушаний зафиксирует все твои слова, взносы и встречные условия в протоколе.\\n" """

repl_p2 = """            f"ИНСТРУКЦИЯ К ЭТАПУ 2:\\n"
            f"Веди торг, отстаивай интересы своей стороны, предлагай компромиссы или выдвигай требования и ультиматумы.\\n"
            f"ФОКУС НА СТОЛЕ ПЕРЕГОВОРОВ: договаривайся СТРОГО вокруг параметров таблицы выше (взносы сторон, тарифы ЖКХ, эко-датчики 24/7, сроки, мощность завода, налоговые льготы, мораторий на протесты, питьевая вода). Не выдумывай несуществующие внешние структуры (советы, фонды, центры), которых нет на столе переговоров.\\n"
            f"Секретарь слушаний зафиксирует все твои слова, взносы и встречные условия в протоколе.\\n" """

if target_p2.strip() in code:
    code = code.replace(target_p2.strip(), repl_p2.strip(), 1)
    print("OK: Replaced _build_prompt_phase2 table focus")
else:
    print("Trying substring for phase 2...")
    t2 = 'f"Секретарь слушаний зафиксирует все твои слова, взносы и встречные условия в протоколе.\\n"'
    r2 = 'f"ФОКУС НА СТОЛЕ ПЕРЕГОВОРОВ: договаривайся СТРОГО вокруг параметров таблицы выше (взносы сторон, тарифы ЖКХ, эко-датчики 24/7, сроки, мощность завода, налоговые льготы, мораторий на протесты, питьевая вода). Не выдумывай несуществующие внешние структуры (советы, фонды, центры), которых нет на столе переговоров.\\n"\n            f"Секретарь слушаний зафиксирует все твои слова, взносы и встречные условия в протоколе.\\n"'
    if t2 in code:
        code = code.replace(t2, r2, 1)
        print("OK: Replaced phase 2 via substring")

with open(file_path, "w", encoding="utf-8") as f:
    f.write(code)

py_compile.compile(file_path, doraise=True)
print("dialog.py compiled successfully with py_compile!")
