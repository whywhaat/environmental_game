# -*- coding: utf-8 -*-
import re

def clean_speech_message(msg: str) -> str:
    if not msg:
        return ""
    res = re.sub(r"```[a-zA-Z]*\n?.*?\n?```", "", msg, flags=re.S)
    res = re.sub(r"===\s*.+?\s*===", "", res, flags=re.S)

    clean_lines = []
    for line in res.splitlines():
        line_s = line.strip()
        if not line_s:
            continue
        # Пропускаем служебные строки команд
        if re.match(r"^[\s*+\-]*[\[\(]?\b(?:SET|DEMAND|ТРЕБУЮ|TO|ПЕРЕДАТЬ|РЕШЕНИЕ|VOTE|ГОЛОСУЮ)\b", line_s, flags=re.I):
            continue
        # Префикс SPEAK
        line_s = re.sub(r"^[\s\"\'«»*+\-]*\b(?:SPEAK|СКАЗАТЬ|РЕЧЬ)\b\s*[:\-\—]?\s*", "", line_s, flags=re.I).strip()
        # TO @...
        line_s = re.sub(r"\bTO\s*@[A-Za-zА-Яа-я0-9_\(\)\s]{2,40}(?:[:,\s\.\!\?]|$)", "", line_s, flags=re.I).strip()
        # Технические пометки вида (key = value)
        line_s = re.sub(r"\([a-zA-Z0-9_]+\s*=\s*[a-zA-Z0-9_]+\s*(?:,\s*[a-zA-Z0-9_]+\s*=\s*[a-zA-Z0-9_]+\s*)*\)", "", line_s)
        # Артефакты чанков RAG [chunk...] и [chunk_id...]
        line_s = re.sub(r"\[chunk[^\]]*\]", "", line_s, flags=re.I)
        # Замкнутые команды в скобках [SET ...] или (SET ...)
        line_s = re.sub(r"[\[\(](?:SET|DEMAND|ТРЕБУЮ|VOTE|ГОЛОСУЮ)[^\]\)]*[\]\)]", "", line_s, flags=re.I).strip()
        # НЕЗАМКНУТЫЕ команды в скобках в конце строки, например: [SET me Взнос 160 млн;
        line_s = re.sub(r"[\[\(](?:SET|DEMAND|ТРЕБУЮ|VOTE|ГОЛОСУЮ)[^\]\n]*$", "", line_s, flags=re.I).strip()
        # Одиночные остаточные команды SET/DEMAND в конце строки
        line_s = re.sub(r"\b(?:SET|DEMAND|ТРЕБУЮ)\s+(?:me|@[A-Za-zА-Яа-я0-9_]+)[^;\n]*;?$", "", line_s, flags=re.I).strip()

        line_s = re.sub(r"[ \t]{2,}", " ", line_s).strip("«»\"' ")
        if line_s and line_s not in {".", ":", "-", "—"}:
            clean_lines.append(line_s)

    res = "\n".join(clean_lines).strip()
    if not res:
        res = "Подтверждаю свою позицию и ранее заявленные условия участия."
    return res

msg17 = "Давайте определимся с мониторингом или льготами? [SET me Взнос 160 млн;"
c17 = clean_speech_message(msg17)
print("Turn 17 cleaned:", repr(c17))
assert "[SET" not in c17
assert "Давайте определимся с мониторингом или льготами?" in c17

msg23 = "Вклад города минимален и обоснован стратегией. [chunk_2]"
c23 = clean_speech_message(msg23)
print("Turn 23 cleaned:", repr(c23))
assert "[chunk" not in c23
assert "Вклад города минимален и обоснован стратегией." in c23

print("Clean speech tests passed!")
