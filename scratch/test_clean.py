import re

def _clean_speech_message(msg: str) -> str:
    if not msg:
        return ""
    # Удаление markdown-блоков кода
    res = re.sub(r"```[a-zA-Z]*\n?.*?\n?```", "", msg, flags=re.S)
    # Удаление системных баннеров === ... ===
    res = re.sub(r"===\s*.+?\s*===", "", res, flags=re.S)
    # Удаление RAG чанков [chunk...] и [chunk_id...] во всем тексте
    res = re.sub(r"\[\s*chunk[^\]]*\]", "", res, flags=re.I)
    # Удаление тегов <speech> и </speech>
    res = re.sub(r"</?speech>", "", res, flags=re.I)

    # Построчная очистка
    clean_lines = []
    for line in res.splitlines():
        line_s = line.strip()
        if not line_s:
            continue
        # Пропускаем служебные строки команд
        if re.match(r"^[\s*+\-]*[\[\(]?\b(?:SET|DEMAND|ТРЕБУЮ|TO|ПЕРЕДАТЬ|РЕШЕНИЕ|VOTE|ГОЛОСУЮ)\b", line_s, flags=re.I):
            continue
        # Фильтруем утечки инструкций парсера и промпта
        line_s = re.sub(r"(?i)Верни\s+СТРОГО\s+валидный\s+JSON.*", "", line_s).strip()
        line_s = re.sub(r"(?i)структуры\s+без\s+markdown-оберток.*", "", line_s).strip()
        line_s = re.sub(r"(?i)Реплика\s+игрока\s*:?", "", line_s).strip()
        line_s = re.sub(r"(?i)ВЫСТУПЛЕНИЕ\s+УЧАСТНИКА\s*:?", "", line_s).strip()
        # Если строка начинается со SPEAK / СКАЗАТЬ / РЕЧЬ, удаляем этот префикс
        line_s = re.sub(r"^[\s\"\'«»*+\-]*\b(?:SPEAK|СКАЗАТЬ|РЕЧЬ)\b\s*[:\-\—]?\s*", "", line_s, flags=re.I).strip()
        # Удаляем директивы адресации TO @... внутри строки
        line_s = re.sub(r"\bTO\s*@[A-Za-zА-Яа-я0-9_\(\)\s]{2,40}(?:[:,\s\.\!\?]|$)", "", line_s, flags=re.I).strip()
        # Удаляем технические пометки вида (key = value)
        line_s = re.sub(r"\([a-zA-Z0-9_]+\s*=\s*[a-zA-Z0-9_]+\s*(?:,\s*[a-zA-Z0-9_]+\s*=\s*[a-zA-Z0-9_]+\s*)*\)", "", line_s)
        # Удаляем любые артефакты чанков RAG [chunk...] и [chunk_id...]
        line_s = re.sub(r"\[\s*chunk[^\]]*\]", "", line_s, flags=re.I)
        # Удаляем остатки команд в круглых и квадратных скобках, например [SET ...] или (SET ...)
        line_s = re.sub(r"[\[\(](?:SET|DEMAND|ТРЕБУЮ|VOTE|ГОЛОСУЮ)[^\]\)]*[\]\)]", "", line_s, flags=re.I).strip()
        # Удаляем незамкнутые скобки команд в конце строк
        line_s = re.sub(r"[\[\(](?:SET|DEMAND|ТРЕБУЮ|VOTE|ГОЛОСУЮ)[^\]\n]*$", "", line_s, flags=re.I).strip()
        # Удаляем одиночные остаточные команды SET/DEMAND в конце строк
        line_s = re.sub(r"\b(?:SET|DEMAND|ТРЕБУЮ)\s+(?:me|@[A-Za-zА-Яа-я0-9_]+)[^;\n]*;?$", "", line_s, flags=re.I).strip()

        line_s = re.sub(r"[ \t]{2,}", " ", line_s).strip("«»\"' ")
        if line_s and line_s not in {".", ":", "-", "—"}:
            clean_lines.append(line_s)

    res = "\n".join(clean_lines).strip()
    if not res:
        res = "Подтверждаю свою позицию и ранее заявленные условия участия."
    return res

test_leak = "Но без открытых датчиков 24/7 и гарантии питьевой воды до конца работ проект для нас не работает. Верни СТРОГО валидный JSON следующей структуры без markdown-оберток."
print("Test prompt leak:")
print("Result:", repr(_clean_speech_message(test_leak)))

test_chunk = "компромиссный статус вины shared устраивает обе стороны. [chunk_1]"
print("Test chunk leak:")
print("Result:", repr(_clean_speech_message(test_chunk)))
