# -*- coding: utf-8 -*-
import py_compile

file_path = "langflow/custom_components/my_components/dialog.py"
with open(file_path, "r", encoding="utf-8") as f:
    code = f.read()

old_block = """            # 3. Определение следующего оратора
            if starved_player is not None and (pending_speaker_idx is None or turns_silent.get(players[pending_speaker_idx]["name"], 0) < 3):
                current_idx = name_to_idx[starved_player]
                pending_speaker_idx = None"""

new_block = """            # 3. Определение следующего оратора
            if starved_player is not None:
                pending_name = players[pending_speaker_idx]["name"] if pending_speaker_idx is not None else None
                if pending_name is None or turns_silent[starved_player] > turns_silent.get(pending_name, 0):
                    current_idx = name_to_idx[starved_player]
                    pending_speaker_idx = None
                else:
                    current_idx = pending_speaker_idx
                    pending_speaker_idx = None"""

# normalize CRLF if needed
if old_block not in code:
    old_block_crlf = old_block.replace("\n", "\r\n")
    new_block_crlf = new_block.replace("\n", "\r\n")
    if old_block_crlf in code:
        code = code.replace(old_block_crlf, new_block_crlf)
        print("Replaced with CRLF!")
    else:
        print("Could not find old_block even with CRLF")
else:
    code = code.replace(old_block, new_block)
    print("Replaced with LF!")

with open(file_path, "w", encoding="utf-8") as f:
    f.write(code)

py_compile.compile(file_path, doraise=True)
print("Updated dialog.py rotation priority successfully!")
