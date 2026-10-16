import copy
import json
import logging
import re

from lfx.custom import Component
from lfx.io import BoolInput, DataInput, MultilineInput, Output
from lfx.schema.data import Data
from langchain_core.messages import HumanMessage, SystemMessage
from config.prompts import judge_event, read_prompt, render_prompt

TOTAL_PROJECT_COST = 1_000_000_000.0  # 1 млрд руб.
PROMPT_LOGGER = logging.getLogger("environmental_game.meeting.prompts")


def _extract_text(content) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                parts.append(str(item.get("text", "")))
            elif hasattr(item, "text"):
                parts.append(str(item.text))
            else:
                parts.append(str(item))
        return "".join(parts).strip()
    return str(content).strip()


def _clean_command(raw_text: str) -> str:
    text = raw_text.strip()
    text = re.sub(r"^```[a-zA-Z]*\n?|\n?```$", "", text).strip()
    text = re.sub(r"^`|`$", "", text).strip()
    return text


def _clean_speech_message(msg: str) -> str:
    if not msg:
        return ""
    res = re.sub(r"^[\s\"\'«»]*[\+\*]*\s*(?:SPEAK|СКАЗАТЬ)?\s*[:\-\—]?\s*", "", msg, flags=re.I).strip()
    res = re.sub(r"^[^:\n]{1,100}\s*[➔→]\s*@?[^:\n]{1,100}\s*\((?:Вопрос|Question)\):\s*", "", res, flags=re.I).strip()
    res = re.sub(r"(?:\n|\s)+TO\s*@[A-Za-zА-Яа-я0-9_\(\)\s]{2,40}$", "", res, flags=re.I).strip()
    return res.strip("«»\"' ")


def _format_money(amount) -> str:
    if amount is None:
        return "—"
    try:
        val = float(amount)
    except (ValueError, TypeError):
        return str(amount)
    if val >= 1_000_000:
        millions = val / 1_000_000
        if millions.is_integer():
            return f"{int(millions)} млн руб."
        return f"{millions:.1f} млн руб."
    if val > 0:
        return f"{int(val):,} руб.".replace(",", " ")
    return "0 руб."


def _claims_full_funding(message: str) -> bool:
    return bool(re.search(
        r"(?:закры(?:ть|вает|ваем|т|ли)\s+(?:весь\s+)?дефицит|"
        r"(?:полностью|в полном объ[её]ме)\s+(?:обеспеч(?:ено|ивает|им)|профинансир)|"
        r"(?:бюджет|сумма)\s+(?:уже\s+)?(?:собран[ао]?|закрыт[ао]?))",
        message, re.I,
    ))


class TwoLLMContractGame(Component):
    display_name = "Two LLM Contract Game"
    description = "Два круга докладов и вопросов, затем оглашение взносов и поимённое голосование."
    icon = "messages-square"

    inputs = [
        BoolInput(
            name="use_rag",
            display_name="Использовать RAG",
            value=True,
        ),
        DataInput(name="rag_config", display_name="Общий RAG", required=False),
        DataInput(name="judge", display_name="Судья совещания", required=True),
        DataInput(
            name="players",
            display_name="Players",
            required=True,
            is_list=True,
        ),
        MultilineInput(
            name="rules_prompt",
            display_name="Инструкция регламента слушаний",
            info="Промпт правил двух кругов докладов, вопросов и голосования.",
            value="",
            required=False,
        ),
    ]

    outputs = [
        Output(
            name="result",
            display_name="Game Result",
            method="run_chat",
            types=["Data"],
        )
    ]

    @staticmethod
    def _parse_vote(text: str):
        clean = text.strip()
        m = re.match(r"^(?:VOTE|ГОЛОСУЮ|ГОЛОС)?\s*(ЗА|ПРОТИВ|ВОЗДЕРЖАЛСЯ|ACCEPT|REJECT|ABSTAIN)[:\s—–-]*(.*)$", clean, re.I | re.S)
        if m:
            decision = m.group(1).upper()
            comment = m.group(2).strip()
            if decision in {"ЗА", "ACCEPT"}:
                return "ЗА", comment
            elif decision in {"ПРОТИВ", "REJECT"}:
                return "ПРОТИВ", comment
            elif decision in {"ВОЗДЕРЖАЛСЯ", "ABSTAIN"}:
                return "ВОЗДЕРЖАЛСЯ", comment
        if "ПРОТИВ" in clean.upper() or "REJECT" in clean.upper():
            return "ПРОТИВ", clean
        elif "ВОЗДЕРЖ" in clean.upper():
            return "ВОЗДЕРЖАЛСЯ", clean
        return "ВОЗДЕРЖАЛСЯ", clean

    def _rag_context(self, speaker, profile, history, phase):
        roles = {
            "Промышленный холдинг": "holding",
            "Администрация города N": "admin",
            "Жители города N": "residents",
            "Региональная власть": "region",
            "Экологическая организация (НКО)": "ngo",
        }
        role = roles[speaker]
        config = self.rag_config.data
        recent = "\n".join(str(h.get("action", {}).get("message", ""))[:900] for h in history[-2:])
        query = render_prompt("rag_query", speaker=speaker, phase=phase,
                              intent=profile.get("intent", ""), recent=recent)[:6000]
        try:
            import chromadb
            from chromadb.config import Settings
            from openai import OpenAI
            key = config['embedding_api_key']
            if hasattr(key, 'get_secret_value'):
                key = key.get_secret_value()
            client = OpenAI(base_url=config['embedding_base_url'], api_key=key, timeout=45, max_retries=1)
            vector = client.embeddings.create(model=config['embedding_model'], input=query).data[0].embedding
            if len(vector) != config['dimension']:
                raise ValueError('Размерность запроса не совпадает с базой')
            db = chromadb.PersistentClient(path=config['chroma_path'], settings=Settings(anonymized_telemetry=False))
            collection = db.get_collection(config['collection_name'], embedding_function=None)
            result = collection.query(query_embeddings=[vector], n_results=config['top_k'],
                                      where={'visible_' + role: True}, include=['documents', 'metadatas', 'distances'])
            chunks = [{'id': cid, 'text': text, 'metadata': meta} for cid, text, meta in
                      zip(result['ids'][0], result['documents'][0], result['metadatas'][0])]
        except Exception as exc:
            raise ValueError(f"Ошибка RAG для {speaker}: {type(exc).__name__}") from None
        self._rag_audit.append({'speaker': speaker, 'phase': phase,
                                'chunk_ids': [c['id'] for c in chunks]})
        parts = [read_prompt("rag_context_intro")]
        for chunk in chunks:
            meta = chunk['metadata']
            parts.append(render_prompt(
                "rag_context_chunk", chunk_id=chunk["id"],
                document_title=meta.get("document_title", ""),
                section_title=meta.get("section_title", ""),
                verification_status=meta.get("verification_status", ""),
                authority_note=meta.get("authority_note", ""),
                text=chunk["text"][:3500],
            ))
        return "\n\n".join(parts)

    def run_chat(self):
        self._rag_audit = []
        prompt_logs = []
        judge_data = getattr(self, "judge", None)
        judge_config = judge_data.data if hasattr(judge_data, "data") else judge_data
        judge = judge_config.get("service") if isinstance(judge_config, dict) else None
        if judge is None:
            raise ValueError("Подключите выход «Судья в ходе совещания» из MeetingJudge.py.")
        judge.prompt_log = prompt_logs
        use_rag = bool(self.use_rag)
        if use_rag and getattr(self, "rag_config", None) is None:
            raise ValueError("Для использования RAG подключите Общий RAG.")

        players = []
        for item in self.players or []:
            profile = item.data if hasattr(item, "data") else item
            name = str(profile.get("player_name", "")).strip()
            prompt = str(profile.get("prompt", "")).strip()
            llm = profile.get("llm")
            if not name or not prompt or llm is None:
                raise ValueError("Каждому участнику нужны имя, промпт роли и модель.")
            players.append({"name": name, "prompt": prompt, "llm": llm, "profile": profile})
        names = [p["name"] for p in players]
        if not players:
            raise ValueError("Для слушаний нужен хотя бы один участник.")
        if len(set(names)) != len(players):
            raise ValueError("Имена участников должны быть уникальными.")

        state = {name: {"contribution": None, "share": None} for name in names}
        history = []
        rules = str(getattr(self, "rules_prompt", "") or "").strip()
        turn = 0

        def record(round_number, speaker, kind, message, target=None, **details):
            nonlocal turn
            turn += 1
            action = {"type": kind, "message": message, **details}
            history.append({"turn": turn, "phase": round_number, "round": round_number,
                            "speaker": speaker, "target": target, "command": message,
                            "action": action, "state": copy.deepcopy(state)})

        def ask(player, round_number, kind, instruction, target=None):
            profile = player["profile"]
            current_block = ""
            if kind in {"QUESTION", "ANSWER"} and target:
                block = []
                for entry in reversed(history):
                    if entry["action"]["type"] == "REPORT" and entry["speaker"] == target:
                        block.append(entry)
                        break
                    block.append(entry)
                current_block = "\n\nТЕКУЩИЙ ДОКЛАД И УЖЕ ЗАДАННЫЕ ВОПРОСЫ:\n" + self._render_history(list(reversed(block)))
            elif kind == "ANSWER":
                block = []
                for entry in reversed(history):
                    if entry["action"]["type"] == "REPORT" and entry["speaker"] == player["name"]:
                        block.append(entry)
                        break
                    block.append(entry)
                current_block = "\n\nТЕКУЩИЙ ДОКЛАД И ВОПРОСЫ:\n" + self._render_history(list(reversed(block)))
            rag_context = ""
            if use_rag and kind in {"REPORT", "PLEDGE"}:
                rag_context = self._rag_context(player["name"], profile, history, round_number)
            if kind == "VOTE":
                scheduled_actions = len(players)
                completed_actions = sum(entry["action"]["type"] == "VOTE" for entry in history)
            else:
                scheduled_actions = len(players) * (len(players) + 2)
                completed_actions = sum(
                    entry["round"] == round_number and entry["action"]["type"] in {
                        "REPORT", "QUESTION", "ANSWER", "SET", "REJECTED"
                    } for entry in history
                )
            remaining_actions = scheduled_actions - completed_actions
            secretary_prompt = render_prompt(
                "secretary_brief", speaker=player["name"], kind=kind,
                round_number=round_number, role_prompt=player["prompt"],
                economic_profile=self._render_profile(profile),
                intent=str(profile.get("intent", "")), rules=rules,
                state=self._render_state(state), history=self._render_history(history),
                current_block=current_block, instruction=instruction,
                rag_context=rag_context, remaining_actions=remaining_actions,
            )
            secretary_system = read_prompt("secretary_system")
            prompt_logs.append({"role": f"Секретарь для {player['name']}",
                                "round": round_number, "action": "BRIEF",
                                "system_prompt": secretary_system,
                                "user_prompt": secretary_prompt})
            secretary_response = player["llm"].invoke([
                SystemMessage(content=secretary_system),
                HumanMessage(content=secretary_prompt),
            ])
            secretary_brief = _extract_text(
                getattr(secretary_response, "content", secretary_response)
            ).strip()[:1500] or "Сводка недоступна; следуй своему профилю и регламенту."
            prompt = render_prompt(
                "participant_turn", role_prompt=player["prompt"],
                economic_profile=self._render_profile(profile),
                other_speeches=self._render_other_speeches(history, player["name"]),
                round_number=round_number, kind=kind, instruction=instruction,
                secretary_brief=secretary_brief,
            )
            system_prompt = read_prompt("participant_system")
            prompt_entry = {"role": player["name"], "round": round_number,
                            "action": kind, "system_prompt": system_prompt, "user_prompt": prompt}
            prompt_logs.append(prompt_entry)
            PROMPT_LOGGER.warning("ПРОМПТ УЧАСТНИКА %s | круг %s | %s\nSYSTEM:\n%s\nUSER:\n%s",
                                  player["name"], round_number, kind, system_prompt, prompt)
            response = player["llm"].invoke([
                SystemMessage(content=system_prompt), HumanMessage(content=prompt),
            ])
            return _clean_command(_extract_text(getattr(response, "content", response)))

        rounds_completed = 0
        announcements = []
        for round_number in (1, 2):
            if round_number == 2 and announcements[-1]["next_stage"] != "second_round":
                break
            rounds_completed = round_number
            record(round_number, "Судья", "JUDGE",
                   judge.speak(judge_event("open_round"),
                               {"round": round_number, "participants": names,
                                "target_rub": TOTAL_PROJECT_COST,
                                "agenda": "реконструкция очистных сооружений и безопасная вода"}, history))

            for speaker in players:
                record(round_number, "Судья", "JUDGE",
                       judge.speak(judge_event("give_floor"),
                                   {"round": round_number, "speaker": speaker["name"]}, history))
                report_instruction = read_prompt("report")
                if round_number == 2:
                    report_instruction += "\n" + read_prompt("report_final_round")
                report = ask(speaker, round_number, "REPORT",
                             report_instruction)
                record(round_number, speaker["name"], "REPORT", _clean_speech_message(report))

                questions = []
                question_focus = read_prompt("question_focus").splitlines()
                for question_index, participant in enumerate(p for p in players if p["name"] != speaker["name"]):
                    question = ask(participant, round_number, "QUESTION",
                                   render_prompt("question", speaker=speaker["name"],
                                                 focus=question_focus[question_index % len(question_focus)]),
                                   target=speaker["name"])
                    clean_question = _clean_speech_message(question)
                    questions.append(f"{participant['name']}: {clean_question}")
                    record(round_number, participant["name"], "QUESTION", clean_question,
                           target=speaker["name"])

                answer = ask(speaker, round_number, "ANSWER",
                             render_prompt("answer", questions="\n".join(questions)))
                record(round_number, speaker["name"], "ANSWER", _clean_speech_message(answer))

            record(round_number, "Судья", "JUDGE",
                   judge.speak(judge_event("invite_pledges"),
                               {"round": round_number, "speakers": names}, history))
            for player in players:
                previous = state[player["name"]]["contribution"]
                previous_text = _format_money(previous) if previous is not None else "ещё не заявлен"
                pledge = ask(player, round_number, "PLEDGE",
                             render_prompt("pledge", previous=previous_text))
                parsed = self._parse(pledge, player["name"], state,
                                     {p["name"]: p["profile"] for p in players})
                if parsed.get("type") != "SET":
                    verdict = {"accepted": False, "remark": "Взнос не назван в формате SET; ход пропущен."}
                else:
                    verdict = judge.review(player["name"], round_number, parsed,
                                           player["profile"], history=history, state=state)
                if verdict["accepted"]:
                    value = parsed["value"]
                    contribution = value if parsed["field"] == "contribution" else round(value / 100 * TOTAL_PROJECT_COST, 2)
                    state[player["name"]]["contribution"] = contribution
                    state[player["name"]]["share"] = round(contribution / TOTAL_PROJECT_COST * 100, 2)
                    record(round_number, player["name"], "SET", parsed["action"].get("message", ""),
                           contribution=contribution, share=state[player["name"]]["share"],
                           judge_remark=verdict["remark"])
                    remaining = max(0, TOTAL_PROJECT_COST - sum(
                        item["contribution"] or 0 for item in state.values()))
                    if remaining and _claims_full_funding(parsed["action"].get("message", "")):
                        record(round_number, "Судья", "JUDGE",
                               judge.speak(judge_event("correct_funding"),
                                           {"speaker": player["name"], "deficit_rub": remaining,
                                            "pledge_accepted": True}, history))
                else:
                    record(round_number, player["name"], "REJECTED", pledge,
                           judge_remark=verdict["remark"])

            announcement = judge.choose_stage(round_number, state, TOTAL_PROJECT_COST, history=history)
            announcements.append(announcement)
            record(round_number, "Судья", "JUDGE", announcement["remark"])

        total_money = announcements[-1]["total"]
        deficit = announcements[-1]["deficit"]
        voting_records = []
        voting_held = announcements[-1]["next_stage"] == "vote"
        if voting_held:
            for player in players:
                vote_text = ask(player, 3, "VOTE",
                                render_prompt("vote", total=_format_money(total_money),
                                              deficit=_format_money(deficit)))
                decision, comment = self._parse_vote(vote_text)
                comment = _clean_speech_message(comment)
                voting_records.append({"player": player["name"], "vote": decision, "comment": comment})
                record(3, player["name"], "VOTE", comment, vote=decision, comment=comment)
                if deficit and _claims_full_funding(comment):
                    record(3, "Судья", "JUDGE",
                           judge.speak(judge_event("correct_vote"),
                                       {"speaker": player["name"], "deficit_rub": deficit}, history))

        votes_for = sum(v["vote"] == "ЗА" for v in voting_records)
        votes_against = sum(v["vote"] == "ПРОТИВ" for v in voting_records)
        votes_abstain = sum(v["vote"] == "ВОЗДЕРЖАЛСЯ" for v in voting_records)
        passed = voting_held and judge.decide(votes_for, len(players), total_money, TOTAL_PROJECT_COST)
        summary = {"records": voting_records, "votes_for": votes_for,
                   "votes_against": votes_against, "votes_abstain": votes_abstain,
                   "quorum": len(players), "passed": passed, "held": voting_held}
        if voting_held:
            record(3, "Судья", "JUDGE",
                   judge.speak(judge_event("final_decision"),
                               {"passed": passed, "votes_for": votes_for,
                                "votes_against": votes_against, "votes_abstain": votes_abstain,
                                "total_rub": total_money, "deficit_rub": deficit}, history))

        serializable_profiles = []
        for player in players:
            profile = copy.copy(player["profile"])
            profile.pop("llm", None)
            serializable_profiles.append(profile)
        dialogue_text = self._render_dialogue(history, state, summary, total_money, deficit)
        return Data(text=dialogue_text, data={
            "history": history, "rag_retrievals": self._rag_audit,
            "prompt_logs": prompt_logs,
            "players": serializable_profiles, "state": copy.deepcopy(state),
            "voting_summary": summary, "consensus_reached": passed,
            "total_money": total_money, "deficit": deficit,
            "rounds_completed": rounds_completed, "announcements": announcements,
        })

    @staticmethod
    def _parse(command, speaker, state, profiles):
        # Some models put both protocol commands on one line despite the prompt.
        text = re.sub(r";\s*(?=(?:SPEAK|СКАЗАТЬ)\s)", "\n", command.strip(), flags=re.I)
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        if not lines:
            return {"valid": False, "error": "Пустой ответ."}

        set_line = None
        speak_line = None
        other_lines = []

        for line in lines:
            line_clean = re.sub(r"^\s*[\+\*]\s*", "", line).strip()
            upper = line_clean.upper()
            if upper.startswith("SET "):
                set_line = line_clean
            elif upper.startswith("SPEAK ") or upper.startswith("СКАЗАТЬ "):
                speak_line = line_clean
            elif upper.startswith("TO ") or upper.startswith("ПЕРЕДАТЬ "):
                pass
            else:
                other_lines.append(line)

        if set_line:
            clean_cmd = set_line.replace('"', '').replace("'", '').replace('«', '').replace('»', '')
            match = re.match(r"^SET\s+(.+?)\s+(доля|взнос)\s+([-+]?\d+(?:[.,]\d+)?%?)(?:\s+(.*))?$", clean_cmd, re.I)

            if match:
                raw_player, field_raw, raw_value, tail_msg = match.groups()
                raw_player = raw_player.strip()

                if raw_player.lower() in {"me", "myself", "я", "мне", "себе"}:
                    player = speaker
                else:
                    matched = None
                    for sname in state:
                        if sname.casefold() == raw_player.casefold():
                            matched = sname
                            break
                    if not matched:
                        candidates = [sname for sname in state if raw_player.casefold() in sname.casefold() or sname.casefold() in raw_player.casefold()]
                        if len(candidates) == 1:
                            matched = candidates[0]
                    player = matched or speaker

                field_map = {"доля": "share", "взнос": "contribution"}
                field = field_map.get(field_raw.lower(), "contribution")

                try:
                    raw_num = raw_value.replace("%", "").replace(",", ".").replace(" ", "")
                    value = float(raw_num)
                    if field == "contribution" and 0 < value <= 1000:
                        value = value * 1_000_000.0
                except ValueError:
                    value = 0.0

                msg_parts = []
                if speak_line:
                    sp_match = re.match(r"^(?:SPEAK|СКАЗАТЬ)\s+(.+)$", speak_line, re.I | re.S)
                    if sp_match:
                        msg_parts.append(sp_match.group(1).strip())
                if tail_msg and tail_msg.strip():
                    tm = tail_msg.strip()
                    if not re.match(r"^(?:млн|млн\.|руб|руб\.|рублей|тыс|тыс\.|%|процентов)(?:\s+(?:руб|руб\.|рублей))?$", tm, re.I):
                        tm = re.sub(r"^(?:млн|млн\.|руб|руб\.|рублей|тыс|тыс\.|%|процентов)(?:\s+(?:руб|руб\.|рублей))?\s*[:\-\—]?\s*", "", tm, flags=re.I).strip()
                        if tm:
                            msg_parts.append(tm)
                if not msg_parts and other_lines:
                    msg_parts.append(" ".join(other_lines))

                final_msg = _clean_speech_message(" ".join(msg_parts))

                return {
                    "valid": True,
                    "type": "SET",
                    "player": player,
                    "field": field,
                    "value": value,
                    "action": {
                        "type": "SET",
                        "player": player,
                        "field": field,
                        "value": value,
                        "message": final_msg,
                    },
                }

        if speak_line:
            match = re.match(r"^(?:SPEAK|СКАЗАТЬ)\s+(.+)$", speak_line, re.I | re.S)
            msg = match.group(1).strip() if match else ""
            if other_lines:
                msg += " " + " ".join(other_lines)
            return {
                "valid": True,
                "type": "SPEAK",
                "action": {"type": "SPEAK", "message": _clean_speech_message(msg)},
            }

        fallback_text = _clean_speech_message(" ".join(lines))
        return {
            "valid": True,
            "type": "SPEAK",
            "action": {"type": "SPEAK", "message": fallback_text},
        }

    @staticmethod
    def _render_profile(profile):
        attrs = profile.get("attrs", {})
        return (
            f"Доступный бюджет: {_format_money(profile.get('money', 0))}\n"
            f"Минимально приемлемая доля: {attrs.get('min_share', 0):g}%\n"
            f"Комфортная доля: {attrs.get('comfort_share', 0):g}%\n"
            f"Комфортный взнос: {_format_money(attrs.get('comfort_money', 0))}\n"
            f"Максимальный допустимый взнос: {_format_money(attrs.get('max_money', 0))}\n"
            f"Красные линии: {', '.join(attrs.get('red_lines', [])) if attrs.get('red_lines') else 'Нет жестких ограничений'}"
        )

    @staticmethod
    def _render_state(state):
        lines = []
        total_c = 0
        total_s = 0
        for name, data in state.items():
            c = data["contribution"]
            s = data["share"]
            c_str = _format_money(c)
            s_str = f"{s:g}%" if s is not None else "—"
            if c:
                total_c += c
            if s:
                total_s += s
            lines.append(f"• {name}: Взнос: {c_str} | Доля: {s_str}")
        lines.append(f"  ИТОГО СОБРАНО: {_format_money(total_c)} (из 1 000 млн руб.) | Распределено: {total_s:g}%")
        deficit = max(0.0, TOTAL_PROJECT_COST - total_c)
        if deficit > 1000:
            lines.append(f"  ⚠️ ДЕФИЦИТ ДО 1 000 МЛН: {_format_money(deficit)} (БЮДЖЕТ ЕЩЕ НЕ СОБРАН!)")
        else:
            lines.append("  ✅ ДЕФИЦИТ: 0 руб. (Бюджет 100% собран!)")
        return "\n".join(lines)

    @staticmethod
    def _render_other_speeches(history, listener):
        lines = [
            f"[{entry['turn']}] {entry['speaker']}: {entry['action']['message']}"
            for entry in history
            if entry.get("speaker") not in {listener, "Судья"}
            and entry.get("action", {}).get("type") in {
                "REPORT", "QUESTION", "ANSWER", "SET", "REJECTED", "VOTE"
            }
            and entry.get("action", {}).get("message")
        ]
        return "\n".join(lines) or "(другие участники пока не выступали)"

    @staticmethod
    def _render_history(history):
        if not history:
            return "(начало переговоров)"
        blocks = []
        for item in history:
            action = item.get("action", {})
            act_type = action.get("type")
            speaker = item.get("speaker")
            target = item.get("target")
            target_str = f" ➔ @{target}" if target else ""
            t_num = item.get("turn")

            if act_type in {"SPEAK", "REPORT", "QUESTION", "ANSWER"}:
                label = {"REPORT": "Доклад", "QUESTION": "Вопрос", "ANSWER": "Ответ"}.get(act_type, "Реплика")
                blocks.append(f"[{t_num}] {speaker}{target_str} ({label}): «{action.get('message', '')}»")
            elif act_type == "SET":
                c_val = action.get("contribution")
                s_val = action.get("share", 0)
                details = f"{_format_money(c_val)} ({s_val:g}%)"
                msg = action.get("message", "").strip()
                if msg:
                    blocks.append(f"[{t_num}] {speaker}{target_str}: Взнос {details} | «{msg}»")
                else:
                    blocks.append(f"[{t_num}] {speaker}{target_str}: Взнос {details}")
            elif act_type == "VOTE":
                blocks.append(f"[{t_num}] {speaker}: VOTE {action.get('vote')} — «{action.get('comment', '')}»")
            elif act_type == "JUDGE":
                blocks.append(f"Судья: {action.get('message', '')}")
            elif act_type == "REJECTED":
                blocks.append(f"[{t_num}] {speaker}: доклад отклонён; замечание судьи: {action.get('judge_remark', '')}")
            else:
                blocks.append(f"[{t_num}] {speaker}: {item.get('command')}")
        return "\n".join(blocks)

    @staticmethod
    def _render_dialogue(history, state, voting_summary, total_money, deficit):
        lines = [
            "# Протокол общественных слушаний города N",
            "**Повестка:** реконструкция очистных сооружений и ликвидация экологического кризиса",
            f"**Участники:** {', '.join(state)}",
        ]
        for phase in (1, 2, 3):
            entries = [h for h in history if h.get("phase") == phase]
            if not entries:
                continue
            title = f"Круг {phase}: доклады, вопросы и взносы" if phase < 3 else "Поимённое голосование"
            lines.append(f"\n## {title}\n")
            for entry in entries:
                speaker = entry["speaker"]
                target = entry.get("target")
                action = entry.get("action", {})
                kind = action.get("type")
                message = action.get("message", "")
                number = entry["turn"]
                if kind == "JUDGE":
                    lines.append(f"**Судья:** {message}\n")
                elif kind == "REPORT":
                    lines.append(f"**[{number}] Доклад — {speaker}:** {message}\n")
                elif kind == "QUESTION":
                    lines.append(f"**[{number}] Вопрос — {speaker} → {target}:** {message}\n")
                elif kind == "ANSWER":
                    lines.append(f"**[{number}] Ответ — {speaker}:** {message}\n")
                elif kind == "SET":
                    amount = _format_money(action.get("contribution"))
                    lines.append(f"**[{number}] Взнос — {speaker}: {amount}** {message}\n")
                elif kind == "REJECTED":
                    lines.append(f"**[{number}] Заявление — {speaker}:** {message}\n"
                                 f"**Замечание судьи:** {action.get('judge_remark', '')}\n")
                elif kind == "VOTE":
                    lines.append(f"**[{number}] {speaker} — {action.get('vote')}:** {action.get('comment', '')}\n")

        passed = voting_summary["passed"]
        lines.append("\n## Итоговое решение")
        status = "протокол принят" if passed else (
            "протокол отклонён" if voting_summary["held"] else "слушания завершены без голосования")
        lines.append(f"**Статус:** {status}.")
        if voting_summary["held"]:
            lines.append(f"**Голоса:** за {voting_summary['votes_for']}, против {voting_summary['votes_against']}, "
                         f"воздержались {voting_summary['votes_abstain']}.")
        lines.append(f"**Финансирование:** {_format_money(total_money)} из 1 000 млн руб.; "
                     f"дефицит {_format_money(deficit)}\n")
        lines.extend(["| Участник | Засчитанный взнос | Доля |", "|---|---:|---:|"])
        for name, item in state.items():
            share = item.get("share")
            lines.append(f"| {name} | {_format_money(item.get('contribution'))} | "
                         f"{share:g}% |" if share is not None else
                         f"| {name} | — | — |")
        heading = "Утверждённые условия" if passed else "Предложенные условия, не вступившие в силу"
        lines.extend([
            f"\n### {heading}",
            "- Реконструкция очистных сооружений за 24 месяца.",
            "- Поставка безопасной воды жителям до ввода сооружений и заморозка тарифов ЖКХ.",
            "- Независимый мониторинг воды с открытой публикацией результатов.",
            "- Участие жителей и экологической НКО в общественном контроле.",
        ])
        return "\n".join(lines)
