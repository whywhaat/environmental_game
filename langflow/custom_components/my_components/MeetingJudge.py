import json
import logging

from langchain_core.messages import HumanMessage, SystemMessage
from lfx.base.models.unified_models import get_llm
from lfx.custom import Component
from lfx.inputs.inputs import ModelInput
from lfx.io import MultilineInput, Output
from lfx.schema.data import Data
from config.prompts import judge_event, read_prompt, render_prompt

PROMPT_LOGGER = logging.getLogger("environmental_game.meeting.prompts")


def _extract_text(content) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        return "".join(
            str(item.get("text", "")) if isinstance(item, dict) else str(getattr(item, "text", item))
            for item in content
        ).strip()
    return str(content).strip()


class MeetingJudgeService:
    """Checks live pledges and records the public financial result."""

    def __init__(self, llm, instruction=""):
        self.llm = llm
        self.instruction = instruction.strip()
        self.prompt_log = []

    def _log_prompt(self, action, phase, system_prompt, user_prompt):
        self.prompt_log.append({"role": "Судья", "round": phase,
                                "action": action, "system_prompt": system_prompt,
                                "user_prompt": user_prompt})
        PROMPT_LOGGER.warning("ПРОМПТ СУДЬИ | круг %s | %s\nSYSTEM:\n%s\nUSER:\n%s",
                              phase, action, system_prompt, user_prompt)

    def _secretary_brief(self, action, phase, facts, history):
        recent = "\n".join(
            f"{entry.get('speaker', '')} ({entry.get('action', {}).get('type', '')}): "
            f"{entry.get('action', {}).get('message', '')}"
            for entry in history or []
        ) or "(начало заседания)"
        prompt = render_prompt(
            "judge_secretary_brief", action=action,
            facts=json.dumps(facts, ensure_ascii=False, default=str), history=recent,
        )
        system_prompt = read_prompt("judge_secretary_system")
        self.prompt_log.append({"role": "Секретарь судьи", "round": phase,
                                "action": "BRIEF", "system_prompt": system_prompt,
                                "user_prompt": prompt})
        response = self.llm.invoke([
            SystemMessage(content=system_prompt), HumanMessage(content=prompt),
        ])
        return _extract_text(getattr(response, "content", response)).strip()[:1500] or "Факты и ограничения указаны ниже."

    def speak(self, event, facts, history=None):
        recent = []
        for entry in history or []:
            action = entry.get("action", {})
            recent.append(f"{entry.get('speaker', '')} ({action.get('type', '')}): "
                          f"{str(action.get('message', ''))}")
        prompt = render_prompt(
            "judge_speech", instruction=self.instruction, event=event,
            facts=json.dumps(facts, ensure_ascii=False, default=str),
            recent="\n".join(recent) or "(начало заседания)",
            secretary_brief=self._secretary_brief(event, facts.get("round"), facts, history),
        )
        system_prompt = read_prompt("judge_system")
        self._log_prompt(event, facts.get("round"), system_prompt, prompt)
        response = self.llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=prompt),
        ])
        message = _extract_text(getattr(response, "content", response)).strip()
        if not message or message.startswith("{"):
            raise ValueError(f"Судья не сформулировал реплику для события {event}.")
        return message[:1200]

    def review(self, speaker, phase, parsed, profile, history=None, state=None):
        action = parsed.get("action", {})
        if not parsed.get("valid"):
            return {"accepted": False, "remark": parsed.get("error", "Некорректное предложение.")}
        if parsed.get("type") != "SET":
            return {"accepted": True, "remark": "Доклад принят без изменения взноса."}
        if parsed.get("player") != speaker:
            return {"accepted": False, "remark": "Участник может менять только собственный взнос."}
        value = float(parsed.get("value") or 0)
        if parsed.get("field") == "share":
            value *= 10_000_000
        limit = float(profile.get("attrs", {}).get("max_money", profile.get("money", 0)))
        if value < 0 or value > limit:
            return {"accepted": False, "remark": "Взнос выходит за доступный предел участника."}
        message = str(action.get("message", "")).strip()
        if len(message.split()) < 8:
            return {"accepted": False, "remark": "Взнос не засчитан: назовите причину и условия предложения."}
        recent = []
        for entry in history or []:
            event = entry.get("action", {})
            kind = event.get("type")
            if kind not in {"REPORT", "QUESTION", "ANSWER", "SET", "REJECTED", "JUDGE"}:
                continue
            line = f"{entry.get('speaker', '')} ({kind}): {event.get('message', '')}"
            recent.append(line)
        ledger = {
            name: item.get("contribution")
            for name, item in (state or {}).items()
        }
        discussion = "\n".join(recent) or "(пока нет)"
        prompt = render_prompt(
            "judge_review", instruction=self.instruction, speaker=speaker,
            phase=phase, limit=f"{limit:g}", ledger=json.dumps(ledger, ensure_ascii=False),
            discussion=discussion, action=json.dumps(action, ensure_ascii=False, default=str),
            secretary_brief=self._secretary_brief(
                "проверка взноса", phase,
                {"speaker": speaker, "limit_rub": limit, "ledger": ledger,
                 "proposal": action}, history,
            ),
        )
        try:
            system_prompt = read_prompt("judge_system")
            self._log_prompt("проверка взноса", phase, system_prompt, prompt)
            response = self.llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=prompt),
            ])
            raw = _extract_text(getattr(response, "content", response))
            result = json.loads(raw.removeprefix("```json").removesuffix("```").strip())
            if isinstance(result.get("accepted"), bool):
                return {"accepted": result["accepted"], "remark": str(result.get("remark", ""))[:300]}
        except (ValueError, TypeError, AttributeError):
            pass
        return {"accepted": False, "remark": "Судья не смог проверить обоснование; взнос не засчитан."}

    def announce(self, state, target, history=None):
        total = sum(item.get("contribution") or 0 for item in state.values())
        deficit = max(0, target - total)
        return {
            "total": total,
            "deficit": deficit,
            "remark": self.speak(judge_event("announce_pledges"),
                                 {"contributions_rub": {
                                     name: item.get("contribution") for name, item in state.items()
                                 }, "total_rub": total, "target_rub": target,
                                     "deficit_rub": deficit}, history),
        }

    def choose_stage(self, round_number, state, target, history=None):
        contributions = {name: item.get("contribution") for name, item in state.items()}
        total = sum(value or 0 for value in contributions.values())
        deficit = max(0, target - total)
        allowed = ["vote", "close"]
        if round_number == 1 and deficit > 0:
            allowed.insert(0, "second_round")
        recent = "\n".join(
            f"{entry.get('speaker', '')} ({entry.get('action', {}).get('type', '')}): "
            f"{entry.get('action', {}).get('message', '')}"
            for entry in history or []
        ) or "(начало заседания)"
        prompt = render_prompt(
            "judge_transition", instruction=self.instruction,
            allowed=", ".join(allowed), round_number=round_number,
            contributions=json.dumps(contributions, ensure_ascii=False),
            total=f"{total:g}", target=f"{target:g}", deficit=f"{deficit:g}",
            history=recent,
            secretary_brief=self._secretary_brief(
                "выбор следующего этапа", round_number,
                {"allowed": allowed, "contributions_rub": contributions,
                 "total_rub": total, "target_rub": target, "deficit_rub": deficit},
                history,
            ),
        )
        system_prompt = read_prompt("judge_system")
        self._log_prompt("выбор этапа", round_number, system_prompt, prompt)
        response = self.llm.invoke([
            SystemMessage(content=system_prompt), HumanMessage(content=prompt),
        ])
        raw = _extract_text(getattr(response, "content", response)).strip()
        try:
            decision = json.loads(raw.removeprefix("```json").removesuffix("```").strip())
        except (ValueError, TypeError) as exc:
            raise ValueError("Судья не выбрал следующий этап в формате JSON.") from exc
        if not isinstance(decision, dict) or decision.get("next_stage") not in allowed:
            raise ValueError(f"Судья выбрал недопустимый следующий этап: {decision!r}")
        remark = decision.get("remark")
        if not isinstance(remark, str) or not remark.strip():
            raise ValueError("Судья не объявил решение о следующем этапе.")
        if decision["next_stage"] != "close" and any(
            phrase in remark.casefold() for phrase in (
                "заседание окончено", "заседание завершено", "заседание закрыто",
                "слушания окончены", "слушания завершены", "закрываю заседание",
            )
        ):
            raise ValueError("Речь судьи завершает заседание, но выбран следующий этап.")
        return {"total": total, "deficit": deficit,
                "next_stage": decision["next_stage"], "remark": remark.strip()[:1200]}

    @staticmethod
    def decide(votes_for, participant_count, total, target):
        return votes_for > participant_count / 2 and total >= target


class MeetingJudge(Component):
    display_name = "Судья заседания"
    description = "Проверяет взносы и оглашает результат во время слушаний."
    icon = "gavel"

    inputs = [
        ModelInput(
            name="model", display_name="Judge Model", required=True,
            model_type="language", is_list=False,
        ),
        MultilineInput(
            name="instruction",
            display_name="Инструкция председателя",
            info="Манера речи председателя и критерии проверки взносов",
            value="",
            required=False,
        ),
    ]

    outputs = [
        Output(name="judge", display_name="Судья в ходе совещания",
               method="build_judge", types=["Data"]),
    ]

    def build_judge(self) -> Data:
        llm = get_llm(model=self.model, user_id=self.user_id)
        return Data(data={"service": MeetingJudgeService(
            llm, str(getattr(self, "instruction", "") or "").strip() or read_prompt("judge_instruction")
        )})
