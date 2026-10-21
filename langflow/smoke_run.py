"""One complete five-player game through the Langflow model adapter."""

import json
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from my_components.openai_compatible_model import OpenAICompatibleModel
from negotiation_mvp.langflow_bridge import run_langflow_game


def main():
    config = json.loads(Path("/app/negotiation_flow/demo_config.json").read_text(encoding="utf-8"))
    config["max_rounds"] = 1
    component = OpenAICompatibleModel()
    component.base_url = "http://host.docker.internal:9090/v1"
    component.model_name = "qwen3.8-9b-q4"
    component.timeout_seconds = 360
    component.max_tokens = 512
    model = component.build_model()
    classes = {"system": SystemMessage, "user": HumanMessage, "assistant": AIMessage}

    counter = 0

    def invoke(messages):
        nonlocal counter
        counter += 1
        print(f"Model call {counter} started", flush=True)
        answer = model.invoke([classes[item["role"]](content=item["content"]) for item in messages])
        print(f"Model call {counter} finished", flush=True)
        return str(answer.content).strip()

    invokers = {player["id"]: invoke for player in config["players"]}
    invokers["__converter__"] = invoke
    output = run_langflow_game(config, config["players"], invokers, Path("/app/games"))
    print(json.dumps({"rounds": len(output["result"]["rounds"]), "votes": output["result"]["votes"],
                      "accepted": output["result"]["accepted"], "files": output["files"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
