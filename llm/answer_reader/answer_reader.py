from llm.answer import get_answer

def read_answer(llm_answer_path, llm_response_path, label, llm_client):
    # Always call the LLM (cache bypass) — IsaacSim FFW-SG2 fork
    llm_answer, response = get_answer(prompt=label, client=llm_client)
    print(f"[Fresh LLM] Answer for {label}: {llm_answer}")

    with open(llm_response_path, "a+") as response_file:
        response_file.write(f"\n{label}: {response}")

    if not llm_answer or not isinstance(llm_answer, list) or len(llm_answer) < 3:
        raise ValueError(f"LLM returned malformed answer: {llm_answer}")

    if isinstance(llm_answer[-1], str):
        room = llm_answer[-1]
        llm_answer.pop()
    else:
        raise ValueError("Room answer is not correct!!!!")

    if isinstance(llm_answer[-1], float):
        fusion_score = llm_answer[-1]
        llm_answer.pop()
    else:
        raise ValueError("Score answer is not correct!!!!")

    return llm_answer, room, fusion_score
