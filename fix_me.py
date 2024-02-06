from tatara import tatara, Eval, Record, init_dataset, run_evals
from openai import OpenAI
from tatara import Record

model_name = "gpt-3.5-turbo-1106"
freq_penalty = 0.1
temperature = 0.1
max_tokens = 200
    openai_client = OpenAI()


def create_dataset():
    ds = init_dataset("test_dataset")
    new_records = [
    {"input":"Generate a description of a charcater named Cyborg Guy. Cyborg Guy is a cyberpunk robot who lives in the city of Zail.","output":"""You are Cyborg Guy, a cyborg living in the the futuristic city of Zail. You have a bionic arm and a holoband. You're walking down the dark city streets while neon lights flash brightly above you. As you pass an alley you seeYou are Cyborg Guy, a cyborg living in the the futuristic city of Zail. You have a bionic arm and a holoband. You're walking down the dark city streets while neon lights flash brightly above you. As you pass an alley you see
    a figure huddled in the corner, trying to stay hidden in the shadows. They look up as you approach, and you can see that they are wearing tattered clothes and have a gaunt, hungry appearance. They extend a hand towards you, pleading for help."""
    }
    ]
    ds.insert(new_records)



def is_cyberpunk_check(record: Record) -> bool:
    prompt = "Check that the following text is an example of cyberpunk fiction: \n\n {}\n\n If it is cyberpunk, write <CYBERPUNK>. If it is not, write <NOT_CYBERPUNK>.".format(record['output'])
    response = openai_client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model=model_name,
        temperature=temperature,
        frequency_penalty=freq_penalty,
        max_tokens=max_tokens,

    )
    ret = response.choices[0].message.content
    if "<CYBERPUNK>" in ret:
        return True
    elif "<NOT_CYBERPUNK>" in ret:
        return False
    else:
        # If the model doesn't output a clear answer, we it again until it does
        return is_cyberpunk_check(record)

if __name__ == "__main__":
    my_test_project = "test_project"
    tatara.init(project=my_test_project, flush_interval=1, queue_size=1)

    create_dataset()
    
    cyberpunk_check = Eval(
            name = "is_cyberpunk_check",
            description = "Checks whether the text is cyberpunk fiction. If it is, it returns True. If it is not, it returns False. If it is unsure, it will call itself again.",
            eval_fn = is_cyberpunk_check,
        )

    run_evals([cyberpunk_check], "test_dataset", print_only=True)
