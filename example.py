from openai import OpenAI
from tatara.client import LLMParams, LLMUsageMetrics, ProviderEnum
from tatara import client as tatara_client
import uuid
import os

# set your api key as an environment variable at TATARA_API_KEY
os.environ["TATARA_API_KEY"] = "<your api key here>"

# pick a name for your project
project_name = "<your project name here>"

# initialize the client. By default, logs flush every 60 seconds to the server or hits the max queue size (200 for SW, 1000 by default).
# Here, we set the queue size to 1 and the flush interval to 1 second so we can see the logs immediately.
tatara_client.init(project=project_name, queue_size=1, flush_interval=1)

"""
Tatara has a concept of a trace, which is a collection of spans. A trace is a logical unit of work that can be used to group spans together.
Because all of our spans are model calls, you can think of a trace as a "chain" of model calls.

Below, I'll be making two calls to OpenAI's API. The first call will ask ChatGPT to write a poem in the style of William Carlos Williams. 
The second call will ask ChatGPT to critique that poem as Michiko Kakutani (a former NYT literary critic).
"""

### Call 1

openai_client = OpenAI()
model_name = "gpt-3.5-turbo-1106"
freq_penalty = 0.1
temperature = 0.1
max_tokens = 200


def openai_completion_with_logging(
    prompt: str,
    model_name: str,
    temperature: float,
    freq_penalty: float,
    max_tokens: int,
):
    response = openai_client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model=model_name,
        temperature=temperature,
        frequency_penalty=freq_penalty,
        max_tokens=max_tokens,
    )
    ret = response.choices[0].message.content
    if ret is None:
        raise Exception("OpenAI returned 0 length response")

    # calling current_span will get the span that is currently active
    # this is useful if we're several calls deep in the stack and want to log something
    # this way we don't have to pass around the span object
    tatara_client.current_span().log_llm_success(
        prompt=prompt,
        output=ret,
        params=LLMParams(
            model=model_name,
            temperature=temperature,
            frequency_penalty=freq_penalty,
            max_tokens=max_tokens,
            provider=ProviderEnum.OPENAI,
        ),
        usage_metrics=LLMUsageMetrics.from_oai_completion_usage(response.usage)
        if response.usage
        else None,
    )
    return ret


trace_name = "poem_and_critique"
uuid4: str = str(uuid.uuid4())

# I'll specify my own id for the trace via the id_ param so I can search for it in the UI later on.
# I'll also specify the user_id param, which is _your_ user_id, so you know which of your users is responsible for this interaction.
with tatara_client.start_trace(event=trace_name, id_=uuid4, user_id="user-123"):
    with tatara_client.start_span(event="write_poem"):
        poem_prompt = "Write me a poem in the style of William Carlos Williams"
        poem = openai_completion_with_logging(
            poem_prompt, model_name, temperature, freq_penalty, max_tokens
        )
    with tatara_client.start_span(event="write_critique"):
        poem_critique_prompt = (
            "Write me a critique of this poem in the style of Michiko Kakutani: "
            + poem_prompt
        )
        critique = openai_completion_with_logging(
            poem_critique_prompt, model_name, temperature, freq_penalty, max_tokens
        )
