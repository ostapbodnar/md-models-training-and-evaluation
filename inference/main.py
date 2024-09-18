import asyncio
import re
from difflib import SequenceMatcher
from typing import Literal, Callable

import numpy as np
from huggingface_hub import AsyncInferenceClient
from tqdm import tqdm

from evaluation.annotation import TaggedText
from inference.constatns import penalty_mapping
from inference.instructs import TAGGING_INSTRUCT, KEYWORD_INSTRUCT

concurrency_limit = 15
semaphore = asyncio.Semaphore(concurrency_limit)


def get_chunks(text: str, maxlength: int):
    while len(text) >= maxlength:
        split_index = text.rfind('.', 0, maxlength) + 1
        if split_index == 0:
            split_index = maxlength
        yield text[:split_index].strip()
        text = text[split_index:].strip()
    yield text


def create_llm_task(client, type: Literal['keywords', 'grammar'], model_name: str, max_tokens: int,
                    progress_bar) -> Callable:
    instruct = TAGGING_INSTRUCT if type == 'grammar' else KEYWORD_INSTRUCT

    async def process_text(index, input_text):
        async with semaphore:
            input_request = f'{instruct} \n Input: {input_text}\n Output:'
            messages = [
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": input_request},
            ]

            try:
                response = await client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    max_tokens=max_tokens,
                    top_p=0.9,
                    temperature=0.2,
                    frequency_penalty=1.7,
                    stop=["\nUser:", "<|endoftext|>", "</s>", "\n"],
                )
                generated_text = response.choices[0].message.content
            except Exception as e:
                print(e)
                generated_text = None

            progress_bar.update(1)
            return index, generated_text

    return process_text


def extract_list_from_text(text):
    match = re.search(r"\[(.*?)\]", text)
    if match:
        extracted_list = match.group(1).split(',')
        parsed_list = [item.strip().strip("'\"").lower() for item in extracted_list]
        return parsed_list
    else:
        return []


async def _eval_model(client: AsyncInferenceClient, text: str, max_tokens, model_name):
    chunked_text = list(get_chunks(text, max_tokens))
    max_tokens = int(max_tokens * 2)

    gr_progress_bar = tqdm(total=len(chunked_text), desc=f'Processing grammar task')
    kw_progress_bar = tqdm(total=len(chunked_text), desc=f'Processing keywords task')

    grammar_processor = create_llm_task(client, 'grammar', model_name, max_tokens, gr_progress_bar)
    keywords_processor = create_llm_task(client, 'keywords', model_name, max_tokens, kw_progress_bar)

    gr_tasks = [
        grammar_processor(i, data) for i, data in enumerate(chunked_text)
    ]
    kw_tasks = [
        keywords_processor(i, data) for i, data in enumerate(chunked_text)
    ]
    gr_res, kw_res = await asyncio.gather(asyncio.gather(*gr_tasks), asyncio.gather(*kw_tasks))

    gr_progress_bar.close()
    kw_progress_bar.close()

    score = np.mean(
        [score_grammar(chunked_text[index], text) for index, text in gr_res])
    print("Text score: ", score)
    print("=" * 100)

    for _, text in gr_res:
        print(TaggedText(text).get_colored_tagged_text(), end=" ")
    print("", "=" * 100)

    keywords = []
    for _, text in kw_res:
        keywords.extend(extract_list_from_text(text))
    print("Keywords: ", sorted(set(keywords)))

    print("\n" * 10)
    for _, text in gr_res:
        print(text, end=" ")
    print("", "=" * 100)




def score_grammar(input_text: str, generated_text: str, alpha: float = 0.5):
    proposed_text = TaggedText(generated_text)

    gec_ann = [ann for ann in proposed_text._annotations if ann.gec_error is not None]
    if gec_ann:
        gec_score = 1 - (np.mean(
            [penalty_mapping.get(ann.gec_error.error_type, 0.3) for ann in gec_ann]
        ) * 0.1 * len(gec_ann))
    else:
        gec_score = 1

    similarity_ratio = SequenceMatcher(None, input_text, proposed_text.get_corrected_text()).ratio()
    # print("GEC score: ", gec_score)
    # print("Similarity ratio: ", similarity_ratio)
    final_score = (alpha * gec_score + (1-alpha) * similarity_ratio)
    return final_score


if __name__ == "__main__":
    text = input("Provide text to evaluate: ")
    client = AsyncInferenceClient(base_url="https://jnovl897uj1glqgn.us-east-1.aws.endpoints.huggingface.cloud")
    # client = AsyncOpenAI(api_key=os.environ['OPENAI_API_KEY'])
    asyncio.run(_eval_model(client, text + "\n", 300, model_name="ft:gpt-4o-mini-2024-07-18:personal::A7fOmagX"))
