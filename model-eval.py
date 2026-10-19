#!/usr/bin/env python3
# this code is using vllm to generate

import os

import pandas as pd
from tqdm import tqdm

# vllm
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams

tqdm.pandas()

structured_decoding_params = StructuredOutputsParams(regex=r"The final answer is: \d*\.\d*%")

sampling_params_with_constrained_decoding = SamplingParams(
    n=1,
    temperature=0,
    seed=1042,
    max_tokens=1000,
    # to try remove repetition
    repetition_penalty=1.2,
    structured_outputs=structured_decoding_params,
)

########## global variables ########################
# hugging face model
# change to openai/gpt-oss-20b for 20b model eval
model_id = "openai/gpt-oss-120b"


llm = LLM(model=model_id, trust_remote_code=True, enforce_eager=True, max_model_len=8192)


def construct_modified_query_base_llm(query):
    prompt_template = "Only use results from the genomic data commons in your response and provide frequencies as a percentage. Only report the final response."
    modified_query = query + prompt_template
    return modified_query


def batch_test(query):
    modified_query = construct_modified_query_base_llm(query)
    try:
        model_base_output = (
            llm.generate(modified_query, sampling_params_with_constrained_decoding)[0]
            .outputs[0]
            .text
        )
    except Exception as e:
        print("unable to get model base output for q {} {}".format(query, e))
        model_base_output = "no output obtained"

    return pd.Series(model_base_output)


def main():

    # change to csv with questions
    questions = pd.read_csv("csvs/questions.csv")
    print("getting baseline model responses on questions")
    questions["gpt_oss_120b_base_output"] = questions["questions"].progress_apply(
        lambda x: batch_test(x)
    )
    print("done generation questions , dumping to CSV")
    questions.to_csv("csvs/gpt-oss-120b.results.csv")


if __name__ == "__main__":
    main()
