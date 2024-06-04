import asyncio
import os
import numpy as np
import nltk
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_openai import OpenAIEmbeddings
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate import meteor_score
from nltk.tokenize import word_tokenize
from rouge import Rouge
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from prompt.prompt_of_gpt_self_evaluation import GPT_SELF_EVALUATION_SYSTEM, GPT_SELF_EVALUATION_PROMPT
from util.prompt_based_generation import prompt_based_generation, aprompt_based_generation

nltk.download("wordnet")
nltk.download('omw-1.4')
nltk.download('punkt')

tokenizer = AutoTokenizer.from_pretrained("Elron/bleurt-base-512")
model = AutoModelForSequenceClassification.from_pretrained("Elron/bleurt-base-512")


def cosine_similarity(embeddings1, embeddings2):
    dot_product = np.dot(embeddings1, embeddings2)
    norm_emb1 = np.linalg.norm(embeddings1)
    norm_emb2 = np.linalg.norm(embeddings2)
    similarity = dot_product / (norm_emb1 * norm_emb2)

    return similarity


def calculate_BLEU(generation_text: str, target_text: str) -> float:
    """
    BLEU can evaluate the lexical overlap between two texts,
    but its drawbacks are that it cannot consider the semantic level of the text
    and ignores the coherence and sparsity of the text.
    :param generation_text: The text you want to evaluate
    :param target_text: Your target example
    :return: BLEU score
    """
    reference_tokens = [word_tokenize(target_text)]
    candidate_tokens = word_tokenize(generation_text)

    smoothie = SmoothingFunction().method4

    bleu_result = sentence_bleu(reference_tokens, candidate_tokens, smoothing_function=smoothie)

    return float(bleu_result)


def calculate_METEOR(generation_text: str, target_text: str) -> float:
    """
    METEOR evaluates the similarity between two texts, considering synonyms, roots, and suffixes.
    However, it is sensitive to the order of generated sentences,
    which can lead to a lower score for sentences that have the same meaning but different word orders.
    :param generation_text: The text you want to evaluate
    :param target_text: Your target example
    :return: METEOR score
    """
    reference_tokens = [word_tokenize(target_text)]
    candidate_tokens = word_tokenize(generation_text)

    meteor_result = meteor_score.meteor_score(reference_tokens, candidate_tokens)

    return float(meteor_result)


async def calculate_METEOR_of_summary(generation_text: str, target_text: str) -> float:
    """
    METEOR evaluates the similarity between two texts, considering synonyms, roots, and suffixes.
    However, it is sensitive to the order of generated sentences,
    which can lead to a lower score for sentences that have the same meaning but different word orders.
    :param generation_text: The text you want to evaluate
    :param target_text: Your target example
    :return: METEOR score
    """
    tasks = [
        aprompt_based_generation(prompt=f"please summarize the following section context:\n{generation_text}", model='gpt-4o', temperature=0),
        aprompt_based_generation(prompt=f"please summarize the following section context:\n{target_text}", model='gpt-4o', temperature=0)
    ]
    response = await asyncio.gather(*tasks)

    reference_tokens = [word_tokenize(response[1].content)]
    candidate_tokens = word_tokenize(response[0].content)

    meteor_result = meteor_score.meteor_score(reference_tokens, candidate_tokens)

    return float(meteor_result)


def calculate_ROUGE(generation_text: str, target_text: str) -> dict:
    """
    Evaluating text similarity using the longest common subsequence is suitable for long text generation tasks.
    It considers both recall and precision.
    However, it can be affected by noise and cannot discern semantic diversity.
    :param generation_text: The text you want to evaluate
    :param target_text: Your target example
    :return: ROUGE-L score
    """
    rouge = Rouge(metrics=['rouge-l', 'rouge-1'])

    rouge_result = rouge.get_scores(generation_text, target_text, avg=True)

    return rouge_result


def calculate_Embedding_similarity_of_consine(generation_text: str, target_text: str) -> float:
    """
    Based on the transformer model, it can support contextual understanding and
    provide more accurate semantic-level judgments.
    However, it has a high computational cost and relies heavily on the effectiveness of the pre-trained model,
    making it unable to determine precision and recall.
    :param generation_text: The text you want to evaluate
    :param target_text: Your target example
    :return: BLEURT score
    """
    embeddings_model = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))
    embeddings = embeddings_model.embed_documents(
        [
            generation_text,
            target_text
        ]
    )
    embedding_similarity_of_cosine = cosine_similarity(embeddings[0], embeddings[1])

    return float(embedding_similarity_of_cosine)


async def calculate_GPT_SELF_EVALUATION(generation_text: str, target_text: str,
                                  template_requirement: str) -> dict:
    """
    By leveraging the model's discriminative capabilities, this tool evaluates the generated text,
    providing both the reasoning and corresponding scores.
    :param generation_text: The text you want to evaluate
    :param target_text: Your target example
    :param template_requirement: The template for your generated text
    :return: GPT-SELF-EVALUATION score
    """
    chat_template = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content=GPT_SELF_EVALUATION_SYSTEM),
            HumanMessagePromptTemplate.from_template(GPT_SELF_EVALUATION_PROMPT),
        ]
    )

    messages = chat_template.format_messages(template_requirement=template_requirement,
                                             generation_text=generation_text, target_text=target_text)

    while 1:
        try:
            response = await aprompt_based_generation(prompt=messages, model='gpt-4o', temperature=0.5, json_format=True)
            break
        except Exception as e:
            print(e)

    gpt_self_evaluation_result = {
        'reason': response['reason'],
        'score': float(response['score'])
    }

    return gpt_self_evaluation_result
