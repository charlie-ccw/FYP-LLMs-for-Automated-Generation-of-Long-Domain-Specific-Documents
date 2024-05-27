import nltk
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate import meteor_score
from nltk.tokenize import word_tokenize
from rouge import Rouge
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from prompt.prompt_of_gpt_self_evaluation import GPT_SELF_EVALUATION_SYSTEM, GPT_SELF_EVALUATION_PROMPT
from util.prompt_based_generation import prompt_based_generation

nltk.download("wordnet")
nltk.download('omw-1.4')
nltk.download('punkt')

tokenizer = AutoTokenizer.from_pretrained("Elron/bleurt-base-512")
model = AutoModelForSequenceClassification.from_pretrained("Elron/bleurt-base-512")


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


def calculate_ROUGE_L(generation_text: str, target_text: str) -> float:
    """
    Evaluating text similarity using the longest common subsequence is suitable for long text generation tasks.
    It considers both recall and precision.
    However, it can be affected by noise and cannot discern semantic diversity.
    :param generation_text: The text you want to evaluate
    :param target_text: Your target example
    :return: ROUGE-L score
    """
    rouge = Rouge(metrics=['rouge-l'])

    rouge_l_result = rouge.get_scores(generation_text, target_text, avg=True)['rouge-l']['f']

    return float(rouge_l_result)


def calculate_BLEURT(generation_text: str, target_text: str) -> float:
    """
    Based on the transformer model, it can support contextual understanding and
    provide more accurate semantic-level judgments.
    However, it has a high computational cost and relies heavily on the effectiveness of the pre-trained model,
    making it unable to determine precision and recall.
    :param generation_text: The text you want to evaluate
    :param target_text: Your target example
    :return: BLEURT score
    """
    inputs = tokenizer(target_text, generation_text, return_tensors='pt', padding=True, truncation=True, max_length=512)

    with torch.no_grad():
        outputs = model(**inputs)

    bleurt_result = outputs.logits.squeeze().cpu().numpy()

    return float(bleurt_result)


def calculate_GPT_SELF_EVALUATION(generation_text: str, target_text: str,
                                  template_requirement: str, model_name: str = 'gpt-4o') -> dict:
    """
    By leveraging the model's discriminative capabilities, this tool evaluates the generated text,
    providing both the reasoning and corresponding scores.
    :param generation_text: The text you want to evaluate
    :param target_text: Your target example
    :param template_requirement: The template for your generated text
    :param model_name: The model you want to use for exaluation
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

    response = prompt_based_generation(prompt=messages, model=model_name, temperature=0.5, json_format=True)

    gpt_self_evaluation_result = {
        'reason': response['reason'],
        'score': float(response['score'])
    }

    return gpt_self_evaluation_result
