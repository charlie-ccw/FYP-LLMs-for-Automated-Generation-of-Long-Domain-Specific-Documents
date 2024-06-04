"""
After generating the files for the corresponding version, they need to be evaluated.
This Python file integrates the data and corresponding evaluation methods required for the evaluation,
such as BLEU, ROUGE, etc.
Note:
    You need to manually adjust the version you want to evaluate in main() program.
"""
import asyncio
import json
import os.path

from util.evaluation_matrix import calculate_BLEU, calculate_METEOR, calculate_METEOR_of_summary, calculate_ROUGE, \
    calculate_Embedding_similarity_of_consine, calculate_GPT_SELF_EVALUATION


async def evaluate_generated_file(file_name_without_extension: str, generation_version: str):
    print(f">>>>>>>>>>>>>>> {file_name_without_extension} <<<<<<<<<<<<<<< Evaluation")
    # Set the File Path for generation, target and template file
    template_file_path = 'project_template/template_version_1.json'
    generated_file_path = f'generated_file/version_{generation_version}/{file_name_without_extension}.json'
    target_file_path = f'file/Energy_demand_extract/structure_1/{file_name_without_extension}.json'

    # Load the Template requirement
    with open(template_file_path, 'r', encoding='utf-8') as f:
        template_requirements = json.load(f)
    # Load the generation file
    with open(generated_file_path, 'r', encoding='utf-8') as f:
        generated_sections = json.load(f)
    # Load the Target file
    with open(target_file_path, 'r', encoding='utf-8') as f:
        target_sections = json.load(f)

    # Initialise the variables for storing evaluation scores
    evaluation_matrix = {}

    bleu_result = {}
    meteor_result = {}
    meteor_of_summary_result = {}
    rouge_1_result = {}
    rouge_l_result = {}
    embedding_similarity_of_cosine_result = {}
    gpt_self_evaluation_result = {}

    bleu_total = 0
    meteor_total = 0
    meteor_of_summary_total = 0
    rouge_1_total = 0
    rouge_l_total = 0
    embedding_similarity_of_cosine_total = 0
    gpt_self_evaluation_total = 0

    # Start iteration and get all evaluation score to store
    for section_id, section_data in generated_sections.items():
        # Check the section name
        if (section_id in target_sections.keys()) and (
                section_data["section_name"].lower() == target_sections[section_id]["section_name"].lower()):
            # Get the generation and target text of current section
            generated_data = section_data["generation"]
            target_data = target_sections[section_id]["section_info"]
            # Get the template requirement of current section
            template_requirement = template_requirements[section_id.split('.')[0]]['sections'][section_id][
                'description']

            # Call functions to calculate evaluation scores
            while 1:
                try:
                    bleu_score = calculate_BLEU(generation_text=generated_data, target_text=target_data)
                    meteor_score = calculate_METEOR(generation_text=generated_data, target_text=target_data)
                    meteor_of_summary_score = await calculate_METEOR_of_summary(generation_text=generated_data,
                                                                                target_text=target_data)
                    rouge_1_score = calculate_ROUGE(generation_text=generated_data, target_text=target_data)['rouge-1']['f']
                    rouge_l_score = calculate_ROUGE(generation_text=generated_data, target_text=target_data)['rouge-l']['f']
                    embedding_similarity_of_cosine_score = calculate_Embedding_similarity_of_consine(
                        generation_text=generated_data,
                        target_text=target_data)
                    gpt_self_evaluation_score = await calculate_GPT_SELF_EVALUATION(template_requirement=template_requirement,
                                                                                    generation_text=generated_data,
                                                                                    target_text=target_data)
                    # Store these evaluation scores
                    bleu_result[section_id] = bleu_score
                    meteor_result[section_id] = meteor_score
                    meteor_of_summary_result[section_id] = meteor_of_summary_score
                    rouge_1_result[section_id] = rouge_1_score
                    rouge_l_result[section_id] = rouge_l_score
                    embedding_similarity_of_cosine_result[section_id] = embedding_similarity_of_cosine_score
                    gpt_self_evaluation_result[section_id] = gpt_self_evaluation_score

                    # calculate the sum of each score for calculating final score of whole file
                    bleu_total += bleu_score
                    meteor_total += meteor_score
                    meteor_of_summary_total += meteor_of_summary_score
                    rouge_1_total += rouge_1_score
                    rouge_l_total += rouge_l_score
                    embedding_similarity_of_cosine_total += embedding_similarity_of_cosine_score
                    gpt_self_evaluation_total += gpt_self_evaluation_score['score']
                    break
                except Exception as e:
                    print(f'ERROR----------{file_name_without_extension}----------{section_id}----------{e}')
                    break

        else:
            # Skip the section if section structure doesn't fit
            print(f"The Evaluation of {file_name_without_extension} --- {section_id} is: None")

    # calculate the final score for the whole file for each evaluation matrix
    bleu_result['final'] = bleu_total / len(list(bleu_result.keys()))
    meteor_result['final'] = meteor_total / len(list(meteor_result.keys()))
    meteor_of_summary_result['final'] = meteor_of_summary_total / len(list(meteor_of_summary_result.keys()))
    rouge_1_result['final'] = rouge_1_total / len(list(rouge_1_result.keys()))
    rouge_l_result['final'] = rouge_l_total / len(list(rouge_l_result.keys()))
    embedding_similarity_of_cosine_result['final'] = embedding_similarity_of_cosine_total / len(
        list(embedding_similarity_of_cosine_result.keys()))
    gpt_self_evaluation_result['final'] = gpt_self_evaluation_total / len(list(gpt_self_evaluation_result.keys()))

    # Combine all evaluation matrixs together
    evaluation_matrix['BLEU'] = bleu_result
    evaluation_matrix['METEOR'] = meteor_result
    evaluation_matrix['METEOR_OF_SUMMARY'] = meteor_of_summary_result
    evaluation_matrix['ROUGE_1'] = rouge_1_result
    evaluation_matrix['ROUGE_L'] = rouge_l_result
    evaluation_matrix['EMBEDDING_SIMILARITY_OF_COSINE'] = embedding_similarity_of_cosine_result
    evaluation_matrix['GPT_SELF_EVALUATION'] = gpt_self_evaluation_result

    if not os.path.isdir(f"evaluation_result/version_{generation_version}"):
        os.mkdir(f"evaluation_result/version_{generation_version}")

    # Save scores
    with open(f"evaluation_result/version_{generation_version}/{file_name_without_extension}.json", 'w',
              encoding='utf-8') as f:
        json.dump(evaluation_matrix, f, indent=4)


async def main():
    # Load the train and test files with summary
    train_test_file_path = "project_template/template_version_1_summary.json"
    with open(train_test_file_path, 'r', encoding='utf-8') as f:
        train_test_datasets = json.load(f)

    # Get the test files with summary
    test_datasets = train_test_datasets['test']

    # Set the generated version you want to evvaluate
    generation_versions = ['1', '2', '4', '5', '6', '3_1', '3_2', '3_4', '3_5', '3_6']

    for generation_version in generation_versions:
        print(
            f"============================================================ {generation_version} ============================================================")
        tasks = []
        already_done_files = os.listdir(f"evaluation_result/version_{generation_version}")
        # Start iteration and evaluate each generation of specific version
        for idx, test_dataset in enumerate(test_datasets):
            file_name_without_extension = test_dataset['file_name']
            if file_name_without_extension + '.json' not in already_done_files:
                tasks.append(evaluate_generated_file(file_name_without_extension=file_name_without_extension,
                                                     generation_version=generation_version))

            if len(tasks) == 25:
                await asyncio.gather(*tasks)
                tasks = []

        # Process any remaining files
        if tasks:
            await asyncio.gather(*tasks)


if __name__ == "__main__":
    asyncio.run(main())
