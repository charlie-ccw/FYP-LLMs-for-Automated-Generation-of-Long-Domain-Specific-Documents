"""
After generating the files for the corresponding version, they need to be evaluated.
This Python file integrates the data and corresponding evaluation methods required for the evaluation,
such as BLEU, ROUGE, etc.
Note:
    You need to manually adjust the version you want to evaluate in main() program.
"""
import json
from util.evaluation_matrix import calculate_BLEU, calculate_METEOR, calculate_ROUGE_L, calculate_BLEURT, \
    calculate_GPT_SELF_EVALUATION


def evaluate_generated_file(file_name_without_extension: str, generation_version: int):
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
    rouge_l_result = {}
    meteor_result = {}
    bleurt_result = {}
    gpt_self_evaluation_result = {}
    bleu_total = 0
    rouge_l_total = 0
    meteor_total = 0
    bleurt_total = 0
    gpt_self_evaluation_total = 0

    # Start iteration and get all evaluation score to store
    for section_id, section_data in generated_sections.items():
        # Check the section name
        if (section_id in target_sections.keys()) and (section_data["section_name"].lower() == target_sections[section_id]["section_name"].lower()):
            # Get the generation and target text of current section
            generated_data = section_data["generation"]
            target_data = target_sections[section_id]["section_info"]
            # Get the template requirement of current section
            template_requirement = template_requirements[section_id.split('.')[0]]['sections'][section_id][
                'description']

            # Call functions to calculate evaluation scores
            bleu_score = calculate_BLEU(generation_text=generated_data, target_text=target_data)
            rouge_l_score = calculate_ROUGE_L(generation_text=generated_data, target_text=target_data)
            meteor_score = calculate_METEOR(generation_text=generated_data, target_text=target_data)
            bleurt_score = calculate_BLEURT(generation_text=generated_data, target_text=target_data)
            gpt_self_evaluation_score = calculate_GPT_SELF_EVALUATION(template_requirement=template_requirement,
                                                                      generation_text=generated_data,
                                                                      target_text=target_data)
            # Store these evaluation scores
            bleu_result[section_id] = bleu_score
            rouge_l_result[section_id] = rouge_l_score
            meteor_result[section_id] = meteor_score
            bleurt_result[section_id] = bleurt_score
            gpt_self_evaluation_result[section_id] = gpt_self_evaluation_score

            # calculate the sum of each score for calculating final score of whole file
            bleu_total += bleu_score
            rouge_l_total += rouge_l_score
            meteor_total += meteor_score
            bleurt_total += bleurt_score
            gpt_self_evaluation_total += gpt_self_evaluation_score['score']

        else:
            # Skip the section if section structure doesn't fit
            print(f"The ROUGE Scores of {section_id} is: None")

    # calculate the final score for the whole file for each evaluation matrix
    bleu_result['final'] = bleu_total / len(list(bleu_result.keys()))
    rouge_l_result['final'] = rouge_l_total / len(list(rouge_l_result.keys()))
    meteor_result['final'] = meteor_total / len(list(meteor_result.keys()))
    bleurt_result['final'] = bleurt_total / len(list(bleurt_result.keys()))
    gpt_self_evaluation_result['final'] = gpt_self_evaluation_total / len(list(gpt_self_evaluation_result.keys()))

    # Combine all evaluation matrixs together
    evaluation_matrix['BLEU'] = bleu_result
    evaluation_matrix['ROUGE_L'] = rouge_l_result
    evaluation_matrix['METEOR'] = meteor_result
    evaluation_matrix['BLEURT'] = bleurt_result
    evaluation_matrix['GPT_SELF_EVALUATION'] = gpt_self_evaluation_result

    # Save scores
    with open(f"evaluation_result/version_{generation_version}/{file_name_without_extension}.json", 'w',
              encoding='utf-8') as f:
        json.dump(evaluation_matrix, f, indent=4)


def main():
    # Set the generated version you want to evvaluate
    generation_version = 1

    # Load the train and test files with summary
    train_test_file_path = "project_template/template_version_1_summary.json"
    with open(train_test_file_path, 'r', encoding='utf-8') as f:
        train_test_datasets = json.load(f)

    # Get the test files with summary
    test_datasets = train_test_datasets['test']

    # Start iteration and evaluate each generation of specific version
    for idx, test_dataset in enumerate(test_datasets):
        file_name_without_extension = test_dataset['file_name']
        evaluate_generated_file(file_name_without_extension=file_name_without_extension,
                                generation_version=generation_version)


if __name__ == "__main__":
    main()
