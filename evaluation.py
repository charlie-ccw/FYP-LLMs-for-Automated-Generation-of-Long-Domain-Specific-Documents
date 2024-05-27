import json
from util.evaluation_matrix import calculate_BLEU, calculate_METEOR, calculate_ROUGE_L, calculate_BLEURT, \
    calculate_GPT_SELF_EVALUATION

template_file = '../project_template/template_version_1.json'
generated_file = '../generated_file/version_2/01. VCS PD v4.2 - Timor-Leste ICS Project_REV_clean.json'
origin_file = '../file/Energy_demand_extract/structure_1/01. VCS PD v4.2 - Timor-Leste ICS Project_REV_clean.json'

with open(template_file, 'r', encoding='utf-8') as f:
    template_requirements = json.load(f)

with open(generated_file, 'r') as f:
    generated_sections = json.load(f)

with open(origin_file, 'r') as f:
    origin_sections = json.load(f)

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

for section_id, section_data in generated_sections.items():
    print(f'--------------- {section_id} ---------------')
    if section_data["section_name"].lower() == origin_sections[section_id]["section_name"].lower():
        generated_data = section_data["generation"]
        origin_data = origin_sections[section_id]["section_info"]
        template_requirement = template_requirements[section_id.split('.')[0]]['sections'][section_id]['description']

        bleu_score = calculate_BLEU(generation_text=generated_data, target_text=origin_data)
        rouge_l_score = calculate_ROUGE_L(generation_text=generated_data, target_text=origin_data)
        meteor_score = calculate_METEOR(generation_text=generated_data, target_text=origin_data)
        bleurt_score = calculate_BLEURT(generation_text=generated_data, target_text=origin_data)
        gpt_self_evaluation_score = calculate_GPT_SELF_EVALUATION(template_requirement=template_requirement,
                                                                  generation_text=generated_data,
                                                                  target_text=origin_data)

        bleu_result[section_id] = bleu_score
        rouge_l_result[section_id] = rouge_l_score
        meteor_result[section_id] = meteor_score
        bleurt_result[section_id] = bleurt_score
        gpt_self_evaluation_result[section_id] = gpt_self_evaluation_score

        bleu_total += bleu_score
        rouge_l_total += rouge_l_score
        meteor_total += meteor_score
        bleurt_total += bleurt_score
        gpt_self_evaluation_total += gpt_self_evaluation_score

    else:
        print(f"The ROUGE Scores of {section_id} is: None\n\n")

bleu_result['final'] = bleu_total / len(list(bleu_result.keys()))
rouge_l_result['final'] = rouge_l_total / len(list(rouge_l_result.keys()))
meteor_result['final'] = meteor_total / len(list(meteor_result.keys()))
bleurt_result['final'] = bleurt_total / len(list(bleurt_result.keys()))
gpt_self_evaluation_result['final'] = gpt_self_evaluation_total / len(list(gpt_self_evaluation_result.keys()))

evaluation_matrix['BLEU'] = bleu_result
evaluation_matrix['ROUGE_L'] = rouge_l_result
evaluation_matrix['METEOR'] = meteor_result
evaluation_matrix['BLEURT'] = bleurt_result
evaluation_matrix['GPT_SELF_EVALUATION'] = gpt_self_evaluation_result


with open("../evaluation_result/version_2/01. VCS PD v4.2 - Timor-Leste ICS Project_REV_clean.json", 'w') as f:
    json.dump(evaluation_matrix, f, indent=4)

