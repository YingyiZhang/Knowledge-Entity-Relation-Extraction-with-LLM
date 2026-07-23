from openai import OpenAI
import re
import os
import json


client = OpenAI(
    api_key='',
    base_url=""
)


MODEL_NAME = "gpt-5-mini"


def parse_document(file_path):
    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}")
        return

    results = []
    with open(file_path, 'r', encoding='utf-8') as file:
            content = file.readlines()
            for line in content:
                item = json.loads(line)
                if 'new_golden_label' not in item:
                    continue
                id = item['id']
                label = item['new_golden_label']
                Task_instruction = item['Task_instruction']
                fw_shot = item['fw_shot']
                Query = item['Query']
                results.append({
                        'id': id,
                        'label': label,
                        'task_instruction': Task_instruction,
                        'fw_shot': fw_shot,
                        'query': Query
                })
    return results


for fold_num in range(0,5):
    file_path = '../datasets/Abstract/closed-source LLM/fold_'+str(fold_num)+'/val/cot_sample.json'
    output_file_path = '../output/output_qwen_turbo/fold_'+str(fold_num)+'/val/cot_sample_output.json'

    parsed_data = parse_document(file_path)

    with open(output_file_path, 'w', encoding='utf-8') as f:
        if parsed_data is None:
            print("No data was parsed. Please check the file path and format.")
        else:
            for item in parsed_data:
                id = item['id']
                label = item['label'],
                task_instruction =  item['task_instruction']
                fw_shot =  item['fw_shot']
                query =  item['query']


                try:
                        response = client.chat.completions.create(
                            model=MODEL_NAME,

                            messages=[
                                {"role": "system", "content": task_instruction},
                                {"role": "user", "content": "FEW_SHOT DEMONSTRACTIONS: "+fw_shot+" QUERY: "+query}
                            ],

                            temperature=0.7,
                            max_tokens=2000,
                        )


                        if response.choices and len(response.choices) > 0:
                            generated_text = response.choices[0].message.content.strip()
                            print("Generated:", generated_text)

                            result={
                                "id": id,
                                "context": query,
                                "golden_label": label,
                                "generated_reasoning": generated_text
                            }
                            f.write(json.dumps(result, ensure_ascii=False))
                            f.write("\n")
                        else:
                            print("The API returned an empty response")
                            f.write(json.dumps({"id": id, "error": "Empty response"},ensure_ascii=False))
                            f.write("\n")
                except Exception as e:
                        print(f"Request failed: {e}")
                        f.write(json.dumps({
                            "id": id,
                            "error": str(e)
                        },ensure_ascii=False))
                        f.write("\n")


    print(f"Completed. Results saved to: {output_file_path}")