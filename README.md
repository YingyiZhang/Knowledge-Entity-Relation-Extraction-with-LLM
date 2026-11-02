# Knowledge-Entity-Relation-Extraction-with-LLM
This repository contains code for relation extraction experiments using large language models and pretrained language models.

## Model Checkpoints

Download the required model checkpoints from Hugging Face before running an experiment.

| Model | Checkpoint |
|---|---|
| T5 (UnifiedQA) | [allenai/unifiedqa-t5-3b](https://huggingface.co/allenai/unifiedqa-t5-3b) |
| Llama 2 | [meta-llama/Llama-2-7b-hf](https://huggingface.co/meta-llama/Llama-2-7b-hf) |
| Llama 2 Chat | [meta-llama/Llama-2-7b-chat-hf](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf) |
| ChatGLM3 | [zai-org/chatglm3-6b](https://huggingface.co/zai-org/chatglm3-6b) |
| Baichuan2 | [baichuan-inc/Baichuan2-7B-Chat](https://huggingface.co/baichuan-inc/Baichuan2-7B-Chat) |
| Qwen2.5 | [Qwen/Qwen2.5-7B](https://huggingface.co/Qwen/Qwen2.5-7B) |
| DeepSeek-R1-Distill-Llama | [deepseek-ai/DeepSeek-R1-Distill-Llama-8B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-8B) |

Some checkpoints may require accepting the model license or signing in to Hugging Face before downloading.

## Running an Experiment

Experiments are launched through the corresponding shell scripts. For example, run the Baichuan experiment with:

```bash
bash baichuan.sh
```

Before running a script, update its model, dataset, and output paths for your local environment.

## Notes

- Make sure the required model checkpoint is available locally or can be downloaded from Hugging Face.
- Use a Python environment containing the dependencies required by the selected training script.
- GPU memory requirements vary by model size and training configuration.

# Consolidated Results

For the Precision, Recall, and F1 scores of each relation type within each model category, please refer to the file output/detail_results.md.
