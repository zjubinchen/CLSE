"""
Convert VizWiz annotation JSONL to LLaVA question JSONL format.

Input format (llava_val.jsonl):
  {"image": "VizWiz_val_00000000.jpg", "question": "...", "answers": [...], ...}

Output format required by model_vqa_loader.py:
  {"question_id": 0, "image": "VizWiz_val_00000000.jpg", "text": "..."}
"""

import argparse
import json


def convert(input_file, output_file):
    with open(input_file, "r") as fin, open(output_file, "w") as fout:
        for idx, line in enumerate(fin):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            new_obj = {
                "question_id": idx,
                "image": obj["image"],
                "text": obj["question"],
            }
            fout.write(json.dumps(new_obj) + "\n")
    print(f"Done. Wrote converted file to: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", type=str,
                        default="./playground/data/eval/vizwiz/llava_val.jsonl")
    parser.add_argument("--output-file", type=str,
                        default="./playground/data/eval/vizwiz/llava_val_converted.jsonl")
    args = parser.parse_args()
    convert(args.input_file, args.output_file)
