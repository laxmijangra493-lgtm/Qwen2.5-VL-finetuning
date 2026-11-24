import argparse
import torch
from datasets import load_dataset
from peft import PeftModel
from qwen_vl_utils import process_vision_info

from model import MAX_PIXELS, TRANSCRIPTION_PROMPT, load_model_and_processor

DATASET_ID = "radhakrishnadeshpande/gnhk-pagelevel"
DEFAULT_ADAPTER_DIR = "./qwen2_5_vl_3b_gnhk_lora"


def run_inference(adapter_dir: str, sample_idx: int = 0, max_new_tokens: int = 512):
    base_model, processor = load_model_and_processor(for_training=False)
    model = PeftModel.from_pretrained(base_model, adapter_dir)
    model.eval()

    test_row = load_dataset(DATASET_ID, split="test")[sample_idx]
    img = test_row["image"].convert("RGB")

    messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": img, "max_pixels": MAX_PIXELS},
                {"type": "text", "text": TRANSCRIPTION_PROMPT}]}]

    prompt_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[prompt_text],
        images=image_inputs if image_inputs else None,
        videos=video_inputs if video_inputs else None,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

    trimmed_ids = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
    ]
    prediction = processor.batch_decode(trimmed_ids, skip_special_tokens=True)[0]

    print("PREDICTION:\n", prediction)
    print("\nTRUTH:\n", str(test_row["text"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter-dir", default=DEFAULT_ADAPTER_DIR)
    parser.add_argument("--sample-idx", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    args = parser.parse_args()

    run_inference(
        adapter_dir=args.adapter_dir,
        sample_idx=args.sample_idx,
        max_new_tokens=args.max_new_tokens,
    )