import torch
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from qwen_vl_utils import process_vision_info

BASE_MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
MIN_PIXELS = 256 * 28 * 28
MAX_PIXELS = 512 * 28 * 28

TRANSCRIPTION_PROMPT = (
    "Transcribe all handwritten text from this notebook page accurately. "
    "Preserve headings, bullet points, and form fields cleanly. "
    "Do not guess illegible words."
)


def get_bnb_config() -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )


def load_model_and_processor(for_training: bool = True, lora_r: int = 16, lora_alpha: int = 32):
    processor = AutoProcessor.from_pretrained(BASE_MODEL_ID,
        min_pixels=MIN_PIXELS,
        max_pixels=MAX_PIXELS,)
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    processor.tokenizer.padding_side = "right"

    model = AutoModelForImageTextToText.from_pretrained(
        BASE_MODEL_ID,
        quantization_config=get_bnb_config(),
        dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )

    if for_training:
        model = prepare_model_for_kbit_training(model)
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj",],)
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    return model, processor


def build_collate_fn(processor):
    assistant_marker = "<|im_start|>assistant\n"
    marker_ids = processor.tokenizer.encode(assistant_marker, add_special_tokens=False)
    pad_id = processor.tokenizer.pad_token_id

    def collate_fn(batch):
        texts = []
        image_inputs = []
        video_inputs = []

        for item in batch:
            msgs = item["messages"]
            full_text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
            texts.append(full_text)

            img_in, vid_in = process_vision_info(msgs)
            if img_in:
                image_inputs.append(img_in)
            if vid_in:
                video_inputs.append(vid_in)

        inputs = processor(
            text=texts,
            images=image_inputs if image_inputs else None,
            videos=video_inputs if video_inputs else None,
            padding=True,
            return_tensors="pt",
        )

        labels = inputs["input_ids"].clone()
        if pad_id is not None:
            labels[labels == pad_id] = -100

        for b_idx in range(labels.size(0)):
            seq = labels[b_idx].tolist()
            start_idx = None
            for pos in range(len(seq) - len(marker_ids) + 1):
                if seq[pos : pos + len(marker_ids)] == marker_ids:
                    start_idx = pos + len(marker_ids)
                    break
            if start_idx is not None:
                labels[b_idx, :start_idx] = -100

        inputs["labels"] = labels
        return inputs

    return collate_fn