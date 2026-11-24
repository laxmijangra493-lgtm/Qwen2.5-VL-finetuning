import gc
import os
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import get_cosine_schedule_with_warmup

from model import (
    MAX_PIXELS,
    TRANSCRIPTION_PROMPT,
    build_collate_fn,
    load_model_and_processor,
)

DATASET_ID = "radhakrishnadeshpande/gnhk-pagelevel"
OUTPUT_DIR = "./qwen2_5_vl_3b_gnhk_lora"
MAX_TRAIN_SAMPLES = 32   
BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 8
LR = 2e-4
EPOCHS = 1
MAX_STEPS = None        
SAVE_EVERY_STEPS = 32


class NotesDataset(Dataset):
    def __init__(self, hf_dataset, prompt_text: str, max_pixels: int):
        self.ds = hf_dataset
        self.prompt_text = prompt_text
        self.max_pixels = max_pixels

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        row = self.ds[idx]
        img = row["image"].convert("RGB")
        target_text = "" if row["text"] is None else str(row["text"])

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img, "max_pixels": self.max_pixels},
                    {"type": "text", "text": self.prompt_text},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": target_text}],
            },
        ]
        return {"messages": messages}


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model, processor = load_model_and_processor(for_training=True)

    raw_ds = load_dataset(DATASET_ID, split="train")
    if MAX_TRAIN_SAMPLES is not None:
        raw_ds = raw_ds.select(range(min(MAX_TRAIN_SAMPLES, len(raw_ds))))

    train_dataset = NotesDataset(raw_ds, TRANSCRIPTION_PROMPT, MAX_PIXELS)
    collate_fn = build_collate_fn(processor)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )

    steps_per_epoch = max(1, len(train_loader) // GRAD_ACCUM_STEPS)
    total_steps = MAX_STEPS if MAX_STEPS is not None else steps_per_epoch * EPOCHS

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR,
        weight_decay=0.01,
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, total_steps // 10),
        num_training_steps=total_steps,
    )

    model.train()
    optimizer.zero_grad()

    global_step = 0
    micro_step = 0
    running_loss = 0.0

    for epoch in range(EPOCHS):
        for batch in train_loader:
            if global_step >= total_steps:
                break

            batch = {
                k: v.to(model.device) if hasattr(v, "to") else v
                for k, v in batch.items()
            }

            outputs = model(**batch)
            loss = outputs.loss / GRAD_ACCUM_STEPS
            loss.backward()

            running_loss += outputs.loss.item()
            micro_step += 1

            if micro_step % GRAD_ACCUM_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                avg_loss = running_loss / GRAD_ACCUM_STEPS
                print(
                    f"Epoch {epoch + 1}/{EPOCHS} | Step {global_step}/{total_steps} | "
                    f"Loss: {avg_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.2e}"
                )
                running_loss = 0.0

                if SAVE_EVERY_STEPS and global_step % SAVE_EVERY_STEPS == 0:
                    ckpt_dir = os.path.join(OUTPUT_DIR, f"checkpoint-{global_step}")
                    model.save_pretrained(ckpt_dir)
                    processor.save_pretrained(ckpt_dir)
                    print(f"Saved checkpoint to {ckpt_dir}")
                if global_step % 8 == 0:
                    gc.collect()
                    torch.cuda.empty_cache()
            del outputs, loss, batch

        if global_step >= total_steps:
            break

    model.save_pretrained(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)
    print(f"Saved final adapter and processor to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()