import os
from pathlib import Path

import torch
import pypdfium2 as pdfium
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration


ROOT = Path(__file__).resolve().parent.parent.parent

MODEL = "lightonai/LightOnOCR-3-0.8B"
PDF_PATH = str(ROOT / "data" / "raw" / "Core Rules.pdf")
OUTPUT_DIR = str(ROOT / "data" / "ocr")


def process_pdf(
    pdf_path: str,
    output_dir: str,
    max_pages: int | None = None,
):
    os.makedirs(output_dir, exist_ok=True)

    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        MODEL,
        torch_dtype=torch.bfloat16,
    ).to("cuda")

    processor = AutoProcessor.from_pretrained(MODEL)

    pdf = pdfium.PdfDocument(pdf_path)
    total_pages = len(pdf)

    pages_to_process = (
        total_pages
        if max_pages is None
        else min(max_pages, total_pages)
    )

    print(f"PDF: {pdf_path}")
    print(f"Total pages: {total_pages}")
    print(f"Pages to process: {pages_to_process}")

    for page_number in range(pages_to_process):
        page_index = page_number + 1

        output_path = os.path.join(
            output_dir,
            f"page_{page_index:03d}.md",
        )

        if os.path.exists(output_path):
            print(f"[{page_index}/{pages_to_process}] Skipping...")
            continue

        print(f"[{page_index}/{pages_to_process}] Processing...")

        page = pdf[page_number]
        image = page.render(scale=200 / 72).to_pil()

        conversation = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image,
                    }
                ],
            }
        ]

        inputs = processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )

        inputs = {
            key: value.to(
                device="cuda",
                dtype=(
                    torch.bfloat16
                    if value.is_floating_point()
                    else value.dtype
                ),
            )
            for key, value in inputs.items()
        }

        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=4096,
            )

        generated_ids = output_ids[
            0,
            inputs["input_ids"].shape[1]:,
        ]

        text = processor.decode(
            generated_ids,
            skip_special_tokens=True,
        )

        with open(output_path, "w", encoding="utf-8") as file:
            file.write(
                f"<!-- Source: {os.path.basename(pdf_path)} -->\n"
                f"<!-- Page: {page_index} -->\n\n"
                f"{text.strip()}\n"
            )

        page.close()
        del image
        del inputs
        del output_ids
    pdf.close()

    print(f"Done. {pages_to_process} pages processed.")


if __name__ == "__main__":
    process_pdf(PDF_PATH, OUTPUT_DIR)
