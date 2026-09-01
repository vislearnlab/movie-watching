import argparse

from PIL import Image
from vllm import LLM, EngineArgs, SamplingParams

MODEL_NAME = "Qwen/Qwen3.5-4B"
IMAGE_PLACEHOLDER = "<|vision_start|><|image_pad|><|vision_end|>"


def load_model(gpu_memory_utilization: float = 0.7) -> LLM:
    engine_args = EngineArgs(
        model=MODEL_NAME,
        max_model_len=4096,
        max_num_seqs=5,
        gpu_memory_utilization=gpu_memory_utilization,
        mm_processor_kwargs={
            "min_pixels": 28 * 28,
            "max_pixels": 1280 * 28 * 28,
            "fps": 1,
        },
        limit_mm_per_prompt={"image": 1},
    )
    return LLM.from_engine_args(engine_args)


def caption_image(llm: LLM, image: Image.Image, question: str = "Describe this image.") -> str:
    prompt = (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        f"<|im_start|>user\n{IMAGE_PLACEHOLDER}{question} /no_think<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    inputs = {"prompt": prompt, "multi_modal_data": {"image": image}}
    sampling_params = SamplingParams(temperature=0.2, max_tokens=150)
    outputs = llm.generate(inputs, sampling_params=sampling_params)
    text = outputs[0].outputs[0].text.strip()

    if "</think>" in text:
        text = text.split("</think>")[-1].strip()

    return text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--question", default="Describe this image.")
    args = parser.parse_args()

    image = Image.open(args.image).convert("RGB")
    llm = load_model()
    caption = caption_image(llm, image, args.question)
    print(f"Caption: {caption}")


if __name__ == "__main__":
    main()
