from PIL import Image
from vllm import LLM, EngineArgs, SamplingParams
from vllm.sampling_params import StructuredOutputsParams

MODEL_NAME = "Qwen/Qwen3.5-4B"
IMAGE_PLACEHOLDER = "<|vision_start|><|image_pad|><|vision_end|>"

CAPTION_REGEX = r"This is [^.\n]+ that is [^.\n]+ and could be [^.\n]+\.\n"

CAPTIONING_QUESTION = (
    'Complete this sentence about the image: "This is ___ that is ___ and '
    'could be ___." The first blank names the main object. The second '
    "blank describes its relation to the scene. The third blank describes "
    "what it might be doing or used for. Respond with only the completed "
    "sentence, ending in a period."
)


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


def structured_caption_image(llm: LLM, image: Image.Image) -> str:
    prompt = (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        f"<|im_start|>user\n{IMAGE_PLACEHOLDER}{CAPTIONING_QUESTION} /no_think<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    inputs = {"prompt": prompt, "multi_modal_data": {"image": image}}

    structured_outputs_params = StructuredOutputsParams(regex=CAPTION_REGEX)
    sampling_params = SamplingParams(
        structured_outputs=structured_outputs_params,
        stop=["\n"],
        max_tokens=100,
    )

    outputs = llm.generate(inputs, sampling_params=sampling_params)
    return outputs[0].outputs[0].text.strip()


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    args = parser.parse_args()

    image = Image.open(args.image).convert("RGB")
    llm = load_model()
    caption = structured_caption_image(llm, image)
    print(f"Structured caption: {caption}")


if __name__ == "__main__":
    main()
