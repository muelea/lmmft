import argparse
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model


def parse_args():
    parser = argparse.ArgumentParser(description="Run LLAVA")
    parser.add_argument(
        "--model_path", 
        type=str, 
        default="essentials/llava/llava-v1.6-mistral-7b",
        help="Path to the model to use"
    )
    parser.add_argument(
        "--prompt", 
        type=str, 
        default = "What are the things I should be cautious about when I visit here?",
        help="Prompt to use"
    )
    parser.add_argument(
        "--image_url", 
        type=str, 
        default="https://www.weltderphysik.de/fileadmin/_processed_/1/e/csm_16920180628_Fuego_Thinkstock_6e2093c444.webp", 
        help="Image file to use"
    )
    args = parser.parse_args()
    return args 


if __name__ == "__main__":

    args = parse_args()
    model_path = args.model_path 
    prompt = args.prompt 
    image_file = args.image_url 

    args = type('Args', (), {
        "model_path": model_path,
        "model_base": None,
        "model_name": get_model_name_from_path(model_path),
        "query": prompt,
        "conv_mode": None,
        "image_file": image_file,
        "sep": ",",
        "temperature": 0,
        "top_p": None,
        "num_beams": 1,
        "max_new_tokens": 512
    })()

    eval_model(args)