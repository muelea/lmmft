import argparse 
import torch 
import os

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

    torch.cuda.empty_cache()

    args = parse_args()
    model_path = args.model_path 
    prompt = args.prompt 
    image_file = args.image_url 

    # Assign paths to variables
    DEEPSPEED_SCRIPT = "deepspeed third-party/LLaVA/llava/train/train_mem.py"
    DEEPSPEED_JSON = "third-party/LLaVA/scripts/zero3.json"
    MODEL_NAME = model_path #"liuhaotian/llava-v1.5-7b"
    VISION_TOWER = "openai/clip-vit-large-patch14-336"

    DATA_PATH = "datasets/original/lm_chat_demo/train.json" #llava_v1_5_mix665k.json"  # Replace with your JSON data path
    IMAGE_FOLDER = "datasets/original/lm_chat_demo/images"  # Replace with your image folder path
    OUTPUT_DIR = "outdebug"  # Replace with your desired output directory path

    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=model_path,
        model_base=None,
        model_name=get_model_name_from_path(model_path),
        offload_folder="/content/llava_model"
    )

    # Command to run the script
    finetune_script = f'''
    {DEEPSPEED_SCRIPT} \
        --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
        --deepspeed {DEEPSPEED_JSON} \
        --model_name_or_path {MODEL_NAME} \
        --version v1 \
        --data_path {DATA_PATH} \
        --image_folder {IMAGE_FOLDER} \
        --vision_tower {VISION_TOWER} \
        --mm_projector_type mlp2x_gelu \
        --mm_vision_select_layer -2 \
        --mm_use_im_start_end False \
        --mm_use_im_patch_token False \
        --image_aspect_ratio pad \
        --group_by_modality_length True \
        --fp16 True \
        --bf16 False \
        --tf32 False \
        --output_dir {OUTPUT_DIR} \
        --num_train_epochs 5 \
        --per_device_train_batch_size 16 \
        --per_device_eval_batch_size 4 \
        --gradient_accumulation_steps 1 \
        --evaluation_strategy "no" \
        --save_strategy "steps" \
        --save_steps 50000 \
        --save_total_limit 1 \
        --learning_rate 2e-4 \
        --weight_decay 0. \
        --warmup_ratio 0.03 \
        --lr_scheduler_type "cosine" \
        --logging_steps 1 \
        --model_max_length 2048 \
        --gradient_checkpointing True \
        --dataloader_num_workers 4 \
        --lazy_preprocess True \
        --report_to wandb
    '''

    os.system(finetune_script)