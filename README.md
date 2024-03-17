# lmmft

# Get repo & LLaVA (= submodule)
```
git clone https://github.com/muelea/lmmft.git
git submodule update --init --recursive
```

# To get latest changes to submodule
```
git pull --recurse-submodules
```


# Install Virtual Environment
```
conda create -n lmmft python=3.10 -y
conda activate lmmft
cd third-party/llava
pip install -e .
pip install -e ".[train]"
pip install flash-attn --no-build-isolation
```

# Download LLaVA Model Weights 
We use LLaVA v1.6. You can download the model files from [here](
https://huggingface.co/liuhaotian/llava-v1.6-mistral-7b?clone=true)


You need a folder named $ESSENTIALS to store the LLaVa weights.
```
sudo apt-get install git-lfs
mkdir $ESSENTIALS/llava && cd $ESSENTIALS/llava
git clone https://huggingface.co/liuhaotian/llava-v1.6-mistral-7b
```

Symlink the essentials folder to the repo folder
```
cd $REPO_DIR
ln -s $ESSENTIALS
```

# Demo code
LLaVA inferece 
```
python llava_inference.py --model_path essentials/llava/llava-v1.6-mistral-7b \
--prompt What are the things I should be cautious about when I visit here? \
--image_url https://www.weltderphysik.de/fileadmin/_processed_/1/e/csm_16920180628_Fuego_Thinkstock_6e2093c444.webp
```

Chat about an image with LLaVA 
```
python -m llava.serve.cli \
    --model-path essentials/llava/llava-v1.6-mistral-7b \
    --image-file "https://llava-vl.github.io/static/images/view.jpg" \
    --load-4bit
```

# Training of original Lava Data
Folder structure, download links and promts/labels are described (here)[https://github.com/haotian-liu/LLaVA?tab=readme-ov-file#visual-instruction-tuning]

