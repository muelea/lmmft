# lmmft

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