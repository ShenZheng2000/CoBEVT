 # NOTE: use python 3.8 instead of 3.7, and pip install of conda

# Clone repo
git clone https://github.com/DerrickXuNu/CoBEVT.git
cd CoBEVT/opv2v

# Setup conda environment
# conda create -y --name cobevt python=3.7
conda create -y --name cobevt python=3.8

conda activate cobevt
# conda install pytorch==1.11.0 torchvision==0.12.0 cudatoolkit=11.3 -c pytorch
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113

# Install dependencies
python opencood/utils/setup.py build_ext --inplace
python setup.py develop

# # # Other things to add (making it default)
echo "export LD_PRELOAD=$(realpath $CONDA_PREFIX/lib/libstdc++.so.6)" > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
conda deactivate && conda activate cobevt

pip install numpy==1.23.5
pip install timm


# # for unzipping, check here: https://opencood.readthedocs.io/en/latest/md_files/data_intro.html#opv2v
# for f in train_chunks-*.zip; do unzip -n "$f"; done
# cat train_chunks/train.zip.part* > train.zip
# unzip train.zip

# for f in validate_chunks-*.zip; do unzip -n "$f"; done
# cat validate_chunks/validate.zip.part* > validate.zip
# unzip validate.zip

# for f in test_chunks-*.zip; do unzip -n "$f"; done
# cat test_chunks/test.zip.part* > test.zip
# unzip test.zip