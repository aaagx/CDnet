conda create -n CDnet  python=3.9 -y
conda activate CDnet
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia -y
conda install -c conda-forge pytorch-metric-learning -y
pip install https://s3-us-west-2.amazonaws.com/ray-wheels/master/43267ae9afae1522c4cd3c6b418bf9c1c419728c/ray-1.13.0-cp39-cp39-win_amd64.whl

pip install albumentations 
pip install tensorboard 
pip install pycocotools 
pip install notebook 
pip install matplotlib 
pip install pandas 
pip install timm
pip install -U openmim
mim install mmcv