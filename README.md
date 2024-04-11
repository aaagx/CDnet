enviroment:enviroment.sh
train:python train.py --cfg configs/prw.yaml --eval --ckpt logs/prw_coat/epoch_15.pth EVAL_USE_CBGM True
