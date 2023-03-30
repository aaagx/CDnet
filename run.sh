#！\bin\bash
cp /dev/null nohup.out
conda activate osr
python3 setup.py install --user    
nohup /home/ubuntu/.local/bin/osr_run --trial_config=/home/ubuntu/GFN-1.1.0/configs/custom/prw_resnet_train.yaml       


