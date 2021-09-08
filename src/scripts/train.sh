
spring.submit run  -n 8 -p VI_UCV_1080TI --gres=gpu:8 --gres=gpu:8 --gpu --ntasks-per-node=8 --cpus-per-task=1 \
 --job-name=rec_pose \
"python train_multigpu.py --env 512_m0.5_8gpu --mask_ratio 0.5"
