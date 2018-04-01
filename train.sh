#!/usr/bin/env bash

clear

func_xe_rl()
{
    #-----------------------------------------
    XE_ENABLE=1
    if [ "$XE_ENABLE" -eq 1 ];then
        TIME_TAG=`date "+%Y%m%d-%H%M%S"` # Time stamp
        if [ -z "$CKPT_PATH" ]; then
            CKPT_PATH="save/"$TIME_TAG"."$MODEL_TYPE # Generate save path
            mkdir $CKPT_PATH
        fi
        echo "Current saving path is "$CKPT_PATH
        MODEL_ID=${CKPT_PATH#"save/"}
        echo "Current saving id is "$MODEL_ID
        if [ ! -f $CKPT_PATH"/infos-best.pkl" ]; then
            START_FROM=""
        else
            START_FROM="--start_from "$CKPT_PATH
        fi
        echo "Current checkpoint path: "$CKPT_PATH
        echo "Start from pretrained: "$START_FROM

        CUDA_VISIBLE_DEVICES=$GPU_ID python train.py --caption_model $MODEL_TYPE --batch_size 100 --beam_size 1 --learning_rate 5e-4 --learning_rate_decay_start 0 --scheduled_sampling_start 0 --checkpoint_path $CKPT_PATH $START_FROM --save_checkpoint_every 3000 --language_eval 1 --val_images_use 10000 --max_epoch 37  --rnn_size 512 --use_box 0 --use_bn 1
    fi
    #-----------------------------------------
    RL_ENABLE=1
    if [ "$RL_ENABLE" -eq 1 ];then
        echo "Current saving path is "$CKPT_PATH
        MODEL_ID=${CKPT_PATH#"save/"}
        echo "Current saving id is "$MODEL_ID
        if [ ! -f $CKPT_PATH"/infos-best.pkl" ]; then
            START_FROM=""
        else
            START_FROM="--start_from "$CKPT_PATH
        fi
        echo "Current checkpoint path: "$CKPT_PATH
        echo "Start from pretrained: "$START_FROM
        if [ ! -d save/rl ]; then
            mkdir save/rl
        fi
        if [ ! -d save/rl/$MODEL_ID ]; then
            cp -r $CKPT_PATH save/rl/
        fi
        CUDA_VISIBLE_DEVICES=$GPU_ID python train.py --caption_model $MODEL_TYPE --batch_size 100 --beam_size 1 --learning_rate 5e-5 --learning_rate_decay_start 0 --learning_rate_decay_every 55  --learning_rate_decay_rate 0.1  --scheduled_sampling_start 0 --checkpoint_path save/rl/$MODEL_ID --start_from $CKPT_PATH --save_checkpoint_every 3000 --language_eval 1 --val_images_use 10000 --self_critical_after 0 --rnn_size 512 --use_bn 1 --use_box 0
    fi
}

MODEL_TYPE=''
ROOT_DIR=$PWD
echo "run "$MODEL_TYPE
GPU_ID=$2 # Get gpu id
CKPT_PATH=$3 # Get save path
echo "GPU using "$GPU_ID

case "$1" in
     0) MODEL_TYPE='fc' && func_xe_rl;;
     1) MODEL_TYPE='att2in' && func_xe_rl;;
     2) MODEL_TYPE='att2in2' && func_xe_rl;;
     3) MODEL_TYPE='att2all2' && func_xe_rl;;
     4) MODEL_TYPE='adaatt' && func_xe_rl;;
     5) MODEL_TYPE='adaattmo' && func_xe_rl;;
     6) MODEL_TYPE='topdown' && func_xe_rl;;
     7) MODEL_TYPE='stackatt' && func_xe_rl;;
     8) MODEL_TYPE='denseatt' && func_xe_rl;;
     9) MODEL_TYPE='stackcap' && func_xe_rl;;
     *) echo "No input" ;;
esac
