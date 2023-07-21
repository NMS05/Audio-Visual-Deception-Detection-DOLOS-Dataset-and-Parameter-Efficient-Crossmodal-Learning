for LR in 1e-4 1e-3
do
  for SEED in 2222 321
  do
    CUDA_VISIBLE_DEVICES=7 python train_test.py \
    --batch_size=16 \
    --model_to_train "fusion" \
    --fusion_type "cross2" \
    --adapter \
    --num_encoders=4 \
    --adapter_type="efficient_conv" \
    --num_epochs=30 \
    --lr=$LR \
    --seed=$SEED \
    --data_root '/home2/xiaobao/datasets/deception/WILTY/data/' \
    --audio_path '/home2/xiaobao/datasets/deception/WILTY/data/audio_files/' \
    --visual_path '/home2/xiaobao/datasets/deception/WILTY/data/face_frames/'
  done
done