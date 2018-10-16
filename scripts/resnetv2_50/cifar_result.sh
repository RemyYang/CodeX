ORIGIN_CHECKPOINT_PREFIX="/home/deepl/Project/moble_ensemble_checkpoint"
ORIGIN_CHECKPOINT_PATH="$ORIGIN_CHECKPOINT_PREFIX/resnetv2_50_on_cifar10"
CHECKPOINT_PATH="./renamed_resnetv2_50_on_cifar10_check_point"
DATASET_NAME="cifar10"
DATASET_SPLIT_NAME="test"
DATASET_DIR="/home/deepl/Project/dataset/cifar10_tf"
MODEL_NAME="resnet_v2_50"
EVAL_IMAGE_SIZE=299

python rename.py \
        --original_ckpt_dir=$ORIGIN_CHECKPOINT_PATH\
        --new_ckpt_dir=$CHECKPOINT_PATH\
        --select_model_num=10
python extract_feature.py\
        --checkpoint_path=$CHECKPOINT_PATH\
        --dataset_name=$DATASET_NAME\
        --dataset_split_name=$DATASET_SPLIT_NAME\
        --dataset_dir=$DATASET_DIR\
        --model_name=$MODEL_NAME\
        --eval_image_size=$EVAL_IMAGE_SIZE\
        --preprocessing_name="inception"\
        --input_layer="input"\
        --output_layer="resnet_v2_50/pool5"

python prediction.py\
        --checkpoint_path=$CHECKPOINT_PATH\
        --dataset_name=$DATASET_NAME\
        --dataset_split_name=$DATASET_SPLIT_NAME\
        --dataset_dir=$DATASET_DIR\
        --model_name=$MODEL_NAME\
        --eval_image_size=$EVAL_IMAGE_SIZE\
        --input_layer="resnet_v2_50/pool5"\
        --output_layer="resnet_v2_50/predictions/Reshape_1"

python ensemble_num.py\
        --dataset_name=$DATASET_NAME\
        --dataset_split_name=$DATASET_SPLIT_NAME\
        --model_name=$MODEL_NAME\
        --eval_image_size=$EVAL_IMAGE_SIZE\
        --dataset_dir=$DATASET_DIR\
        --checkpoint_path=$CHECKPOINT_PATH



