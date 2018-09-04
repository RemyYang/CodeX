
ORIGIN_CHECKPOINT_PATH="./mobilenetv2_on_cifar10_check_point"
CHECKPOINT_PATH="./renamed_mobilenetv2_on_cifar10_check_point"
DATASET_NAME="cifar10"
DATASET_SPLIT_NAME="test"
DATASET_DIR="/home/deepl/PHICOMM/dataset/cifar10_tf"
MODEL_NAME="mobilenet_v2"
EVAL_IMAGE_SIZE=224

python rename.py \
        --original_ckpt_dir=$ORIGIN_CHECKPOINT_PATH\
        --new_ckpt_dir=$CHECKPOINT_PATH

python extract_feature.py\
        --checkpoint_path=$CHECKPOINT_PATH\
        --dataset_name=$DATASET_NAME\
        --dataset_split_name=$DATASET_SPLIT_NAME\
        --dataset_dir=$DATASET_DIR\
        --model_name=$MODEL_NAME\
        --eval_image_size=$EVAL_IMAGE_SIZE\
        --preprocessing_name="inception_v2"

python prediction.py\
        --checkpoint_path=$CHECKPOINT_PATH\
        --dataset_name=$DATASET_NAME\
        --dataset_split_name=$DATASET_SPLIT_NAME\
        --dataset_dir=$DATASET_DIR\
        --model_name=$MODEL_NAME\
        --eval_image_size=$EVAL_IMAGE_SIZE

python ensemble_num.py\
        --dataset_name=$DATASET_NAME\
        --dataset_split_name=$DATASET_SPLIT_NAME\
        --model_name=$MODEL_NAME\
        --eval_image_size=$EVAL_IMAGE_SIZE\
        --dataset_dir=$DATASET_DIR\
        --checkpoint_path=$CHECKPOINT_PATH


