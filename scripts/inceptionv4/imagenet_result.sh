ORIGIN_CHECKPOINT_PREFIX="/home/deepl/Project/moble_ensemble_checkpoint"
ORIGIN_CHECKPOINT_PATH="$ORIGIN_CHECKPOINT_PREFIX/inceptionv4_on_imagenet"
CHECKPOINT_PATH="./renamed_inceptionv4_on_imagenet_checkpoint"
DATASET_NAME="imagenet"
DATASET_SPLIT_NAME="validation"
DATASET_DIR="/home/deepl/Project/dataset/imagenet/tfrecord"
MODEL_NAME="inception_v4"
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
        --preprocessing_name="inception_v2"\
        --input_layer="input"\
        --output_layer="InceptionV4/Logits/AvgPool_1a/AvgPool"

python prediction.py\
        --checkpoint_path=$CHECKPOINT_PATH\
        --dataset_name=$DATASET_NAME\
        --dataset_split_name=$DATASET_SPLIT_NAME\
        --dataset_dir=$DATASET_DIR\
        --model_name=$MODEL_NAME\
        --eval_image_size=$EVAL_IMAGE_SIZE\
        --input_layer="InceptionV4/Logits/AvgPool_1a/AvgPool"\
        --output_layer="InceptionV4/Logits/Predictions"

python ensemble_num.py\
        --dataset_name=$DATASET_NAME\
        --dataset_split_name=$DATASET_SPLIT_NAME\
        --model_name=$MODEL_NAME\
        --eval_image_size=$EVAL_IMAGE_SIZE\
        --dataset_dir=$DATASET_DIR\
        --checkpoint_path=$CHECKPOINT_PATH


