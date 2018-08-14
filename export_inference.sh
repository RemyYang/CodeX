SLIM_DIR=/home/xxxx/PHICOMM/Project/FoodAi/tensorflow/tensorflow_models/models/research/slim
MODEL_NAME=mobilenet_v2
MODEL_DIR=/home/xxxx/PHICOMM/RemyWorkSpace/rename_file/renamed_check_point
FROZEN_DIR=/home/xxxx/PHICOMM/RemyWorkSpace/frozen_pb
DATASET_NAME=imagenet
IMAGE_SIZE=224


python ${SLIM_DIR}/export_inference_graph.py \
            --model_name=$MODEL_NAME \
            --output_file=$MODEL_DIR/inf_graph.pb \
            --dataset_name=$DATASET_NAME \
            --image_size=$IMAGE_SIZE \
            --batch_size=1
