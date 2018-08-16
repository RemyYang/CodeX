SLIM_DIR=/home/deepl/PHICOMM/FoodAI/FoodAi/tensorflow/tensorflow_models/models/research/PHICOMM/slim
MODEL_NAME=mobilenet_v2
MODEL_DIR=/home/deepl/PHICOMM/RemyWorkSpace/ensemble/CodeX/renamed_check_point
DATASET_NAME=cifar10
#DATASET_NAME=imagenet
IMAGE_SIZE=224


python ${SLIM_DIR}/export_inference_graph.py \
            --model_name=$MODEL_NAME \
            --output_file=$MODEL_DIR/inf_graph.pb \
            --dataset_name=$DATASET_NAME \
            --image_size=$IMAGE_SIZE \
            --batch_size=1
