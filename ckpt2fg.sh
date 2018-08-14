SLIM_DIR=/=/home/xxxx/PHICOMM/Project/FoodAi/tensorflow/tensorflow_models/models/research/slim
MODEL_NAME=mobilenet_v2
MODEL_DIR=/home/xxxx/PHICOMM/RemyWorkSpace/rename_file/renamed_check_point
FROZEN_DIR=/home/xxxx/PHICOMM/RemyWorkSpace/rename_file/frozen_pb
DATASET_NAME=imagenet
IMAGE_SIZE=224


MODEL_OUTPUT_NODE_NAME=MobilenetV2/Predictions/Reshape_1

for p in 0 1 2 3
do
for i in 0 1 2 3 4 5 6 7 8 9
do 
for j in 0 1 2 3 4 
do

MODEL_DATA_CHECKPOINT_NAME=model.ckpt-${p}${i}${j}

python /home/xxxx/PHICOMM/Project/FoodAi/tensorflow/tensorflow_master/tensorflow/tensorflow/python/tools/freeze_graph.py \
            --input_graph=$MODEL_DIR/inf_graph.pb \
            --input_checkpoint=$MODEL_DIR/$MODEL_DATA_CHECKPOINT_NAME \
            --input_binary=true --output_graph=$FROZEN_DIR/frozen_${p}${i}${j}.pb \
            --output_node_names=$MODEL_OUTPUT_NODE_NAME 
done
done
done


