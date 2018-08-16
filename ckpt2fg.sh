SLIM_DIR=/home/deepl/PHICOMM/FoodAI/FoodAi/tensorflow/tensorflow_models/models/research/PHICOMM/slim
MODEL_NAME=mobilenet_v2
MODEL_DIR=/home/deepl/PHICOMM/RemyWorkSpace/ensemble/CodeX/renamed_check_point
FROZEN_DIR=/home/deepl/PHICOMM/RemyWorkSpace/ensemble/CodeX/frozen_pb
#DATASET_NAME=cifar10
IMAGE_SIZE=224


MODEL_OUTPUT_NODE_NAME=MobilenetV2/Predictions/Reshape_1

if [ -d $FROZEN_DIR ]; then
    rm -rf $FROZEN_DIR
fi
mkdir -p $FROZEN_DIR


for p in 0
do
for i in 0 1 2 3 4 5 6 7 8 9
do
for j in `seq 0 50`
do
echo "$j"
MODEL_DATA_CHECKPOINT_NAME=model.ckpt-${p}-${i}-${j}

if [ ! -f "$MODEL_DIR/$MODEL_DATA_CHECKPOINT_NAME.index" ]; then
    echo "文件不存在"
else
    echo "文件存在"
    python /home/deepl/PHICOMM/FoodAI/FoodAi/tensorflow/tensorflow_master/tensorflow/tensorflow/python/tools/freeze_graph.py \
            --input_graph=$MODEL_DIR/inf_graph.pb \
            --input_checkpoint=$MODEL_DIR/$MODEL_DATA_CHECKPOINT_NAME \
            --input_binary=true --output_graph=$FROZEN_DIR/frozen_${p}-${i}-${j}.pb \
            --output_node_names=$MODEL_OUTPUT_NODE_NAME
fi
done
done
done


