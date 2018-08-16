FROZEN_DIR=/home/deepl/PHICOMM/RemyWorkSpace/ensemble/CodeX/frozen_pb
BASE_DIR=/home/deepl/PHICOMM/RemyWorkSpace/ensemble/CodeX/base
CLASSIFIER_DIR=/home/deepl/PHICOMM/RemyWorkSpace/ensemble/CodeX/classifier


if [ -d $BASE_DIR ]; then
    rm -rf $BASE_DIR
fi
mkdir -p $BASE_DIR

if [ -d $CLASSIFIER_DIR ]; then
    rm -rf $CLASSIFIER_DIR
fi
mkdir -p $CLASSIFIER_DIR

python /home/deepl/PHICOMM/FoodAI/FoodAi/tensorflow/tensorflow_master/tensorflow/tensorflow/python/tools/optimize_for_inference.py \
--input=$FROZEN_DIR/frozen_0-0-0.pb \
--output=$BASE_DIR/base.pb \
--frozen_graph=True \
--input_names=input \
--output_names=MobilenetV2/Logits/AvgPool

for p in 0
do
for i in 0 1 2 3 4 5 6 7 8 9
do
for j in `seq 0 50`
do

if [ ! -f "$FROZEN_DIR/frozen_${p}-${i}-${j}.pb" ]; then
    echo "文件不存在"
else
    echo "文件存在"
    python /home/deepl/PHICOMM/FoodAI/FoodAi/tensorflow/tensorflow_master/tensorflow/tensorflow/python/tools/optimize_for_inference.py \
        --input=$FROZEN_DIR/frozen_${p}-${i}-${j}.pb \
        --output=$CLASSIFIER_DIR/classifier_${p}-${i}-${j}.pb \
        --frozen_graph=True \
        --input_names=MobilenetV2/Logits/AvgPool \
        --output_names=MobilenetV2/Predictions/Reshape_1
fi
done
done
done
