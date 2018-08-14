python /home/xxxx/PHICOMM/Project/FoodAi/tensorflow/tensorflow_master/tensorflow/tensorflow/python/tools/optimize_for_inference.py \
--input=/home/xxxx/PHICOMM/RemyWorkSpace/rename_file/frozen_pb/frozen_000.pb \
--output=/home/xxxx/PHICOMM/RemyWorkSpace/rename_file/base/base.pb \
--frozen_graph=True \
--input_names=input \
--output_names=MobilenetV2/Logits/AvgPool

for p in 0 1 2 3
do
for i in 0 1 2 3 4 5 6 7 8 9
do 
for j in 0 1 2 3 4
do

python /home/xxxx/PHICOMM/Project/FoodAi/tensorflow/tensorflow_master/tensorflow/tensorflow/python/tools/optimize_for_inference.py \
--input=/home/xxxx/PHICOMM/RemyWorkSpace/rename_file/frozen_pb/frozen_${p}${i}${j}.pb \
--output=/home/xxxx/PHICOMM/RemyWorkSpace/rename_file/classifier/classifier_${p}${i}${j}.pb \
--frozen_graph=True \
--input_names=MobilenetV2/Logits/AvgPool \
--output_names=MobilenetV2/Predictions/Reshape_1

done
done
done
