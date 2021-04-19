train_ssd=/home/akarsh/temp/pytorch-ssd/train_ssd.py
#pretrained=models/mobilenet-v1-ssd-mp-0_675.pth
pretrained=models/mb1-ssd-Epoch-0-Loss-3.5260154936048718.pth
datasets="data/number_plate_labels/voc_trial /home/akarsh/temp/pytorch-ssd/data/number_plate_labels/Pascal_voc"
alias training="python $train_ssd \
    --pretrained-ssd $pretrained \
    --data $datasets \
    --net mb1-ssd \
    --num-workers 4 \
    --batch-size 10 \
    --dataset-type voc"


test_example_script=run_ssd_example.py
# base_model=mobilenet-v1-ssd-mp-0_675.pth
model_path=models/mb1-ssd-Epoch-29-Loss-1.8118051158057318.pth
# model_path=models/$base_model
labels_path=models/labels.txt
image_path=data/number_plate_labels/voc_trial/JPEGImages/
alias testing="python $test_example_script \
	mb1-ssd \
	$model_path \
	$labels_path \
	$image_path \
    cutout
"
onnx_export_script=onnx_export.py
model_to_export=models/mb1-ssd-Epoch-29-Loss-1.8118051158057318.pth
labels_file=models/labels.txt
output_name="exported_model.onnx"
alias export_to_onnx="python $onnx_export_script \
	--net mb1-ssd \
	--input $model_path \
	--output $output_name \
    --labels $labels_file
"
