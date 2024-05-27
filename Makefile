.PHONY: dev_notebook

dev_notebook:
	jupyter lab

data_download:
	./scripts/data/download.sh


train_baseline_small:
	python ./scripts/model/yolov8/train.py \
          --data ./data/03_model_input/yolov8/small/data.yaml \
          --config ./scripts/model/yolov8/configs/baseline.yaml \
          --experiment-name baseline_small_dataset \
          --loglevel "info"

train_baseline_full:
	python ./scripts/model/yolov8/train.py \
          --data ./data/03_model_input/yolov8/full/data.yaml \
          --config ./scripts/model/yolov8/configs/baseline.yaml \
          --experiment-name baseline_full_dataset \
          --loglevel "info"

eval_val:
	python ./scripts/model/yolov8/eval.py \
	  --weights-filepath ./data/04_models/yolov8/baseline_small_dataset/weights/best.pt \
          --split "val" \
          --output-dir ./data/05_model_output/yolov8/baseline_small_dataset/ \
          --loglevel "info"

eval_test:
	python ./scripts/model/yolov8/eval.py \
	  --weights-filepath ./data/04_models/yolov8/baseline_small_dataset/weights/best.pt \
          --split "test" \
          --output-dir ./data/05_model_output/yolov8/baseline_small_dataset/ \
          --loglevel "info"
