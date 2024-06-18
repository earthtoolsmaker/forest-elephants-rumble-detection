.PHONY: dev_notebook

dev_notebook:
	jupyter lab

data_download:
	./scripts/data/download.sh


train_dumb_small:
	python ./scripts/model/yolov8/train.py \
          --data ./data/03_model_input/yolov8/small/data.yaml \
          --config ./scripts/model/yolov8/configs/dumb.yaml \
          --experiment-name dumb_small_dataset \
          --loglevel "info"

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

train_best_full:
	python ./scripts/model/yolov8/train.py \
          --data ./data/03_model_input/yolov8/full/data.yaml \
          --config ./scripts/model/yolov8/configs/best.yaml \
          --experiment-name best_full_dataset \
          --loglevel "info"

train_baseline_testing_v1:
	python ./scripts/model/yolov8/train.py \
          --data ./data/03_model_input/yolov8/testing_features_only_full/data.yaml \
          --config ./scripts/model/yolov8/configs/baseline.yaml \
          --experiment-name baseline_testing_v1_only_features_dataset \
          --loglevel "info"

train_best_testing_v1:
	python ./scripts/model/yolov8/train.py \
          --data ./data/03_model_input/yolov8/testing_features_only_full/data.yaml \
          --config ./scripts/model/yolov8/configs/best.yaml \
          --experiment-name baseline_testing_v1_only_features_dataset \
          --loglevel "info"

eval_val:
	python ./scripts/model/yolov8/eval.py \
	  --weights-filepath ./data/04_models/yolov8/baseline_small_dataset/weights/best.pt \
          --split "val" \
          --output-dir ./data/05_model_output/yolov8/baseline_small_dataset/ \
          --loglevel "info"

eval_dumb_val:
	python ./scripts/model/yolov8/eval.py \
	  --weights-filepath ./data/04_models/yolov8/dumb_small_dataset/weights/best.pt \
          --split "val" \
          --output-dir ./data/06_reporting/yolov8/dumb_small_dataset/ \
          --loglevel "info"

eval_dumb_test:
	python ./scripts/model/yolov8/eval.py \
	  --weights-filepath ./data/04_models/yolov8/dumb_small_dataset/weights/best.pt \
          --split "test" \
          --output-dir ./data/06_reporting/yolov8/dumb_small_dataset/ \
          --loglevel "info"

eval_test:
	python ./scripts/model/yolov8/eval.py \
	  --weights-filepath ./data/04_models/yolov8/baseline_small_dataset/weights/best.pt \
          --split "test" \
          --output-dir ./data/05_model_output/yolov8/baseline_small_dataset/ \
          --loglevel "info"


inference_best:
	python ./scripts/model/yolov8/inference.py \
          --input-dir-yolov8-dataset ./data/03_model_input/yolov8/full/ \
          --output-dir ./data/05_model_output/yolov8/best_full_dataset/ \
          --split "test" \
          --k 25 \
          --random-seed 0 \
          --loglevel "info"

build_features_testing_v1:
	python ./scripts/data/build_features.py \
          --input-rumbles-dir "./data/01_raw/cornell_data/Rumble/" \
          --output-dir "./data/02_features/rumbles/spectrograms_v1/" \
          --duration 60.0 \
          --freq-min 0.0 \
          --freq-max 250.0 \
          --random-seed 0 \
          --ratio-random-offsets 0.2 \
          --loglevel "info"

build_features_testing_v2:
	python ./scripts/data/build_features2.py \
          --input-rumbles-dir "./data/01_raw/cornell_data/Rumble/" \
          --output-dir "./data/02_features/rumbles/spectrograms_torchaudio_v0/" \
          --duration 164.0 \
          --freq-min 0.0 \
          --freq-max 250.0 \
          --random-seed 0 \
          --ratio-random-offsets 0.2 \
          --loglevel "info"
