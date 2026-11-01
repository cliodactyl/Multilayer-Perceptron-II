DATASET   ?= data/data.csv
TRAIN_SET ?= data/data_training.csv
VALID_SET ?= data/data_validation.csv
MODEL     ?= saved_model.npy
CONFIG    ?= configs/network.json
PYTHON    ?= python3

.PHONY: all install split train predict check compare clean fclean re help

all: split train predict

install:
	$(PYTHON) -m pip install -r requirements.txt

split:
	$(PYTHON) split.py $(DATASET) --validation-ratio 0.2 --seed 42

train: $(TRAIN_SET)
	$(PYTHON) train.py --config $(CONFIG)

predict: $(MODEL)
	$(PYTHON) predict.py --model $(MODEL) --dataset $(VALID_SET)

check:
	$(PYTHON) tools/gradient_check.py

compare:
	$(PYTHON) train.py --config $(CONFIG) --optimizer sgd  --history history_sgd.json  --no-plot --quiet
	$(PYTHON) train.py --config $(CONFIG) --optimizer adam --learning_rate 0.001 \
		--history history_adam.json --model model_adam.npy --no-plot --quiet
	$(PYTHON) tools/compare_models.py history_sgd.json history_adam.json --metric val_loss

$(TRAIN_SET): $(DATASET)
	$(MAKE) split

$(MODEL): $(TRAIN_SET)
	$(MAKE) train

clean:
	rm -rf __pycache__ mlp/__pycache__ tools/__pycache__
	rm -f history*.json *.png predictions.csv

fclean: clean
	rm -f $(MODEL) model_*.npy $(TRAIN_SET) $(VALID_SET)

re: fclean all

help:
	@echo "make split     découpe le dataset"
	@echo "make train     entraîne le réseau (configs/network.json)"
	@echo "make predict   évalue le modèle sauvegardé"
	@echo "make check     vérification numérique du gradient"
	@echo "make compare   entraîne en SGD et en Adam, puis superpose les courbes"
	@echo "make clean     supprime les fichiers générés"
	@echo "make re        repart de zéro"
