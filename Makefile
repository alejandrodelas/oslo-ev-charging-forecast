.PHONY: all data train replicate analysis test

all: data train replicate analysis

data:        ## raw NOBIL + traffic -> station-hour table
	python src/build_dataset.py
train:       ## models, metrics, figures
	python src/train.py
replicate:   ## original team model, ported to Python
	python src/replicate_team.py
analysis:    ## extra figures and comparisons
	python src/analysis.py
test:
	pytest -q tests
