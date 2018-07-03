TMP_DIR = dataset/tmp
NUCC_URL = http://mmsrv.ninjal.ac.jp/nucc/nucc.zip
NUCC_ZIP = nucc.zip
DBDC2_URL = https://sites.google.com/site/dialoguebreakdowndetection2/downloads/DBDC2_dev.zip
DBDC2_ZIP = DBDC2_dev.zip
MODEL_DIR = models

VOCAB_SIZE = 10000

all: dataset-all tokenize

dataset-nucc:
	-mkdir $(TMP_DIR)
	wget -N -P $(TMP_DIR) $(NUCC_URL)
	unzip -u $(TMP_DIR)/$(NUCC_ZIP) -d $(TMP_DIR)
	python src/dataset_nucc.py

dataset-miku:
	python src/dataset_miku.py

dataset-dbdc2:
	-mkdir $(TMP_DIR)
	wget -N -P $(TMP_DIR) $(DBDC2_URL)
	unzip -u $(TMP_DIR)/$(DBDC2_ZIP) -d $(TMP_DIR)
	python src/dataset_dbdc2.py

dataset-all: dataset-dbdc2 dataset-miku
