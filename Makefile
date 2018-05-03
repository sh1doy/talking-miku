TMP_DIR = dataset/tmp
NUCC_URL = http://mmsrv.ninjal.ac.jp/nucc/nucc.zip
NUCC_ZIP = nucc.zip

dataset-nucc:
	wget -N -P $(TMP_DIR) $(NUCC_URL)
	unzip -u $(TMP_DIR)/$(NUCC_ZIP) -d $(TMP_DIR)
	python src/dataset_nucc.py

dataset-miku:
	python src/dataset_miku.py 

dataset:
	dataset-nucc
	dataset-miku



