TMP_DIR = dataset/tmp
NUCC_URL = http://mmsrv.ninjal.ac.jp/nucc/nucc.zip
NUCC_ZIP = nucc.zip

dataset-nucc:
	mkdir $(TMP_DIR)
	wget -N -P $(TMP_DIR) $(NUCC_URL)
	unzip -u $(TMP_DIR)/$(NUCC_ZIP) -d $(TMP_DIR)
	python src/dataset_nucc.py
	rm -r $(TMP_DIR)

dataset-miku:
	python src/dataset_miku.py 

dataset-all:
	dataset-nucc
	dataset-miku

