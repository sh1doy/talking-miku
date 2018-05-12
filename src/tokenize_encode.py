from glob import glob
from tqdm import tqdm
import subprocess


files = glob("dataset/charactor/*.txt") + glob("dataset/conversation/*.txt")
for file in tqdm(files, desc="encoding"):
    string = ["spm_encode", "--model", "models/m.model", "<", file, ">", file.replace(".txt", ".parsed")]
    subprocess.run(" ".join(string), shell=True)
