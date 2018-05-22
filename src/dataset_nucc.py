from glob import glob
from tqdm import tqdm
import re


files = sorted(glob("./dataset/tmp/nucc/*.txt"))


def clean(file):
    with open(file, "r") as f:
        data = f.readlines()
        while 1:
            if data[0][0] in ["F", "M", "X"] and data[0][4] == "：":
                break
            data = data[1:]
        for i in range(len(data)):
            if data[i][0] in ["＠", "@"]:
                data = data[:i]
                break
        data = "".join(data).replace("\r", "").replace("\n", "")
        data = re.sub(r"[F|M][0-9]{3}：+", "\n", data)
        data = re.sub(r"Ｘ：", "\n", data)
        data = re.sub(r"X：", "\n", data)
        data = re.sub(r"％ｃｏｍ.*", "", data)
        data = re.sub(r"ＭＳ：", "\n", data)
        data = re.sub(r"＜.*＞", "", data)
        data = re.sub(r"１：＊＊＊。\n", "", data)
        data = re.sub(r"（.*）", "", data)
        data = re.sub(r"【.*】", "", data)
        data = re.sub(r"：", "", data)
        data = re.sub(r"\u3000", "", data)
        data = re.sub(r"[F|M][0-9]{3}", "P", data)
        data = re.sub(r"＊", "", data)
        data = re.sub(r"\t", "", data)
        return "\n".join([line for line in data.split("\n") if line not in ["", "。"]])


data = [clean(file) for file in tqdm(files, "cleaning data")]

for e, d in enumerate(data):
    with open("./dataset/conversation/nucc_{:03d}.txt".format(e), "w") as f:
        d = d.split("\n")
        d = [""] + d
        d_in = d[:-1]
        d_out = d[1:]
        new_d = []
        for i, o, in zip(d_in, d_out):
            new_d.append("I:" + i)
            new_d.append("O:" + o)
        new_d = "\n".join(new_d)
        f.write(new_d)
