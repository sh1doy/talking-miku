import json
from glob import glob


def get_conv(file):
    with open(file, "r", encoding='utf-8') as f:
        j = json.load(f)
    convs = j["turns"]
    res = []
    for conv in convs:
        res.append(conv["utterance"])
    return res


def main():
    files = glob("./dataset/tmp/DBDC2_dev/**/*.json", recursive=True)
    print("converting {} file(s)...".format(len(files)))
    convs = [get_conv(file) for file in files]
    for e, d in enumerate(convs):
        with open("./dataset/conversation/dbdc2_{:03d}.txt".format(e), "w") as f:
            d = [""] + d
            d_in = d[:-1]
            d_out = d[1:]
            new_d = []
            for i, o, in zip(d_in, d_out):
                new_d.append(i)
                new_d.append(o)
            new_d = "\n".join(new_d)
            f.write(new_d)


if __name__ == '__main__':
    main()
