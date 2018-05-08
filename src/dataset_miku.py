import requests
from bs4 import BeautifulSoup
from time import sleep

seq = []
print("Fetching", end="")
for i in range(1, 1000):
    sleep(1.0)
    url = "https://icws.indigo-bell.com/search?q=前川みく&st=n&page=" + str(i)
    responce = requests.get(url)
    print(".", end="", flush=True)
    soup = BeautifulSoup(responce.text, "lxml")
    lines = soup.find("tbody", class_="result").find_all("tr")
    if lines == []:
        break
    for line in lines:
        contents = line.find_all("td")[-1].contents
        seq += contents
print("\nGot {} lines.".format(len(seq)))
seq = [line.replace("○○", "P").replace(" ", "") for line in seq]
with open("./dataset/charactor/miku.txt", "w") as f:
    f.write("\n".join(seq))
