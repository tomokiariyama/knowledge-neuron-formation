#!/bin/bash

# download GenericsKB dataset
mkdir -p "data"

FILE_ID=1QLD5RU2vpKWXqg2wofgIJRS_KtdI8ACf  # GenericsKBのダウンロードリンク内のID
curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=${FILE_ID}" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
#curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=${FILE_ID}" -o data/GenericsKB-Waterloo-With-Context.jsonl.zip
wget -nc "https://drive.google.com/uc?export=download&confirm=${CODE}&id=${FILE_ID}" -O data/GenericsKB-Waterloo-With-Context.jsonl.zip

FILE="data/cskb-waterloo-06-21-with-bert-scores.jsonl"
if [ ! -e $FILE ]; then
  unzip data/GenericsKB-Waterloo-With-Context.jsonl.zip
  mv -f cskb-waterloo-06-21-with-bert-scores.jsonl data/
fi
