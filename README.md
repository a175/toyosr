# toyosr
Mock up of simple optical scoresheet reader / optical character recognizer (ocr)



## Python

PEP668によって,
`pip install --user` が使えなくなったので, 
パッケージを入れるにはvenvを使うのがよいよう.

最初にセットアップ
```
mkdir myvenv
cd myvenv/
python3 -m venv .
cd ../
```

それから,
```
source myenv/bin/activate
```
で仮想環境にはいるので,
`pip install` とかを自由にやる.
仮想環境から抜けるには,
```
deactivate
```
