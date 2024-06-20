# toyosr
Mock-up of simple optical scoresheet reader / optical character recognizer (ocr)



## Python

`pdf2image` がいる.

### pip について.
PEP668によって,
`pip install --user` が使えなくなったので, 
パッケージを入れるにはvenvを使ってからやるのがよいよう.

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
