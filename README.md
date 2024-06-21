# toyosr
Mock-up of simple optical scoresheet reader / optical character recognizer (ocr)



## Python3 の環境について

`opencv`関係のパッケージがいる.
`pdf2image`,
`pyzbar`
がいる.

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
```
pip install pdf2image
pip install pyzbar
pip install opencv-python
```

仮想環境から抜けるには,
```
deactivate
```


## Moodle 関係のメモ

個人にファイルを配布するには, おおよそ以下の手順(らしい):

まず,
* 課題を作成する.
* (課題の提出タイプの設定のチャックボックスを外し, 学生からの提出を受け付けなくする)
* フィードバックタイプでフィードバックファイルとオフライン評定ワークシートをチェックする.
* 以上で保存する.

そのあと,
* 課題のところに行き, すべての提出を表示する.
* 評定操作の, 評定ワークシートをダウンロードする.

ローカルでの操作は以下の通り:
* ダウンロードした評定ワークシート(csv)を見る.
* `参加者ID, フルネーム,` の列が, `123456,ああああ`, なら,
`ああああ_123456_assignsubmission_file_`
という名前 (つまり, `"%s_%s_assignsubmission_file_" % (フルネーム, 参加者ID)`)
の
ディレクトリをつくって, そこにファイルを入れればOK.
* 参加者IDというのは, ログインに使うやつでもなんでもなくて,
課題ごとに変わるなにかということのようなので注意.
* ファイルを保存してzipで固めてアップすればよい.


