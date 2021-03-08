# article_18837_CxNN
日本発の衛星データプラットフォーム Tellus のオウンドメディア「宙畑」の記事、https://sorabatake.jp/18837 で利用しているコードです。

複素ニューラルネットワークを用いたSARデータの物体検出を行います。
なお、SARデータの取得にはTellusの開発環境を申し込む必要があります。

## ソースコード(./src/配下を参照)
- 00.py
  - CEOSダウンロード
- 01.py
  - CEOSをSLCに変換
- 02.py
  - SLCをSAR画像に変換
- 03.py
  - 学習データの生成
- 04.py
  - 複素ニューラルネットワークによる学習及び評価
- slcinfo.py
  - SLCの構造体

## ライセンス、利用規約
ソースコードのライセンスは CC0-1.0（Creative Commons Zero v1.0 Universal）ライセンスです。  
今回コード内で PALSAR-2 データを用いております。利用ポリシーは以下をご参考下さい。
https://www.tellusxdp.com/market/tool_detail/de3c41ac-a8ca-4170-9028-c9e1a39841e1/e364c31c-bfad-49d0-bd6d-f2bc11d67386
※サイトの閲覧にはTellusへのログインが必要です。

## 貢献方法
プルリクエストや Issue はいつでも歓迎します。



by charmegiddo
