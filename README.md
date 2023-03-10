# 知識ニューロンの形成過程
- このリポジトリは、[この論文](https://www.anlp.jp/proceedings/annual_meeting/2023/pdf_dir/B5-4.pdf) の実験結果を再現するためのものです。
- このリポジトリは、https://github.com/EleutherAI/knowledge-neurons のコードを改変して作成されました。

## 実験の再現方法
このコードは Docker 上で動かすことを想定しています。

- 初めに、このリポジトリのクローンとディレクトリの移動を行ってください。
```bash
$ git clone git@github.com:tomokiariyama/knowledge-neuron-formation.git
$ cd knowledge-neuron-formation
```
- 次に、以下のコマンドで各スクリプトの実行権限を付与してください：
```bash
$ chmod +x docker.sh scripts/setup.sh scripts/evaluate.sh scripts/make_graphs.sh experiment.sh
```
- 続けて、次のコマンドで Docker コンテナを起動します：
```bash
./docker.sh
```
- コンテナが起動したら、次のコマンドによって実験と図の作成を行います：
```bash
./experiment.sh
```
  - 注1）このスクリプトは、著者の環境(NVIDIA TITAN X ×1)でおよそ11日かかりました。（一つのチェックポイントあたり平均で約21時間）
  - 注2）なお、`experiment.sh`における`evaluate.sh`の実行部分を、チェックポイントごとに並列化することで実行時間を短縮することができます。
    - （参考）`evaluate.py`に渡す引数`--local_rank`によって、1チェックポイントに対する実験を行うGPUを指定することができます。
    
  - スクリプトが終了すると、`work/figure/generics_kb_best`配下に論文に掲載の図が出力されます：
      - 図3: "violinplot_suppress.png"
      - 図4: "violinplot_enhance.png"


## スクリプト
```yaml
scripts/setup.sh: GenericsKBデータセットをダウンロードします
scripts/evaluate.sh: 実験を行います
  - evaluate.py: 実験コード
scripts/make_graphs.sh: 論文に掲載の図を出力します
  - utils/violinplots.py: 図を作成するコード
```
