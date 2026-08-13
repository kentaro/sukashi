# llm-watermark-lab

LLMテキスト電子透かし（KGW法 / Gumbel-Max法）のスクラッチ実装と検証実験。

## コマンド

- 依存同期: `uv sync`
- ユニットテスト（モデル不要・合成分布）: `uv run pytest`
- フル実験（Qwen2.5-0.5B-Instruct を MPS で実行）: `uv run python -m llm_watermark_lab.experiment`
  - 結果は `results/results.json` に出力

## 構成

- `src/llm_watermark_lab/common.py` — 鍵と直前トークンからの疑似乱数導出（生成側と検出側で共有）
- `src/llm_watermark_lab/watermark.py` — KGW / Gumbel-Max の生成と検出（z検定）
- `src/llm_watermark_lab/attacks.py` — トークン置換攻撃・パラフレーズ攻撃
- `src/llm_watermark_lab/experiment.py` — エンドツーエンド実験ランナー

## 規約

- Python は uv 管理（brew の python を使わない）
- コード・テスト内の文字列は英語のみ
- コミットに生成物マーカーやセッションURLを入れない
