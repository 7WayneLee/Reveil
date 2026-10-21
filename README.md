# Reveil

用 LTX-2.5 的 IC-LoRA 還原影片裡的馬賽克。目前完成第一步：產生訓練資料，並在 Colab A100 上訓練 IC-LoRA。推論（實際還原影片）是下一步。

## 做法

- 底模是 LTX-2.5 22B（dev），用 Lightricks 官方 `ltx-trainer` 的 reference 條件化（IC-LoRA）訓練：打碼的裁切當作參考 token（乾淨、timestep 0、和目標同位置、不算 loss），無碼的裁切是要生成的目標。
- 每個樣本是一對 512×512、121 影格的影片。所有樣本用同一句 caption，推論時重用同一份 caption embedding，不需要文字編碼器。

## 訓練資料

只能用合法取得、所有演出者都確認成年的無碼影片。訓練是在無碼畫面上打假馬賽克，讓模型學著還原成原本的畫面，所以原片就是正確答案：有碼片（包括 AI 去碼的版本）不能用。影片要夠多、夠多樣（建議 20 部以上、不同演員和場景）；配對太少，模型只會背下那幾段。

每部影片需要一個區域標註檔 `regions/<影片檔名>.json`，標出馬賽克要打在哪裡。可以用下面的[自動標註](#自動標註)產生，或用任何偵測或標註工具輸出這個格式：

```json
{
  "version": 1,
  "frames": {
    "0": [{"box": [x0, y0, x1, y1]}],
    "1": [{"polygon": [[x, y], [x, y], [x, y]]}]
  },
  "cuts": [240]
}
```

- frame key 是從 0 起算的解碼影格編號，座標是原始像素。
- 有 `polygon` 時馬賽克照輪廓打；只有 `box` 時隨機用矩形或內切橢圓。
- `cuts` 可省略：每個場景切換後的第一格，track 和視窗不會跨過它。

### 自動標註

`reveil.annotate` 用 YOLO 偵測或分割模型，在無碼影片裡逐格找出要打假馬賽克的部位，寫成上面的標註檔。它找的是露出的部位，不是找馬賽克（還原有碼片時找馬賽克是推論階段的事）：

```bash
pip install -e ".[annotate]"  # onnxruntime；用 GPU 的話改裝 onnxruntime-gpu
python -m reveil.annotate --videos sources/ --model detector.onnx --out regions/ --preview-dir previews/
```

- 模型要先用 ultralytics 匯出成 ONNX：`yolo export model=<權重>.pt format=onnx dynamic=True`。Reveil 只讀 ONNX 檔，不 import ultralytics。
- 影片直接解碼成模型輸入的大小（例如 640），在這個大小上找區域，再換算回原始像素；4K 影片不會比 1080p 慢太多。
- 分割模型的遮罩變成輪廓多邊形，偵測模型的框變成方框，都會再外擴一點（`--grow`，畫面短邊的比例）。
- 每格的偵測結果會連成 track：短於 `--min-track-seconds` 的丟掉，`--max-gap-seconds` 以內的漏偵測用前後影格補上。偵測到的場景切換寫進 `cuts`。
- `--classes` 只保留指定類別（名稱或編號），`--conf` 是分數門檻，`--stride N` 每 N 格偵測一次（中間的影格會補上）。執行時會印出模型的所有類別。
- `--preview-dir` 每部影片輸出一張預覽圖，綠線是標註；產生配對前先看過。
- 已經有標註檔的影片會跳過，`--overwrite` 重做。

Colab notebook 預設用 [NudeNet](https://github.com/notAI-tech/NudeNet) 的 `640m` 偵測模型，只取 `FEMALE_GENITALIA_EXPOSED` 和 `MALE_GENITALIA_EXPOSED` 兩類：一般有碼片只遮生殖器，肛門不打碼，假馬賽克也要一樣。也可以改用 [Lada](https://github.com/ladaapp/lada) 的分割模型 `lada_nsfw_detection_model_v1.3`；它只有一類 `nsfw`，會連肛門一起標。每個偵測模型的標註和預覽各放一個資料夾（`regions/<偵測模型>/`、`previews/<偵測模型>/`），換模型就會重新標註。

產生配對：

```bash
pip install -e .
python -m reveil.make_pairs --videos sources/ --regions regions/ --out data/ --workers 8
```

每個區域 track 切成 121 影格的視窗（預設不重疊），每個視窗：

1. **取景**：裁切框平滑跟著區域、緩慢縮放，小區域會多帶一些周邊畫面；縮放後補邊成 512 正方形（`reveil/camera.py`，推論也會用同一套）。
2. **打碼**：在原始解析度上打。像素塊是畫面長邊的 0.7%–2%（至少 4 px）；網格固定在畫面座標或跟著區域移動；色塊取平均或中心像素；區域先外擴再對齊網格；5% 機率改用模糊。
3. **壓縮**：打碼後的區域用 x264 或 x265、CRF 18–32 重新編碼再解回來，模擬真實發行版本的壓縮。
4. **輸出**：無碼和打碼版用同一組裁切框，寫成 `clean/<key>.mp4`（目標）和 `mosaic/<key>.mp4`（參考），每格都保留，標成 24 fps。

輸出目錄：

```
data/clean/*.mp4       目標
data/mosaic/*.mp4      參考
data/meta/<影片>.json  每個視窗的參數（裁切框、馬賽克樣式、壓縮）
data/dataset.jsonl     給 ltx-trainer process_dataset.py 的清單
```

已完成的影片重跑時會跳過，所以中斷後可以接著跑。參數可以用 YAML 覆寫（`--config`）：

```yaml
pairs: {hop: 104, crf: [20, 30], max_windows_per_video: 20}
camera: {margin: 0.3}
mosaic: {block_fraction: [0.008, 0.015], blur_probability: 0.0}
```

`--random-regions` 用隨機漂移的方框代替標註，只用來測試流程。

## 在 Colab 訓練

在 Colab 選「檔案 → 開啟筆記本 → GitHub」，輸入 `7WayneLee/Reveil`，開啟 `notebooks/train_colab.ipynb`。需要：

- A100 執行階段。80GB 最好；40GB 也可以：標註、配對、前處理都一樣，訓練時底模自動改用 INT8 量化（`int8-quanto`，約 22 GB），每步比較慢。checkpoint 在兩種卡之間可以互相接著訓練。
- Colab 密鑰 `HF_TOKEN`（先在 Hugging Face 同意 LTX-2.5 的條款）
- 訓練用的無碼影片：預設讀 Drive 上的 `reveil/sources/FC2/`，可以是指到別人分享資料夾的 Drive 捷徑，子資料夾也會讀。設定裡的 `SOURCES` 可以改成別的資料夾，或列好幾個。有碼片（例如 `reveil/sources/` 裡的 AV）留給之後去碼用，不要列進 `SOURCES`。
- `reveil/regions/<偵測模型>/`（標註）會自動產生，也可以自己放

notebook 依序安裝、下載 LTX-2.5（約 72 GB）、自動標註、產生配對、前處理、訓練。標註前會先確認每部影片都讀得到，Drive 掛載出問題就馬上停下來。偵測模型第一次下載後存在 Drive 的 `reveil/models/`，標註預覽在 `reveil/previews/<偵測模型>/`。checkpoint 每 250 步存到 Drive，斷線後重跑整本會從最新的 checkpoint 續訓。第一次建議先開 `SMOKE_TEST`（隨機區域、不標註），確認整條流程能跑通。每部影片最多取 `MAX_WINDOWS_PER_VIDEO` 組配對（預設 100，隨機挑），讓資料平均來自不同影片；正式訓練的配對少於 `MIN_SAMPLES`（預設 100）時，notebook 會停下來不訓練。

訓練設定在 `configs/ic_lora_512.yaml`：LoRA rank 32、學習率 2e-4、3000 步、batch 1、bf16、gradient checkpointing；只有 reference 條件，沒有 first_frame 條件。

所有樣本的 caption 相同，前處理產生的 caption embedding 也都一樣。`python -m reveil.conditions share` 只保留一份（`shared_condition.pt`），其餘換成符號連結。Drive 不保存符號連結，新 session 用 `restore` 重建。

## 開發

```bash
pip install -e ".[dev]"
pytest
```

## 授權

Reveil 的程式碼採用 Apache-2.0。LTX-2.5 的權重和 Lightricks 的 `ltx-trainer` 屬於 LTX-2.x Community License，使用時須遵守該授權及其 Acceptable Use Policy。

NudeNet 和 Lada 的偵測模型、匯出用的 ultralytics 都是 AGPL-3.0。Reveil 不包含它們，也不 import 它們：notebook 在你的執行階段下載模型（Lada 的權重再用獨立的 `yolo` 指令匯出成 ONNX）。
