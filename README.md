# Reveil

用 LTX-2.5 的 IC-LoRA 還原影片裡的馬賽克。訓練資料預設採用市售發行版與無碼流出版的真實配對（亦保留無碼影片加假馬賽克的合成流程）。目前完成第一步：產生訓練資料，並在 Colab A100 上訓練 IC-LoRA。推論（實際還原影片）是下一步。

## 做法

- 訓練資料預設來自有碼發行版與無碼流出版的**真實配對**（real pairs），亦支援無碼影片加假馬賽克的**合成配對**（synthetic pairs）作為替代流程。
- 底模是 LTX-2.5 22B（dev），用 Lightricks 官方 `ltx-trainer` 的 reference 條件化（IC-LoRA）訓練：打碼的裁切當作參考 token（乾淨、timestep 0、和目標同位置、不算 loss），無碼（或複合）的裁切是要生成的目標。
- 每個樣本是一對 512×512、121 影格的影片。所有樣本用同一句 caption，推論時重用同一份 caption embedding，不需要文字編碼器。

## 訓練資料

訓練素材只能使用合法取得、所有演出者皆確認成年的影片。素材數量與多樣性要充足（建議 20 部以上、不同演員和場景）；配對太少，模型只會背下那幾段。依據資料流程的需求：

- **真實配對（預設）**：需要同一部片的兩種版本——市售有碼發行版（retail / censored）與流出的無碼版（leak / uncensored）。
- **合成配對（替代流程）**：需要無碼原始影片；訓練是在無碼畫面上打假馬賽克，原片即為正確答案，因此純有碼片或 AI 去碼版本不能用於此流程。

### 真實配對（預設）

真實配對流程使用同部影片的市售有碼發行版與流出的無碼版。

#### 目錄配置

兩側影片依片名資料夾配對，兩側每個片名資料夾各放一部影片，檔名可以不同：

```
sources/
├── censored/
│   └── <片名>/
│       └── <影片檔名>
└── uncensored/
    └── <片名>/
        └── <影片檔名>
```

#### 處理步驟

依序執行以下三個指令：

1. **對齊影格（`reveil.align`）**：比對兩側特徵並建立逐格對應關係，輸出 `align/<片名>.json`（`Alignment` 格式）。
   ```bash
   python -m reveil.align --censored sources/censored/ --uncensored sources/uncensored/ --out align/ --cache cache/ --workers 4
   ```
   - 主要參數：`--censored` 與 `--uncensored` 指定兩側片名資料夾；`--out` 指定輸出目錄；`--cache` 快取縮圖特徵以加速重跑；`--workers` 平行處理不同片名；`--decode-threads` 每部影片解碼執行緒數（預設 2）；`--only` 僅對齊指定片名。單一影片配對可用 `--retail` 與 `--leak`。
   - 已完成的片名會自動跳過，`--overwrite` 重新對齊。

2. **偵測馬賽克與疊圖（`reveil.realmask`）**：在對齊的影格中找出真實馬賽克位置與單側疊圖（片商或網站標誌），輸出 `regions/real/<片名>.json`（區域標註格式，增加 `block` 與 `overlays`）。
   ```bash
   python -m reveil.realmask --align-dir align/ --out regions/real/ --workers 4
   ```
   - 主要參數：`--align-dir` 對齊檔目錄；`--out` 輸出目錄；`--stride` 分析影格間隔（預設 2）；`--scale` 偵測縮放比例（預設 0.5）；`--workers` 平行處理不同片名；`--decode-threads` 解碼執行緒數；`--only` 僅處理指定片名。
   - 已完成的片名會自動跳過，`--overwrite` 重做。

3. **切分視窗與產生配對（`reveil.real_pairs`）**：依區域追蹤切成 121 影格視窗並生成成對訓練影片，輸出架構與 `make_pairs` 相同。
   ```bash
   python -m reveil.real_pairs --align-dir align/ --regions-dir regions/real/ --out data/ --workers 4
   ```
   - 主要參數：`--align-dir` 對齊檔目錄；`--regions-dir` 區域檔目錄；`--out` 輸出目錄；`--target` 訓練目標類型（`composite` 或 `leak`，預設 `composite`）；`--max-windows-per-title` 限制每部片最多產生的視窗數；`--config` 載入 YAML 參數覆寫；`--seed` 隨機種子；`--workers` 平行處理；`--decode-threads` 解碼執行緒數；`--only` 僅處理指定片名。
   - 已完成的片名會自動跳過，`--overwrite` 重做。
   - 輸出目錄結構包含 `data/clean/*.mp4`（目標）、`data/mosaic/*.mp4`（參考）、`data/meta/<片名>.json`（各視窗參數）與 `data/dataset.jsonl`（給訓練器的清單）。

#### 訓練目標：複合畫面（Composite）與完整無碼（Leak）

- 預設目標為**複合畫面（`--target composite`）**：以有碼發行版畫面為底，僅在偵測到的馬賽克範圍（外擴半個 block 並羽化）填入校色後的無碼畫面。
  - 原因：實測顯示發行版裁切至 legal range、色度比無碼版弱 1–5%、無碼版較銳利（Laplacian 變異數高達 5 倍），且發行版邊角常帶有片商或網站標誌（佔全片 14–96%）。若直接以整張無碼畫面為目標，模型會被誤導去學習銳化、重新校色或抹除標誌；採用複合目標，模型專注於學會「去除馬賽克」。
- **完整無碼畫面（`--target leak`）**：整張畫面使用校色後的無碼影片作為目標，並自動剔除任何觸及疊圖（overlay，如標誌）的視窗。

#### 實測數據特徵（基於 14 部可用影片）

- **時間對齊**：兩版本通常畫面相同但有固定時間偏移（片頭 0 至 ±90 秒）。但在實測中：
  - 1 部無碼版為未剪輯加長版（長度為發行版兩倍，發行版只是其中一段剪輯）；
  - 1 部無碼版整部的剪接長短都不同，約每 30–60 秒對應的時間位置就改變一次（差幾秒到約 30 秒）；
  - 1 部無碼版幀率為 59.94 fps；
  - 1 部無碼版約每 1000 格重複一格。
  - `reveil.align` 採逐格比對處理上述所有狀況，僅單側存在的內容保持未配對。
- **畫面幾何**：兩側幾何完全一致（無裁切、縮放或黑邊）。其中 1 部發行版解析度為 1280×720，其無碼版會自動等比縮小對齊。
- **畫質與色彩**：無碼版通常較銳利（Laplacian 變異數高達 5 倍）；發行版色度弱 1–5% 且裁切至 legal range。
- **疊圖**：發行版邊角常有標誌（佔全片 14–96%），無碼版則無。
- **馬賽克分佈**：真實馬賽克覆蓋 33–67% 的影格，在 1080p 下區塊大小約 20–48 px。
- **實測成果範例**：在一條 3 分鐘實測片段中，pipeline 配對了 98.5% 的影格，在 58% 的分析影格中偵測到馬賽克，並成功辨識發行版標誌為疊圖；合成的複合目標與有碼參考畫面僅在 4–8% 的畫布面積上存在差異。

#### 運算成本

每部影片需經過三次解碼（align、realmask、real_pairs），每部片總計約需 1–2 CPU 小時，可透過 `--workers` 跨片名平行處理。

### 資料格式

#### 對齊檔（Alignment）

`reveil.align` 輸出的 `align/<片名>.json`（版本 1）記錄發行版與無碼版之間的影格對應關係：

```json
{
  "version": 1,
  "retail": "sources/censored/TITLE/video.mp4",
  "leak": "sources/uncensored/TITLE/video.mp4",
  "retail_frames": 120000,
  "leak_frames": 120500,
  "retail_fps": 23.976,
  "leak_fps": 23.976,
  "retail_size": [1920, 1080],
  "leak_size": [1920, 1080],
  "runs": [
    [0, 1440, 2400]
  ],
  "segments": [
    {"retail_start": 0, "retail_stop": 2400, "offset_s": 60.06}
  ],
  "stats": {
    "matched_share": 0.985
  }
}
```

- `runs`：連續對齊的影格區段清單，每項為 `[retail_start, leak_start, length]`，代表從發行版 `retail_start` 與無碼版 `leak_start` 起連續 `length` 格完全對齊。runs 依 `retail_start` 排序，允許無碼版鏡頭調換或跳格。
- `segments`：巨觀對齊段落，包含各段的開始與結束影格以及時間位移（`offset_s`）等。
- `stats`：包含配對影格比例（`matched_share`）與耗時等統計資訊。

#### 區域標註檔（Regions）

每部影片需要一個區域標註檔 `regions/<影片檔名>.json`，標出馬賽克的位置與形狀。在合成流程中可用[自動標註](#自動標註)產生，在真實配對流程中由 `reveil.realmask` 產生，亦可用任何工具輸出此格式：

```json
{
  "version": 1,
  "frames": {
    "0": [{"box": [x0, y0, x1, y1], "block": 32}],
    "1": [{"polygon": [[x, y], [x, y], [x, y]], "block": 24.5}]
  },
  "cuts": [240],
  "overlays": [
    {
      "box": [1720, 30, 1880, 90],
      "side": "retail",
      "frequency": 0.85,
      "frames": [0, 9000]
    }
  ]
}
```

- frame key 是從 0 起算的解碼影格編號，座標是原始像素。
- 有 `polygon` 時馬賽克照輪廓打；只有 `box` 時隨機用矩形或內切橢圓。
- `block`（可選）：真實馬賽克在原始像素下的方塊大小（例如 1080p 下約 20–48 px），由 `realmask` 測得。
- `cuts`（可選）：每個場景切換後的第一格，track 和視窗不會跨過它。
- `overlays`（可選）：僅出現在單側（如發行版標誌、浮水印）的疊圖清單，包含：
  - `box`：`[x0, y0, x1, y1]`，發行版原始像素座標。
  - `side`：出現的單側（`"retail"` 或 `"leak"`）。
  - `frequency`：在分析影格中出現的頻率比例（0.0–1.0）。
  - `frames`（可選）：`[start, stop]` 影格區間，標明該疊圖出現的範圍。

### 合成配對（替代流程）

#### 自動標註

`reveil.annotate` 用 YOLO 偵測或分割模型，在無碼影片裡逐格找出要打假馬賽克的部位，寫成上述格式的標註檔。它找的是露出的部位，不是找馬賽克（還原有碼片時找馬賽克是推論階段的事）：

```bash
pip install -e ".[annotate]"  # onnxruntime；用 GPU 的話改裝 onnxruntime-gpu
python -m reveil.annotate --videos sources/ --model detector.onnx --out regions/ --preview-dir previews/
```

- 模型要先用 ultralytics 匯出成 ONNX：`yolo export model=<權重>.pt format=onnx dynamic=True`。Reveil 只讀 ONNX 檔，不 import ultralytics。
- 影片直接解碼成模型輸入的大小（例如 640），在這個大小上找區域，再換算回原始像素；4K 影片不會比 1080p 慢太多。
- 分割模型的遮罩變成輪廓多邊形，偵測模型的框變成方框，都會再外擴一點（`--grow`，畫面短邊的比例）。
- 每格的偵測結果會連成 track：短於 `--min-track-seconds` 的丟掉，`--max-gap-seconds` 以內的漏偵測用前後影格補上。偵測到的場景切換寫進 `cuts`。
- `--classes` 只保留指定類別（名稱或編號），`--conf` 是分數門檻，`--stride N` 每 N 格偵測一次（中間的影格會補上）。執行時會印出模型的所有類別。
- 不可靠的偵測整段丟掉，不打碼：每個區域各自連成 track，分數中位數低於 `--min-track-score` 的（例如被誤判的手、玩具）、或大半影格是「可疑」偵測的。可疑是指框有 `--doubt-inside`（預設 0.7）以上落在 `--doubt-classes` 類別的框裡（例如臉：舌吻的嘴被誤判成生殖器；口交時真正的生殖器和臉重疊通常不到一半），或框裡的灰階標準差低於 `--min-contrast`（預設 8，0–255；例如暗到看不出東西的畫面，打了馬賽克也沒有東西可以還原）。丟掉的區域記在標註檔的 `dropped` 裡。
- `--preview-dir` 每部影片輸出一張預覽圖，綠線是標註，紅線是丟掉的偵測；產生配對前先看過。
- 已經有標註檔的影片會跳過，`--overwrite` 重做。標註設定（模型、類別、門檻等）記在輸出資料夾的 `annotate_settings.json`，設定改了就全部重新標註；所以 `--out` 請用自動標註專用的資料夾。

Colab notebook 預設用 [NudeNet](https://github.com/notAI-tech/NudeNet) 的 `640m` 偵測模型（官方 GitHub 的下載現在要登入，notebook 依序改從 Hugging Face 上的副本下載，並檢查下載到的不是登入頁或錯誤頁），只取 `FEMALE_GENITALIA_EXPOSED` 和 `MALE_GENITALIA_EXPOSED` 兩類，並把 `ANUS_EXPOSED` 的框從生殖器的框裡挖掉（`--exclude-classes`）：一般有碼片只遮生殖器，肛門不打碼，假馬賽克也要一樣。大半落在 `FACE_FEMALE`、`FACE_MALE` 框裡的偵測算可疑（`--doubt-classes`）。也可以改用 [Lada](https://github.com/ladaapp/lada) 的分割模型 `lada_nsfw_detection_model_v1.3`；它只有一類 `nsfw`，會連肛門一起標，也沒有臉的類別。每個偵測模型的標註和預覽各放一個資料夾（`regions/<偵測模型>/`、`previews/<偵測模型>/`），換模型就會重新標註。

#### 產生配對

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
- Colab 密鑰 `HF_TOKEN`（先在 Hugging Face 同意 LTX-2.5 的條款）。
- 訓練素材：
  - **真實配對（`DATA_MODE = "real"`，預設）**：發行版放在 Drive 的 `reveil/sources/censored/<片名>/`，無碼版放在 `reveil/sources/uncensored/<片名>/`。兩側每個片名資料夾各放一部影片，可以是 Drive 捷徑；依片名資料夾配對，檔案名稱可以不同。
  - **合成配對（`DATA_MODE = "synthetic"`）**：無碼原始影片放在 Drive 的 `reveil/sources/FC2/`（可由 `SOURCES` 設定），子資料夾和捷徑都會讀。純有碼影片留給未來去碼使用，不要放進 `SOURCES`。

notebook 依序完成環境檢查、安裝相依套件、下載 LTX-2.5 模型（約 72 GB）、準備訓練資料、前處理以及 IC-LoRA 訓練。

### 模式切換與冒煙測試

- **`DATA_MODE` 開關**：
  - `"real"`（預設）：執行真實配對流程。依序執行 `reveil.align`、`reveil.realmask` 與 `reveil.real_pairs`。對齊檔存於 Drive 的 `reveil/align/`，區域標註存於 `reveil/regions/real/`，斷線重跑會自動沿用。執行時僅輸出各片名的數字統計（影格數、區塊大小中位數、疊圖數、對齊比例），不顯示影像預覽。
  - `"synthetic"`：切換為合成配對流程。下載偵測模型存至 Drive 的 `reveil/models/`，標註存於 `reveil/regions/<偵測模型>/`，預覽存於 `reveil/previews/<偵測模型>/`，再以假馬賽克產生配對。
- **`SMOKE_TEST` 冒煙測試**：
  - 預設為 `True`。初次執行建議保持開啟，確認整條管線能正常運作。
  - 在 `DATA_MODE == "real"` 下：從選取的片名中挑出兩側檔案大小合計最小的一組，執行真實的對齊與馬賽克偵測，最多只產生 1 個視窗（`--max-windows-per-title 1`），並在訓練階段只跑 20 步。產生的對齊與區域檔會保存在 Drive，正式訓練時可直接沿用不需重算。
  - 在 `DATA_MODE == "synthetic"` 下：採用隨機區域（不跑模型標註），每部影片最多產生 1 個視窗，訓練 20 步。
  - 冒煙測試驗證無誤後，將 `SMOKE_TEST` 改為 `False` 進行正式訓練（預設 3000 步）。每部片依 `MAX_WINDOWS_PER_TITLE`（真實配對）或 `MAX_WINDOWS_PER_VIDEO`（合成配對）取樣（預設 100 組）；若正式配對總數低於 `MIN_SAMPLES`（預設 100）時，notebook 會中斷以防模型過擬合。

配對影片與 latent 前處理檔案依 `DATA_MODE` 與 `SMOKE_TEST` 分開儲存在 Drive 的 `reveil/pairs/<DATA_MODE>_<KIND>/` 與 `reveil/precomputed/<DATA_MODE>_<KIND>/`（`KIND` 為 `smoke` 或 `main`），兩種模式互不混淆。加入新素材後可設 `REBUILD_DATA = True` 增量產生資料。

訓練設定在 `configs/ic_lora_512.yaml`：LoRA rank 32、學習率 2e-4、3000 步、batch 1、bf16、gradient checkpointing；只有 reference 條件，沒有 first_frame 條件。checkpoint 每 250 步存到 Drive 的 `reveil/runs/<RUN_NAME>/checkpoints/`，斷線重跑會從最新的 checkpoint 續訓。

所有樣本的 caption 相同，前處理產生的 caption embedding 也都一樣。`python -m reveil.conditions share` 只保留一份（`shared_condition.pt`），其餘換成符號連結。Drive 不保存符號連結，新 session 用 `restore` 重建。

## 開發

```bash
pip install -e ".[dev]"
pytest
```

## 授權

Reveil 的程式碼採用 Apache-2.0。LTX-2.5 的權重和 Lightricks 的 `ltx-trainer` 屬於 LTX-2.x Community License，使用時須遵守該授權及其 Acceptable Use Policy。

NudeNet 和 Lada 的偵測模型、匯出用的 ultralytics 都是 AGPL-3.0。Reveil 不包含它們，也不 import 它們：notebook 在你的執行階段下載模型（Lada 的權重再用獨立的 `yolo` 指令匯出成 ONNX）。
