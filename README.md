# PolyMinutes

會議室即時多語翻譯。一台專用電腦以旁聽身分加入 Teams 會議，把發言即時轉錄、翻譯，投到會議室電視上；會後產出依發言者分段的多語對照逐字稿。

預設語言組合為繁體中文（台灣）、越南語、英語，可改為任意 2 至 3 種。

## 運作方式

這台電腦**不外放、不開麥克風**，純粹旁聽。所有與會者的聲音從同一條音訊進來，因此語者分離不只是為了標示「誰在講」——它同時決定每句話要用哪種語言辨識、翻成哪些語言。

```
虛擬音效裝置 → VAD 切句 → 語者嵌入 → 聚類 → 該語者的主導語言
   │                                              │
   │                              Whisper 辨識（指定語言，不自動偵測）
   │                                              │
   │                                    翻譯成其餘各語言
   │                                              │
   │                        暫定字幕（秒出）──→ 下一句翻譯時順帶修飾
   │                                              │
   │                                   字幕頁（新增 / 就地改寫）→ 電視
   └→ session_{時間}.wav（會後重跑的唯一來源）
```

順序不可對調。Whisper 的語言在建立辨識器時就固定，判錯不會優雅降級而是崩塌成重複填充詞，所以必須先確定語者才能決定語言。語者嵌入只看聲學特徵、不需要先轉錄，放在前面不增加延遲。

設計取捨與被推翻過的方案記在 [plan.md](plan.md)。

## 需求

- Python 3.12+
- Node.js 20+
- 虛擬音效裝置：Windows 用 [VB-Cable](https://vb-audio.com/Cable/)，macOS 用 [BlackHole](https://existential.audio/blackhole/)
- 約 2 GB 磁碟空間放模型
- 翻譯需要 Anthropic API 金鑰（沒有也能跑，只轉錄不翻譯）

## 安裝

### 1. 音訊路徑（最容易出錯的一步）

把 Teams 的**音訊輸出**指向虛擬音效裝置，程式再從該裝置收音。

不要改用系統靜音來達成「不外放」——Windows 的播放裝置一旦靜音，擷取到的就是一片無聲，而且沒有任何錯誤訊息，只會得到空白的逐字稿。收音頁的峰值指示就是為了讓你當場發現這件事。

### 2. 下載模型

模型不在版控裡。在專案根目錄執行（Windows 用 Git Bash；`curl` 與 `tar` 在 Windows 10 以上已內建）：

```bash
mkdir -p models && cd models
curl -L -o silero_vad.onnx https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx
curl -L -o speaker_embedding.onnx https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx
curl -L -o whisper-small.tar.bz2 https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-whisper-small.tar.bz2
tar -xjf whisper-small.tar.bz2 && rm whisper-small.tar.bz2
curl -L -o segmentation.tar.bz2 https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2
tar -xjf segmentation.tar.bz2 && rm segmentation.tar.bz2
```

`models/` 最終應包含 `silero_vad.onnx`、`speaker_embedding.onnx`、`sherpa-onnx-whisper-small/`、
`sherpa-onnx-pyannote-segmentation-3-0/`。

分段模型（6 MB）負責找出「換人講話」的時間點。少了它程式仍可運作，但會退回以整句聲紋分群：
一句話中途換人時整句只會得到一個混合的聲紋。實測一場 2 小時 19 分的會議，859 句裡有 72 句
（8.4%）不只一個人在講。

需要更高準確度可另外下載 `sherpa-onnx-whisper-medium` 或 `large-v3`，程式會自動偵測；會後處理一律使用磁碟上最大的那個。

### 3. 啟動

```bash
start.bat
```

macOS 用 `./start.command`。腳本會清掉佔用 port 的舊程序、建立虛擬環境、安裝套件、建置前端，然後開啟瀏覽器。首次執行需要幾分鐘。

## 使用

啟動後瀏覽器停在 <http://127.0.0.1:8010>。

| 頁面 | 用途 |
|---|---|
| 收音 | 選擇輸入裝置、開始／停止、即時音量與狀態 |
| 會議紀錄 | 歷次逐字稿；在此把 `S1`／`S2` 對應成真實姓名 |
| 詞彙表 | 專有名詞處理方式 |
| 已學會 | 系統自己學到的聲紋與修正，可逐項刪除 |
| 設定 → 字幕顯示 | 語言組合與電視排版 |
| 設定 → LLM | 翻譯模型與供應商 |
| 設定 → LLM 金鑰 | 多組金鑰輪替 |

**字幕投影**：在設定→字幕顯示點「開啟字幕頁」，把該視窗拖到電視那個顯示器，按 F11 全螢幕。

**開會流程**：Teams 加入會議 → 收音頁按「開始」→ 確認峰值有跳動 → 開會 → 結束後按「停止」。

### 詞彙表的三種處理方式

| 方式 | 用途 |
|---|---|
| 指定譯詞 | 強制使用你填的翻譯。公司、產品、部門名稱 |
| 保留原文 | 完全不翻譯。跨國團隊共通的英文詞如 `schedule`、`delay`，硬翻反而更難讀 |
| 僅作辨識提示 | 只提高語音辨識準確度，不介入翻譯 |
| 保護 | 宣告這是真詞：不會被改寫，也不會有別的字被改成它。用於與術語同音的日常用語，例如 `才夠` 之於 `採購` |

### 系統會自己學

三件事從日常操作累積，不需要另外設定。全部可在「已學會」頁檢視與刪除——會自動學就必須能看見和撤銷。

**認得聲音**：在會議紀錄頁把 `S1` 對應成姓名時，該語者的聲紋一併存下。下一場會議這個聲音一出現就直接標上姓名。辨識門檻比會議內的分群嚴格：分群錯了看逐字稿就會發現對話被切開，但把上個月的姓名貼到這個月的陌生人身上，會被當成真人紀錄而沒人去查。

**記住修正**：逐字稿的每一句都可以點擊修改。改完後系統記下「辨識成什麼」與「實際說什麼」，之後所有逐字稿與即時字幕都自動套用。這是整個系統唯一有現場的人背書的資料，所以字面套用、且優先於詞彙表的拼音推論。

學到的內容會擴展到詞的邊界再儲存，但絕不擴展到虛詞上。從「排**會的**需求」學到 `會的 → 櫃的` 會把「開會的時間」改成「開櫃的時間」；跳過虛詞往左擴，得到的是 `排會 → 排櫃`。

**加詞前先掃衝突**：同音字修正是字面生效的，加一個詞可能毀掉另一個。

```bash
.venv\Scripts\python.exe -m scripts.check_terms 料號 採購 工序
```

會列出這個詞在既有逐字稿裡會覆寫掉什麼，以及聲調是否一致。詞彙表頁在你輸入詞條時也會即時做同一件事，比對的是這個會議室已經錄過的內容。實例：`料號` 會把 `料耗` 改掉 42 次，而料耗是製造業真正的術語——沒跑這步就加詞，這 42 處會靜默損毀。

衝突不等於否決。兩邊都是真詞就**兩個都加**：詞彙表裡的詞永遠不會被改寫成另一個詞，`供需` 與 `工序` 因此能共存。兩個詞彙表詞條互為同音時（`生管` 與 `升官` 去掉聲調都是 shengguan），由聲調決定——`生館` 是 sheng1guan**3**，比對到 `生管` 而非 `升官`。

**挖出自己不認得的詞**：LLM 修正被防線擋下的提議，正好指出系統的盲點——模型想寫 `工程變更` 而辨識器寫了 `一夕變更`，被擋的原因就是詞彙表沒有這個詞。

```bash
.venv\Scripts\python.exe -m scripts.learn_terms transcripts/*.txt --ollama qwen3:14b
```

跨逐字稿統計重複出現的候選並列出，確認後加 `--apply` 才寫入詞彙表。

### 語者與語言

程式以聲紋分辨發言者，但取不到 Teams 的參與者名單，所以先標 `S1`、`S2`。在會議紀錄頁對應一次姓名即可。

每位語者的語言由累積統計決定，需要連續數句不符才會切換。中文與英語之間的門檻特別高——台灣的中文經常夾雜英文詞（「這個 schedule 要 delay 一週」），一句話裡幾個英文字不該讓程式判定他改講英語了。若某位固定講某種語言，可在設定中釘死。

中文輸出一律轉為台灣繁體。Whisper 對中文一律輸出簡體，不轉換的話電視上會跳出簡體字。

## 設定與資料

執行期產生的檔案都不進版控：

| 檔案 | 內容 |
|---|---|
| `config.json` | 語言、輸入裝置、模型、顯示格式 |
| `llm.json` | LLM 供應商設定 |
| `llm_keys.json` | 輪替用的 API 金鑰 |
| `polyminutes.db` | 詞彙表、會議紀錄、逐字稿 |
| `recordings/` | 原始錄音 |
| `transcripts/` | `bench_wav` 產出的逐字稿 |

環境變數可覆寫：`ANTHROPIC_API_KEY`、`POLYMINUTES_INPUT_DEVICE`、`POLYMINUTES_LANGUAGES`、`POLYMINUTES_WHISPER_MODEL`、`POLYMINUTES_GPU_INDEX`。

### 隱私模式：全程不出機

客戶訪談常有「資料不能上雲」的硬需求。在設定→LLM 把供應商的金鑰留空、但為 Ollama 配一個本地 model，即時翻譯與會後摘要／校正就自動改走本機 Ollama——逐字稿不離開這台機器。

這是自動的：選定的雲端供應商缺金鑰時，只要設定裡有配 Ollama model 就退到本地，沒配才維持「只轉錄不翻譯／不摘要」。用的是你為 Ollama 配的 model，不是雲端的 model 名（Ollama 拉不到 `claude-opus-5`）。不主動探測 Ollama 是否啟動——daemon 沒開就照一般供應商連不上的方式記為失敗。

即時翻譯走本地 model 會比雲端慢，字幕會延遲而非掉字；不想要就別為 Ollama 配 model。

### 兩張顯卡：讓辨識與本地 LLM 各佔一張

`POLYMINUTES_GPU_INDEX`（預設 `0`）指定辨識器用哪張 CUDA 卡。單張卡不必理會。

會後的摘要與辨識校正若走本地 Ollama（不出機，適合客戶訪談），Ollama 預設也搶第 0 張卡，會與正在錄音的辨識器搶記憶體。兩張卡時把它們分開：辨識器留 `POLYMINUTES_GPU_INDEX=0`，Ollama daemon 以 `CUDA_VISIBLE_DEVICES=1 ollama serve` 啟動，落在另一張。走雲端（Anthropic）則沒有這個問題，摘要不碰 GPU。

## 開發

```bash
.venv\Scripts\python.exe -m server.test_audio      # 裝置解析與設定
.venv\Scripts\python.exe -m server.test_pipeline   # 辨識與語者判斷邏輯
.venv\Scripts\python.exe -m server.test_e2e        # HTTP API 與完整管線
```

用既有錄影檔測辨識率，不必接虛擬音效裝置：

```bash
ffmpeg -i meeting.mp4 -ac 1 -ar 16000 -c:a pcm_s16le recordings/test01.wav
.venv\Scripts\python.exe -m scripts.bench_wav recordings/test01.wav --ref ref.txt
```

輸出分段逐字稿、realtime factor、每位語者的主導語言，以及詞彙表裡哪些詞被辨識出來。`--ref` 給一份人工聽打的參考逐字稿就會算 CER（中文沒有詞邊界，用字錯誤率而非 WER）。`--model medium` 可比較不同模型層級。

### 專有名詞與辨識錯誤修正

三層，由便宜到昂貴，各自擋不同的錯：

**1. 解碼時偏置** — GPU 路徑把詞彙表當作 faster-whisper 的原生 hotwords 傳進模型。sherpa-onnx 的 Whisper 沒有這個能力（contextual biasing 只支援 transducer），CPU 路徑沒有這一層。

**2. 解碼後拼音修正**（`server/correct.py`，一律啟用）— 把結果和詞彙表逐詞比對去聲調拼音，發音完全相同就換成詞彙表的寫法：`公單 → 工單`、`微剛科技 → 威剛科技`、`生館 → 生管`。

中文只接受發音完全相同。七份真實逐字稿實測，容許一個編輯距離會把「知道」改成「製造」156 次、「生產」改成「生管」146 次，共 1578 處誤改——中文音節密度太高。英文與越南語詞容許 25%（`incent → Vincent`），拉丁詞彙夠稀疏。

詞彙表適合放公司、產品、模組名稱。放 `採購` 這種常見雙字詞會把「才夠」改掉。

**3. LLM 上下文修正**（`scripts/refine_transcript.py`，手動執行）— 辨識器一次只看一句，不知道自己身處一場 SAP ERP 訪談。把逐字稿分批送給 LLM，附上會議主題與詞彙表，讓它依上下文修正。

```bash
.venv\Scripts\python.exe -m scripts.refine_transcript transcripts/會議.txt --ollama qwen3:14b
```

`--ollama` 走本機模型，逐字稿不離開這台機器——對客戶訪談而言這是預設選擇。不加 `--ollama` 則用 `llm.json` 設定的雲端模型。輸出寫到 `<檔名>.refined.txt`，並印出每一處改動，原檔不動。

危險的地方和有用的地方是同一件事：模型被要求修逐字稿時會順手潤飾，而潤飾出來的句子沒人說過。四道防線：

| 防線 | 擋掉什麼 |
|---|---|
| 行數必須相同 | 整批重組 |
| 單行變動 ≤ 30% | 改寫成更通順的句子 |
| 語音距離 ≤ 20% | 憑語意猜測（`延伸 → 選項` 不是聽錯，是猜的） |
| 詞彙表詞條放寬到 50% | 保留 `一夕變更 → 工程變更` 這類辨識器不可能知道的詞 |

不確定一律保留原文。寧可留錯，不可造假。

### 改動門檻前先跑回歸

三個決定逐字稿是被修好還是被改壞的門檻，都曾在手寫測試下看起來正確、在真實語音上出錯。

```bash
.venv\Scripts\python.exe -m scripts.regress
```

拿真實門檻跑真實語料，輸出修正數與**疑似誤改**數，並和 `transcripts/regress-baseline.json` 比較。疑似誤改的判定是：被改掉的原文本身在語料中反覆出現，也就是它是個詞而不是誤辨識——`料耗` 出現 42 次卻被 `料號` 改掉，就是這樣被抓到的。

改任何門檻或加詞後跑一次；數字變差會以 exit code 1 回報。確認新數字合理後用 `--save` 更新基線。

基線與語料都不進版控——它們衍生自實際會議內容。這代表新環境要先跑過一次會議、或放幾份逐字稿進 `transcripts/final/`，這個檢查才有東西可比。

### 量測辨識、語者、幻覺三項指標

`regress` 只看修正層；要量整條管線的品質，拿一份人工標好真實語者的參考稿，和管線產出的假設稿比對：

```bash
.venv\Scripts\python.exe -m scripts.eval_harness --ref transcripts/eval/會議.ref.txt --hyp transcripts/eval/會議.hyp.txt --save
```

兩份都是標準逐字稿格式（`[分:秒] S1 (語言) 內容`）。參考稿每行標真實語者，假設稿放 `bench_wav` 或匯出的結果。輸出三項：

- **準確率**：逐語言 CER（中文）/ WER（拉丁語系）。文字依「被解碼成的語言」分桶，語言判錯會誠實反映成該桶準確率下降。
- **語者歸屬**：參考與假設的語者標籤最佳對應後（S1 這邊不等於 S1 那邊），錯歸的語音佔比即 error，以字數加權。`hyp_speakers < ref_speakers` 直接印 `COLLAPSE`——就是共用麥克風把所有人歸成一位的失效。
- **幻覺率**：片語過濾器標記的語音佔比，分語言。乾淨跑接近 0；為壓越南語幻覺去動門檻若誤傷過濾器，這裡會升。

和 `regress` 一樣對基線比較、數字變差以 exit code 1 回報，`--save` 更新基線。`transcripts/eval/` 與基線不進版控。`--selfcheck` 跑內建 asserts。

前端另外開一個 dev server（會透過 `VITE_API_URL` 連到後端）：

```bash
cd dashboard && npm run dev
```

## 效能

量測環境：20 邏輯核心 + RTX 5060 Ti 16 GB。

| 路徑 | 模型 | Realtime factor | 機器可用性 |
|---|---|---|---|
| GPU（有顯卡時自動使用） | large-v3 float16 | 0.15 – 0.30 | CPU 約 12%，可正常工作 |
| CPU | small int8、4 執行緒 | 0.20 | 尚可 |
| CPU | small float32、全部核心 | 0.57 | 整台機器卡死 |

會後重跑七場訪談共 9.5 小時音訊，GPU 上 1 小時 30 分完成。同樣的量在 CPU 上約需 8.5 小時，且期間無法使用電腦——這是 `--threads` 預設只取一半核心的原因。

即時字幕延遲：一般短句約 2 秒，16 秒長句 4.8 秒（含模型首次載入）。

沒有顯卡也能跑，退回 sherpa-onnx CPU 路徑；設 `POLYMINUTES_NO_GPU=1` 可強制不用 GPU。

### GPU 安裝

```bash
.venv\Scripts\python.exe -m pip install -r requirements-gpu.txt
```

CUDA runtime 來自 pip wheel，不需要另外裝 CUDA Toolkit。兩個踩過的坑：

- **Blackwell（RTX 50 系列，sm_120）需要 CTranslate2 4.8 以上**。多數教學釘的 CUDA 12.1 版本不涵蓋這代顯卡
- CTranslate2 透過 **PATH** 尋找 cuBLAS 與 cuDNN，`os.add_dll_directory()` 無效——模型會載入成功，第一次 encode 才報 `cublas64_12.dll is not found`。`server/asr_gpu.py` 在 import 時處理這件事

## 已知限制

- **越南語的幻覺率明顯高於中文**。large-v3 的越南語訓練資料多來自 YouTube 字幕，遇到聽不清的片段會吐出「請訂閱頻道」這類台詞。七場訪談實測佔越南語句數 14.8%（中文 0.2%、英語 0.1%）。兩層防線：已知的字幕台詞由片語過濾器擋掉；GPU 路徑另外過濾「解碼器自評近乎確定是靜音卻仍吐出文字」的片段（`no_speech_prob ≥ 0.85`）——這類幻覺常帶高信心分數，會滑過 faster-whisper 內建耦合了 logprob 的 no_speech 門檻。門檻可用 `scripts.eval_harness` 對標註語料量測後調整。即便如此，越南語段落的可信度本來就較低
- **辨識率仍無 CER 數字**。七場訪談已產出逐字稿，但沒有人工聽打的參考稿可比對，只能定性判斷「可讀」
- 拼音修正只對中文有效。越南語的專有名詞沒有對應機制
- 詞彙表放常見雙字詞有風險。`採購` 會把「才夠」改掉，`料號` 會把「料耗」改掉——同音本身就是歧義。適合放的是公司、產品、模組名稱這類獨特詞
- **僅在 Windows 實測過**。macOS 路徑已寫好但未在實機驗證
- 遠端與會者看不到會議室電視。字幕只服務現場
- **語者分離在會議室錄音下經常失效**。七場真實訪談實測，五場的主導語者佔逐字稿 97–99% 的行數——所有人被歸成同一位。其中一場 37 分鐘的語音，門檻從 0.35 到 0.60 都只分出一群。會議室共用麥克風經 Teams 壓縮降噪後，聲紋差異被抹平

  影響不只是姓名標籤：語者決定每句話用哪種語言辨識，所有人歸為一位代表整場只用一種語言解碼。門檻改為 0.65 後，七場裡有四場的語者時長分布變得合理（例如 `[99, 0, 0]` → `[39, 28, 20, 6]`），三場沒有改變

- **越南語沒有因此變多**。上一句的推論是「語者分離修好後，越南語會從被誤判為中文的狀態回來」——實測推翻：越南語 265 → 205 句。分離出更多語者反而讓每位語者的語言統計樣本變少，只講兩三句的語者若被誤判一次就會整段被強制用錯語言重解。現在語者需累積 4 句才能自行認定語言，不足者沿用整場的主導語言

- **跨會議聲紋辨識未經驗證**。應該用來驗證它的素材（兩家公司七場訪談、有共同與會者）無法驗證，因為分群已經把每場併成一位語者，跨場比對等於「A 場所有人」對「B 場所有人」，相似度 0.69–0.94 什麼也證明不了
- 兩人同時講話時，切出的片段混有兩人聲音，語者分離會不穩
- 會議室發言經 Teams 壓縮與降噪後才進來，辨識率低於直接收音

## 授權與致謝

MIT，見 [LICENSE](LICENSE)。第三方授權聲明見 [NOTICE](NOTICE)。

管理介面衍生自 [OpenWA-Lab](https://github.com/JaplinChen/OpenWA-Lab)（MIT）。整體架構參考 [meetily](https://github.com/Zackriya-Solutions/meetily)，但改為本機服務加瀏覽器，未採用其 Tauri 外殼。

語音處理使用 [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx)：Silero VAD、Whisper、3D-Speaker 語者嵌入。簡繁轉換使用 [OpenCC](https://github.com/BYVoid/OpenCC)。
