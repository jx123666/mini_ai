# mini_ai 鈥斺€?鏍稿績鎺ㄧ悊妯″潡瀛︿範瀹炵幇

瀵圭収璇剧▼浠诲姟銆屾牳蹇冩帹鐞嗘ā鍧椼€嶄腑鐨勪袱鍧楀唴瀹癸紝鍦ㄦ湰浠撳簱瀹炵幇锛?
1. **鍗曞厓娴嬭瘯**锛堟瀬绠€妗嗘灦 + GoogleTest锛?2. **Tensor 涓庡熀纭€绠楀瓙**锛坢atmul / rmsnorm / softmax锛?3. **Tokenizer**锛圓SCII銆丣SON 璇嶈〃銆丅PE 鍚堝苟銆乪ncode/decode roundtrip锛?
鍛藉悕绌洪棿涓庤矾寰勫埢鎰忎笌 `mini-llama` 璇剧▼鏂囨。淇濇寔涓€鑷达紙`mini_llama/...`锛夛紝鏂逛究瀵圭収闃呰瀹屾暣椤圭洰銆?
---

## 蹇€熷紑濮?
```powershell
cd D:\SoftWareWork\ProjectWork\ProjectAI\mini_ai
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release
ctest -C Release --output-on-failure
```

涔熷彲浠ョ洿鎺ヨ窇锛?
```powershell
.\Release\mini-ai-tests.exe          # 鏋佺畝妗嗘灦
.\Release\mini-ai-gtests.exe         # GoogleTest锛堥渶鑱旂綉涓嬭浇渚濊禆锛?```

鑻ヤ笉鎯充笅杞?GoogleTest锛?
```powershell
cmake .. -DMINI_AI_FETCH_GTEST=OFF
```

---

## 姣忎釜鏂囦欢鐨勪綔鐢?
### 鏋勫缓

| 鏂囦欢 | 浣滅敤 |
|------|------|
| `CMakeLists.txt` | 缂栬瘧鏍稿績婧愮爜銆佹瀬绠€娴嬭瘯銆佸彲閫?GoogleTest锛涙敞鍐?`ctest` |

### 鏍稿績澶存枃浠?`include/mini_llama/`

| 鏂囦欢 | 浣滅敤 |
|------|------|
| `tensor.h` | 瀹氫箟绋犲瘑 F32 寮犻噺锛歚data` + `shape`锛屼互鍙婄储寮?/ reshape API |
| `ops.h` | 澹版槑 Matmul銆丩inear銆丷msNorm銆丼oftmax 绛夊熀纭€绠楀瓙 |
| `matmul_dispatch.h` | 鐭╅樀涔樺疄鐜版ā寮忔灇涓句笌鍒嗗彂鎺ュ彛锛堝涔犻樁娈甸粯璁?naive锛?|
| `tokenizer.h` | 鍒嗚瘝鍣ㄦ帴鍙ｏ細ASCII銆丣SON 璇嶈〃銆丅PE |
| `gguf_tokenizer.h` | 鍚屼竴濂?BPE锛岃瘝琛ㄦ潵鑷?GGUF 鍏冩暟鎹紙鏈粨搴撶敤 LoadPieces 鍠傚叆锛?|

### 鏍稿績瀹炵幇 `src/`

| 鏂囦欢 | 浣滅敤 |
|------|------|
| `tensor.cc` | 瀹炵幇 FlatIndex锛堣浼樺厛锛夈€丄t2銆丷owPtr銆丷eshapeChecked 绛?|
| `ops.cc` | 瀹炵幇 **RmsNorm / Softmax**锛堥獙鏀舵墜鍐欓噸鐐癸級浠ュ強 SiLU銆丼wiGLU 绛?|
| `matmul_optimized.cc` | 瀹炵幇 **MatmulNaive / LinearNaive** 涓?Dispatch锛堥獙鏀舵墜鍐欓噸鐐癸級 |
| `tokenizer.cc` | **ASCII tokenizer**锛氫竴涓彲瑙佸瓧绗︿竴涓?id |
| `json_vocab_tokenizer.cc` | 浠?JSON 璇嶈〃鍋氭渶闀垮尮閰?|
| `bpe_tokenizer.cc` | 璇诲彇 vocab / merges锛岃皟鐢?BPE 鍚堝苟 |
| `bpe_common.h` | **BPE 鍚堝苟寰幆**锛坔ello 涓€姝ユ鎷艰捣鏉ョ殑鍦版柟锛?|
| `gguf_tokenizer.cc` | 鍜?BPE 鍚屼竴濂楀悎骞讹紝鍙槸鏁版嵁鏉ユ簮涓嶅悓 |

### 鏋佺畝娴嬭瘯妗嗘灦 `tests/`

| 鏂囦欢 | 浣滅敤 |
|------|------|
| `test_main.h` | 鏂█瀹?+ `RegisterTest` 澹版槑 |
| `test_main.cc` | 娴嬭瘯 runner锛氶亶鍘嗘敞鍐岃〃骞舵墦鍗?PASS/FAIL |
| `test_tensor.cc` | Tensor 琛屼负鍗曟祴锛堟瀬绠€妗嗘灦锛?|
| `test_ops.cc` | 绠楀瓙鍗曟祴锛堟瀬绠€妗嗘灦锛夛紝鍚?matmul/rmsnorm/softmax |
| `test_tokenizer.cc` | ASCII roundtrip銆佹渶闀垮尮閰嶃€丅PE 鍚堝苟 |

### GoogleTest 鐗堟湰 `tests/gtest/`

| 鏂囦欢 | 浣滅敤 |
|------|------|
| `test_tensor.cc` | 涓庢瀬绠€鐗堢浉鍚屽満鏅紝鍐欐硶鎹㈡垚 `TEST` / `EXPECT_*` |
| `test_ops.cc` | 鍚屼笂锛岃鐩栦笁澶ф牳蹇冪畻瀛?|
| `test_tokenizer.cc` | GoogleTest 鐗堝垎璇嶅櫒鐢ㄤ緥 |

---

## 楠屾敹鏍囧噯瀵圭収

| 楠屾敹椤?| 鏈」鐩浣曟弧瓒?|
|--------|----------------|
| 鑳界敤 GoogleTest 鍜屾瀬绠€妗嗘灦鍐欐祴璇?| `tests/` + `tests/gtest/` 涓ゅ骞跺瓨 |
| 鐞嗚В娴嬭瘯濡備綍瀹氫綅鎺ㄧ悊閿欒 | 姣忎釜绠楀瓙鐙珛鐢ㄤ緥锛涘け璐ヤ細鎵撳嵃鏂囦欢:琛屽彿 |
| 鑳芥墜鍐?matmul / rmsnorm / softmax | 瑙?`matmul_optimized.cc`銆乣ops.cc` 涓甫鍏紡娉ㄩ噴鐨勫疄鐜?|
| Tensor 鐩稿叧鍗曟祴鍏ㄩ儴閫氳繃 | `test_tensor.cc`锛堜袱濂楋級瑕嗙洊鏋勯€犮€佺储寮曘€乺eshape |
| 鑳藉疄鐜?ASCII tokenizer | `src/tokenizer.cc`锛宍Encode("Hi!")` 寰楀埌 BOS + 瀛楃鐮?|
| 鐞嗚В BPE 鍚堝苟閫昏緫 | `src/bpe_common.h` 鐨?`ApplyBpeMerges`锛宍hello` 鏀舵垚 1 涓?id |
| encode/decode roundtrip 姝ｇ‘ | ASCII 鍘绘帀 BOS 鍚庡洖鍒板師鏂囷紱BPE 鐨?`hello` 鐩存帴寰€杩?|

---

## 寤鸿闃呰椤哄簭

1. `tensor.h` 鈫?`tensor.cc`锛堝厛鎼炴噦鍐呭瓨甯冨眬锛?2. `matmul_optimized.cc` 閲岀殑 `MatmulNaive`锛堟墜鍐欎笁閲嶅惊鐜級
3. `ops.cc` 閲岀殑 `RmsNorm`銆乣Softmax`
4. `tests/test_ops.cc`锛堢敤鏂█鍙嶆帹鏈熸湜鏁板€硷級
5. 瀵圭収 `tests/gtest/test_ops.cc`锛屼綋浼氫袱绉嶆祴璇曢鏍煎樊寮?
