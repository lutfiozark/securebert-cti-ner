# CTI Named Entity Recognition (APTNER)

RoBERTa tabanlı bir NER sistemi ile STIX 2.1 uyumlu **21 siber tehdit istihbaratı (CTI) varlık türünü** tespit eder. APTNER veri kümesi üzerinde **RoBERTa + BiGRU + CRF** mimarisiyle eğitildi; softmax başlık da desteklenir. Gradio tabanlı bir demo ile etkileşimli etiketleme yapılabilir.

---

## 1) Genel Bakış
- **Model:** RoBERTa-base + BiGRU + CRF (isteğe bağlı softmax head)
- **Alan:** Cyber Threat Intelligence (CTI)
- **Varlık sayısı:** 21 STIX 2.1 tabanlı etiket
- **Performans (APTNER test):** micro F1 ≈ 0.96, macro F1 ≈ 0.93
- **Demo:** Eğitilmiş softmax checkpoint ile Gradio arayüzü

## 2) Hızlı Başlangıç
- Gereksinimler: Python 3.10+, `pip install -r requirements.txt`
- Veri: `data/` altına `APTNERtrain.txt`, `APTNERdev.txt`, `APTNERtest.txt` (lisans/yeniden dağıtım koşullarını kontrol edin; git’e ekli değil)
- Eğitim (softmax varsayılan): `python -m src.train`
- CRF head: `python -m src.train --head crf`
- Değerlendirme: `python -m src.evaluate --checkpoint cti-ner-softmax-best --split test`
- Demo: `python demo_gradio.py` (varsayılan checkpoint `cti-ner-softmax-best/`; farklı konum için `CTI_NER_CHECKPOINT` ortam değişkeni)
- Opsiyonel DAPT: `python run_dapt.py`

## 3) APTNER Veri Kümesi
APTNER, CTI raporlarından türetilmiş ve BIOES formatında etiketlenmiş 21 varlık türü içerir.

### 3.1 İstatistikler
| Özellik          | Değer     |
| ---------------- | --------- |
| Cümle sayısı     | 10,984    |
| Token sayısı     | 260,134   |
| Varlık sayısı    | 39,565    |
| Varlık türü      | 21        |
| Etiket formatı   | BIOES     |
| Dil              | İngilizce |
| Alan             | CTI       |

### 3.2 Örnek etiketler
| Etiket  | Açıklama                         | Örnek                |
| ------- | -------------------------------- | -------------------- |
| APT     | Gelişmiş tehdit grubu            | APT28                |
| SECTEAM | Güvenlik ekibi/kurumu            | bir araştırma grubu  |
| VULID   | Zafiyet kimliği                  | CVE-2021-26855       |
| VULNAME | Zafiyet adı                      | Heartbleed           |
| MAL     | Zararlı yazılım adı              | WannaCry             |
| TOOL    | Saldırı / hack aracı             | Mimikatz             |
| IP      | IP adresi                        | 192.168.0.1          |
| DOM     | Domain                           | malicious.com        |
| URL     | URL                              | http://example.com   |
| EMAIL   | E-posta                          | user@example.com     |
| HASH    | MD5/SHA1/SHA2                    | e99a18…              |
| ACT     | Atak tekniği                     | spear phishing       |
| IDTY    | Kimlik bilgisi                   | username             |
| OS      | İşletim sistemi                  | Windows 10           |
| PROT    | Ağ protokolü                     | HTTP                 |

## 4) Mimari ve Eğitim
- **Akış:** RoBERTa-base → BiGRU → CRF (veya softmax)
- **Tokenizer:** BPE; alt parçalara BIOES uyumlu etiket hizalaması
- **Kayıp:** CRF için negatif log-olabilirlik; softmax için cross-entropy
- **Eğitim:** Hugging Face Trainer; early stopping; değerlendirme F1

### 4.1 Hiperparametreler
| Parametre              | Değer           |
| ---------------------- | --------------- |
| Optimizasyon           | AdamW           |
| Öğrenme hızı           | 5e-5 (CRF: 1e-6 önerilir) |
| Epoch                  | ≤10 (genelde 4–5’te durur) |
| Batch                  | 32              |
| Dropout                | 0.1             |
| Weight decay           | 0.01            |
| Maks. sekans uzunluğu  | 256 (eval 512)  |
| Early stopping         | F1 skoruna göre |

## 5) Sonuçlar (APTNER test)

### 5.1 Sınıf bazlı P/R/F1
| Etiket  | P    | R    | F1   |
| ------- | ---- | ---- | ---- |
| APT     | 0.90 | 0.88 | 0.89 |
| SECTEAM | 0.92 | 0.89 | 0.90 |
| LOC     | 0.95 | 0.94 | 0.94 |
| TIME    | 0.93 | 0.92 | 0.92 |
| VULNAME | 0.88 | 0.86 | 0.87 |
| VULID   | 0.99 | 0.99 | 0.99 |
| TOOL    | 0.91 | 0.92 | 0.92 |
| MAL     | 0.90 | 0.91 | 0.90 |
| FILE    | 0.94 | 0.93 | 0.93 |
| MD5     | 0.99 | 0.98 | 0.98 |
| SHA1    | 0.98 | 0.99 | 0.99 |
| SHA2    | 0.99 | 0.99 | 0.99 |
| IDTY    | 0.85 | 0.84 | 0.85 |
| ACT     | 0.81 | 0.79 | 0.80 |
| DOM     | 0.96 | 0.97 | 0.96 |
| ENCR    | 0.95 | 0.93 | 0.94 |
| EMAIL   | 0.97 | 0.98 | 0.97 |
| OS      | 0.96 | 0.95 | 0.95 |
| PROT    | 0.98 | 0.97 | 0.98 |
| URL     | 0.96 | 0.95 | 0.95 |
| IP      | 0.99 | 0.99 | 0.99 |

### 5.2 Özet metrikler
| Ortalama | Precision | Recall | F1   |
| -------- | --------- | ------ | ---- |
| Micro    | 0.96      | 0.95   | 0.96 |
| Macro    | 0.93      | 0.92   | 0.93 |

**CRF etkisi:** CRF’siz softmax modele kıyasla ≈2 F1 puanı kazanım (etiket tutarlılığı).

## 6) Depo Yapısı
- `src/train.py` — eğitim döngüsü (softmax veya CRF)
- `src/evaluate.py` — kayıtlı checkpoint değerlendirme
- `src/preprocess.py`, `src/data_loader.py`, `src/utils.py` — veri yükleme/etiket hizalama yardımcıları
- `demo_gradio.py` — eğitilmiş softmax modeliyle Gradio demo
- `run_dapt.py` — CTI tweet’lerinde DAPT örneği
- `cti-ner-softmax*` — checkpoint klasörleri (git tarafından yok sayılır)

## 7) Paylaşım Notları
- Büyük modeller ve optimizasyon dosyaları git dışı; gerekiyorsa Git LFS kullanın.
- `data/`, `.gradio/`, `eval_tmp/`, `__pycache__/` ve geçici dosyalar `.gitignore`’da.
- Veri seti yeniden dağıtımı için APTNER makalesi ve lisans koşullarını doğrulayın.

## 8) Alıntı
APTNER: Xuren Wang, Songheng He, Zihan Xiong, Xinxin Wei, Zhangwei Jiang, Sihan Chen, Jun Jiang. “APTNER: A Specific Dataset for NER Missions in Cyber Threat Intelligence Field.” CSCWD 2022.
