# Investigasi provenance dataset dan scaler Min-Max

Tanggal investigasi: 2026-10-19. Tidak ada model yang dilatih ulang dan tidak ada file sumber/artefak yang ditimpa. Pemeriksaan dilakukan terhadap raw CSV, seluruh script Python yang tersedia, metadata/CSV hasil, artefak joblib, dan Git history.

## Kesimpulan utama

**Asal dataset dan split yang benar-benar dipakai untuk baseline tidak dapat dipastikan dari file yang tersedia.** Baseline metadata memberi rentang tanggal dan ukuran split, tetapi tidak menyimpan event ID, hash dataset, ataupun kode training. CSV prediksi baseline adalah inferensi atas 701 positive-duration event, bukan dataset/split training. Karena itu tidak dibuat tabel metrik before-after yang mengklaim observasi identik dan tidak ada training ulang.

Untuk Min-Max, kode saat ini mendefinisikan satu dataset secara eksplisit: `reconstruct_events()` di `minmax_scaling_experiment.py` membentuk grup STOP, membuang `duration_minutes <= 0`, menghitung fitur temporal sesudah filter, mengambil 411 event positif pertama menurut waktu, kemudian memotong 246/82/83. Ini merekonstruksi split sumber pada **kode saat ini**, tetapi extrema scaler dalam `.joblib` tidak cocok dengan fitur train tersebut. Maka kode tidak membuktikan bahwa artefak model tersimpan benar-benar di-fit pada data itu.

## Bukti cara pembentukan event dan fitur

1. `visualize.py` membangun 1.531 grup STOP dan ID `event_id` 1-1531. Fitur temporal dihitung pada semua grup (`load_events()`); sesudahnya `predict_durations()` mengambil baris dengan durasi positif untuk menulis `visualizations/duration_predictions.csv`. File itu berisi 701 baris positif. Ketiga kolom hitungan di CSV cocok persis dengan rekonstruksi ini; fitur menit cocok dalam toleransi pembulatan float saat serialisasi CSV.
2. `minmax_scaling_experiment.py` berbeda urutannya: setelah membuat grup, ia membuang event berdurasi nol, baru menghitung fitur temporal, lalu mengambil 411 baris pertama. Di split train pertama 246 baris, maksimum hitungan hasil rekonstruksi adalah **41 / 84 / 112**.
3. Menghitung fitur pada semua grup lalu mengambil 246 positive-duration event pertama menghasilkan maksimum berbeda, tetapi tetap bukan scaler tersimpan (lihat `feature_maxima_candidate_constructions.csv`). Dengan demikian filtering nol memang mengubah fitur hitungan, namun **tidak menjelaskan** maksimum 1/4/7 pada artefak scaler.
3. Metadata baseline classifier juga menyebut `max_internal_stop_gap_seconds = 300`, tetapi aturan itu tidak ada pada kode pembentuk event yang tersedia, dan metadata tidak menjelaskan apakah grup ditolak atau dipecah. Sebagai uji kandidat saja, menolak grup yang memiliki internal gap >300 detik memberi 1.529 grup/699 positif; memecah grup pada gap tersebut memberi 1.536 grup/706 positif. Pada kedua interpretasi, maksimum empat fitur di 246 positive-duration awal tetap tidak sama dengan scaler tersimpan. Rincian interpretasi dan nilai ada di `quality_rule_300s_candidate.csv`; tidak ada bukti bahwa model baseline benar-benar memakai salah satu interpretasi ini.
4. Memfilter data yang sama menurut periode baseline metadata juga tidak menghasilkan maksimum 1/4/7: train candidate positive-only berisi 245 baris, validation 81, test 82; fitur temporal tetap memiliki maksimum jauh di atas extrema scaler. Periode train candidate pada semua grup berisi 511 baris, bukan 246.

Detail maksimum per variasi pembuatan dan split ada di `feature_maxima_candidate_constructions.csv`; kandidat aturan gap 300 detik dirinci dalam `quality_rule_300s_candidate.csv`.

## Identitas event dan split

### Min-Max menurut script saat ini

Dengan mencocokkan `start_time` ke event ID yang dibuat `visualize.py`, script Min-Max menghasilkan:

- Train: 246 event, IDs mulai-akhir 1-561, waktu 2026-04-29 15:21:48 sampai 2026-06-26 00:00:15.
- Validation: 82 event, IDs 568-813, waktu 2026-06-26 00:08:46 sampai 2026-07-07 20:52:20.
- Test: 83 event, IDs 823-990, waktu 2026-07-08 06:00:15 sampai 2026-07-09 23:42:32.

ID rentang tidak kontigu karena `event_id` mengikuti seluruh grup STOP termasuk yang durasinya nol, sedangkan script Min-Max menghapus nol. Identitas di atas adalah rekonstruksi deterministik kode dan raw CSV saat ini; model joblib tidak menyimpan ID train.

### Baseline menurut metadata dan file saat ini

Metadata classifier menyatakan train 2026-07-01 09:42:14-2026-08-05 08:11:09, validation 2026-08-05 08:29:44-2026-08-23 14:32:23, test 2026-08-23 15:25:39-2026-08-31 06:46:09. Regresi menyatakan ukuran 246/82/83 dan split kronologis, tetapi tidak mencatat periode. Jika periode classifier itu diterapkan ke event reconstruction `visualize.py`, kandidat positive-duration berjumlah 245/81/82, sedangkan jumlah seluruh grupnya 511/154/175. Jadi periode metadata tidak mengidentifikasi tepat 246/82/83 baris positif pada CSV tersedia. Kandidat IDs dan tanggal lengkap ada di `split_identity_evidence.csv` dan bukan bukti membership aktual model baseline.

Dalam rekonstruksi ID yang sama, 35 kandidat event Min-Max validation dan 83 kandidat event Min-Max test jatuh di dalam periode yang metadata baseline sebut training. Ini adalah overlap pada event reconstruction saat ini, bukan bukti bahwa baseline benar-benar memakai event tersebut; membership baseline aktual tidak terverifikasi. Matriks intersection tersedia di `split_identity_evidence.csv` dan menunjukkan asumsi pembagian identik tidak didukung metadata periode.

## Mengapa scaler menyimpan 1/4/7?

Scaler standalone dan scaler di kedua pipeline Min-Max identik pada semua extrema, urutan fitur, rentang scaler `(0,1)`, dan jumlah sampel `n_samples_seen_ = 246`. Nilai yang tersimpan adalah `data_min_ = [0,0,0,0]` dan `data_max_ = [18048.1667, 1, 4, 7]`. Sebaliknya, fit scaler baru **hanya untuk pemeriksaan** pada train yang dihasilkan fungsi Min-Max saat ini (tanpa model training) menghasilkan maksimum `[18048.1667, 41, 84, 112]`. Keduanya cocok pada fitur pertama, berbeda pada tiga hitungan. Angka detail dan perbandingan tiap pipeline terdapat di `scaler_provenance_comparison.csv`.

Kode saat ini memuat `MinMaxScaler(feature_range=(0,1))`, fit pipeline atas `X_train`, dan metadata menyebut scaler fit pada training. Namun artefak tidak konsisten dengan `X_train` versi kode+CSV yang ada. Filter sebelum/sesudah fitur sudah diuji sebagai kandidat dan tidak menghasilkan 1/4/7. Perbedaan dataset/fitur saat artefak dibuat, versi preprocessing yang tidak tersimpan, atau artefak scaler yang stale/tercampur tetap mungkin; file tidak memberikan bukti untuk memilih satu sebab. Git commit `9b85301` memasukkan script Min-Max dan artefak joblib/metadata/summary bersama, sementara raw CSV serta `visualize.py` tidak berubah dalam commit itu; tidak ada log run atau fingerprint data.

## Baseline model dan validitas metrik

- Baseline joblib yang tersedia adalah LogisticRegression dan LinearRegression langsung, tanpa pipeline scaler. `visualize.py` memakai model regresi baseline untuk memprediksi seluruh event yang direkonstruksi; prediksi pada `duration_predictions.csv` sama dengan inferensi artefak pada kolom fitur CSV. Ini membuktikan jalur ekspor inferensi, bukan dataset training atau split evaluasi.
- Baseline classifier/regressor metadata berisi metrik, fitur, threshold klasifikasi 0.30, target regresi `log1p`/inverse `expm1`, ukuran split dan (classifier) periode. Tidak ditemukan baseline training script, notebook `.ipynb`, file split event, atau evaluasi per-event.
- Min-Max metadata berisi ukuran 411 dan ukuran split, tetapi tidak menyertakan tanggal atau ID event; scaler artifacts menunjukkan 246 sampel tetapi extrema-nya tidak cocok dengan rekonstruksi script sekarang.
- `visualizations/minmax_scaling_experiment_summary.csv` menyalin metrik baseline sebagai konstanta di source Min-Max script; file ini bukan bukti evaluasi ulang pada split yang sama.

Karena keanggotaan baseline tidak dapat dipasangkan ke event IDs Min-Max, split yang identik **tidak terverifikasi**. Selain itu, split menurut kode Min-Max (April-Juli) tidak sama dengan periode classifier baseline (Juli-Agustus), dan scaler tersimpan tidak cocok dengan data train yang kode sekarang hasilkan. Jadi perbandingan metrik saat ini **tidak valid untuk mengukur efek Min-Max Scaling**. Nilai metadata aslinya dicatat tanpa penggantian di `recorded_metrics_validity.csv`, semua baris ditandai tidak terverifikasi untuk observasi identik.

## Tindakan yang diperlukan sebelum klaim eksperimen

Temukan atau pulihkan artefak preprocessing/training asal yang mencatat minimal data/split event IDs atau hash, aturan pembentukan fitur, serta scaler fit. Cocokkan artefak tersebut dengan model dan metadata. Jika artefak tidak ada, eksperimen harus dijalankan ulang dari dataset/split baseline yang disahkan; itu belum dilakukan pada investigasi ini karena provenance belum dapat dipastikan.
