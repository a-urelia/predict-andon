# Laporan eksperimen Min-Max Scaling

Dibuat dari file proyek yang tersedia. Dataset, model, metadata, dan script sumber tidak ditimpa; tidak ada model yang dilatih ulang.

## Sumber dan susunan data

- Data mentah: `data/mc09_andon.csv` (893.901 baris).
- Script: `minmax_scaling_experiment.py`. Tidak ditemukan CSV preprocessing berisi 411 baris atau CSV hasil scaling yang tersimpan.
- Dataset before laporan direkonstruksi memakai fungsi `reconstruct_events()` di script: kelompok STOP, durasi positif, diurutkan kronologis, lalu diambil 411 event pertama. Seluruh event dicocokkan ke `event_id` di `visualizations/duration_predictions.csv` lewat `start_time` unik.
- Perbedaan preprocessing: script Min-Max membuang `duration_minutes <= 0` sebelum menghitung fitur temporal. Script visualisasi menghitung fitur pada seluruh event, baru menyaring event positif. Karena itu nilai fitur dari CSV visualisasi berbeda dan tidak dipakai sebagai dataset before eksperimen.

## Variabel, split, dan scaling

- Fitur input: `minutes_since_previous_STOP`, `STOP_count_last_1min`, `STOP_count_last_5min`, `STOP_count_last_10min`.
- Target klasifikasi: `prolonged_STOP = duration_minutes > 5`; batas kelas 5 menit; threshold keputusan Logistic Regression 0.30.
- Target regresi `duration_minutes` tetap dalam menit dan tidak di-scale. Script menerapkan `log1p` pada target training dan `expm1` pada prediksi.
- Split kronologis script: train 246 (2026-04-29 15:21:48 - 2026-06-26 00:00:15), validation 82 (2026-06-26 00:08:46 - 2026-07-07 20:52:20), test 83 (2026-07-08 06:00:15 - 2026-07-09 23:42:32).
- Formula: `X_scaled = (X - X_min) / (X_max - X_min)`, rentang `[0,1]`. Script mem-fit scaler pada `X_train`, lalu menggunakannya untuk transformasi train, validation, dan test.
- Karena file hasil scaled tidak tersedia dan scaler `.joblib` tersimpan tidak cocok dengan rekonstruksi input saat ini, `dataset_after_minmax.csv` dibuat dengan menjalankan konfigurasi scaling script di memori: `MinMaxScaler(feature_range=(0,1))`, fit pada 246 baris train saja; tidak ada model yang di-fit ulang. `scaler_parameters.csv` menampilkan parameter scaler laporan ini dan extrema scaler model tersimpan berdampingan.
- **Ketidaksesuaian scaler:** maksimum fitur training dari data rekonstruksi adalah 18.048,1667 / 41 / 84 / 112. Scaler model tersimpan mencatat `data_max_` 18.048,1667 / 1 / 4 / 7. Karena itu scaler tersimpan (meskipun `n_samples_seen_=246` dan kedua pipeline berbagi scaler identik) tidak cocok dengan train split yang dibangun script saat ini. Klaim metadata bahwa scaler fit pada training tidak cukup untuk membuktikan provenance dataset yang sama.
- Untuk scaling laporan ini tidak tampak leakage: fit hanya train dan validation/test hanya ditransformasi. Fitur temporal dihitung kronologis berdasarkan event sebelumnya, bukan masa depan. Namun, perbedaan preprocessing dan provenance scaler/model harus dibereskan sebelum menyatakan artefak eksperimen konsisten.
- Min-Max Scaling mengubah rentang nilai fitur, bukan secara langsung menghilangkan skewness. Konfigurasi `clip=False` memperbolehkan nilai validation/test di luar rentang train keluar dari `[0,1]`.

## Output data dan statistik

`dataset_before_minmax.csv` dan `dataset_after_minmax.csv` mempertahankan identitas event, waktu, split, target, dan urutan kolom; hanya empat fitur input di-scale. `statistics_before_after.csv` menyajikan minimum, maksimum, mean, median, standard deviation sampel (ddof=1), jumlah unik, dan missing per fitur/split/sebelum-sesudah. Sepuluh baris pasangan event yang sama berdasarkan `event_id` ada di `ten_row_before_after_examples.csv`.

## Evaluasi model dan ketidaksesuaian

`model_metrics_comparison.csv` memuat nilai baseline dan Min-Max sebagaimana tertulis pada metadata; nilai tidak tersedia dibiarkan kosong. Kolom tambahan memuat inferensi tanpa training ulang atas model dan scaler `.joblib` yang sekarang tersimpan, jadi jangan disamakan dengan hasil eksperimen tercatat.

Perbandingan baseline vs Min-Max **tidak setara**: metadata baseline mencatat periode train 2026-07-01 - 2026-08-05, validation 2026-08-05 - 2026-08-23, test 2026-08-23 - 2026-08-31. Script Min-Max mengambil 411 event positif paling awal, dengan test berakhir 2026-07-09. Ukuran split 246/82/83 sama tidak berarti observasi yang dibandingkan sama.

Selain itu, 10 dari 20 metrik metadata Min-Max berbeda dari inferensi ulang artefak `.joblib` yang tersimpan. Scaler artefak juga tidak cocok dengan rentang fitur train hasil rekonstruksi saat ini, sehingga inferensi tersebut hanya ditampilkan sebagai diagnostik provenance, bukan evaluasi valid atas dataset laporan.

`regression_actual_prediction_summary.csv` menyertakan mean dan standard deviation populasi target/prediksi saat tersedia. Statistik baseline berasal dari prediksi di `duration_predictions.csv` dalam periode metadata, tetapi jumlah baris (245/81/82) berbeda dari ukuran split metadata (246/82/83); ketidaksesuaian ditandai. Statistik Min-Max adalah inferensi diagnostik pada model tersimpan, bukan retraining. MAPE dapat tinggi ketika target aktual mendekati nol.

## Angka test yang tercatat di metadata (bukan klaim dampak scaling)

- Klasifikasi Logistic Regression: baseline accuracy 0.7349, balanced accuracy 0.7103, precision 0.4286, recall 0.6667, F1 0.5217, ROC-AUC 0.7449, PR-AUC 0.4165. Metadata Min-Max: accuracy 0.8675, balanced accuracy 0.5000, precision 0.0000, recall 0.0000, F1 0.0000, ROC-AUC 0.7626, PR-AUC 0.2814.
- Regresi Log-Linear: baseline MAE 37.9024, RMSE 223.7904, MAPE 357.4918%; metadata Min-Max MAE 7.5590, RMSE 40.0408, MAPE 641.3457%.

**Kesimpulan:** file laporan before-after kini konsisten dengan formula dan split yang dikodekan di script saat ini. Namun, hasil metrik maupun model/scaler tersimpan tidak cukup konsisten dengan split/data yang direkonstruksi untuk menyatakan bahwa perubahan performa disebabkan oleh Min-Max Scaling. Perlu konfirmasi versi preprocessing dan artefak asal sebelum menarik kesimpulan skripsi.
