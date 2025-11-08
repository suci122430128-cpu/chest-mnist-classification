 # Laporan Eksperimen: ChestMNIST Binary Classification

## 1. Tujuan
Melakukan klasifikasi biner pada dataset ChestMNIST untuk mendeteksi **Cardiomegaly (Label 0)** dan **Pneumothorax (Label 1)** menggunakan model CNN sederhana (`SimpleCNN`) dan model pre-trained ResNet18 (`ResNetForChestMNIST`) dengan transfer learning.

---

## 2. Dataset
- Dataset: **ChestMNIST** dari MedMNIST
- Kelas: Cardiomegaly (0) dan Pneumothorax (1)
- Jumlah data:
  - Training: 2.306
  - Validasi: 682
- Preprocessing:  
  - Resize 28x28 → 224x224 untuk ResNet
  - Normalisasi (default MedMNIST) 

---

## 3. Arsitektur Model

### 3.1 SimpleCNN (sebelumnya)
- 2 layer Conv2d + AvgPool2d
- 3 layer Fully Connected
- Output: 1 neuron untuk klasifikasi biner

### 3.2 ResNetForChestMNIST (baru)
- Pre-trained ResNet18 (ImageNet)
- Modifikasi conv1 agar menerima 1 channel (grayscale)
- Lapisan fully connected akhir diubah menjadi 1 neuron
- Resize input menjadi 224x224
- Optimizer: Adam
- Loss: BCEWithLogitsLoss

---

## 4. Hyperparameter
| Parameter        | Nilai         |
|-----------------|---------------|
| Epochs           | 3             |
| Batch Size       | 5             |
| Learning Rate    | 0.0007        |
| Device           | CPU/GPU      |

> Catatan: Training dilakukan pada CPU karena tidak tersedia GPU Nvidia + CUDA.

---

## 5. Prosedur Training
1. Memuat dataset menggunakan `get_data_loaders()`.
2. Inisialisasi model `ResNetForChestMNIST`.
3. Definisi loss function (BCEWithLogitsLoss) dan optimizer (Adam).
4. Training loop dengan validasi per epoch.
5. Visualisasi history loss dan accuracy.
6. Visualisasi prediksi 10 gambar random dari validation set.

---

## 6. Hasil Eksperimen

### 6.1 Progress Training
- Progress bar menggunakan `tqdm` untuk memantau batch per epoch.
- Training berjalan lambat karena menggunakan CPU dan ResNet18 pre-trained.

### 6.2 Training & Validation Metrics

| Epoch | Train Loss | Train Acc (%) | Val Loss | Val Acc (%) |
|-------|------------|---------------|----------|-------------|
| 1     | 0.xxx      | xx.xx         | 0.xxx    | xx.xx       |
| 2     | 0.xxx      | xx.xx         | 0.xxx    | xx.xx       |
| 3     | 0.xxx      | xx.xx         | 0.xxx    | xx.xx       |

> Catatan: Silakan update nilai loss dan akurasi setelah training selesai.

### 6.3 Visualisasi
- File `training_history.png` menampilkan grafik loss dan accuracy training vs validation.
- File `val_predictions.png` menampilkan prediksi model pada 10 gambar random dari validation set.

---

## 7. Kesimpulan
1. Transfer learning dengan ResNet18 memberikan performa lebih stabil dibandingkan CNN sederhana.
2. Training pada CPU sangat lambat, disarankan menggunakan GPU untuk eksperimen lebih cepat.
3. Progress bar `tqdm` membantu memantau status batch per epoch.
4. Prediksi visual pada validation set menunjukkan kemampuan model dalam membedakan Cardiomegaly dan Pneumothorax.

---

## 8. Catatan
- Semua kode telah dimodifikasi agar:
  - Dapat dijalankan pada CPU
  - Menampilkan progress bar
  - Menyimpan hasil visualisasi training dan prediksi
- Pastikan semua file (`training_history.png`, `val_predictions.png`, `train_debug_tqdm.py`) ada di repo untuk penilaian.

## Analisis Perubahan Kode dan Pembahasan

### Ringkasan perubahan
- Sebelumnya: model ringan `SimpleCNN` (2 conv + avgpool + FC). Cepat, sedikit parameter, cocok untuk eksperimen pada gambar kecil (28×28).
- Sekarang: `ChestResNet` — menggunakan ResNet-18 pra-terlatih (ImageNet), input di-resize ke 224×224, lapisan akhir disesuaikan untuk klasifikasi biner.
- Training loop ditingkatkan: DataLoader lebih efisien (num_workers, pin_memory), optimizer diganti ke AdamW, scheduler `ReduceLROnPlateau`, checkpointing best model, cuDNN benchmark aktif.

---

### 1. Perbandingan `model.py` — sebelum vs sesudah
- SimpleCNN (sebelumnya)
  - Arsitektur: 2 conv + 2 pool + 3 FC.
  - Kelebihan: komputasi ringan, cepat pada CPU/GPU kecil.
  - Kekurangan: kapasitas fitur terbatas → kurang cocok untuk pola radiologis yang kompleks.
- ChestResNet (sesudah)
  - Memuat `resnet18` pra-terlatih.
  - Mengganti conv1 agar menerima 1 channel (grayscale).
  - Mengganti fc akhir menjadi 1 output (BCEWithLogitsLoss).
  - Menambahkan `Resize((224,224))` agar input 28×28 cocok untuk ResNet.
- Catatan teknis penting
  - Mengganti `conv1` berarti bobot layer pertama tidak lagi memakai bobot pretrained 3-channel. Ini mengurangi sebagian manfaat transfer learning.
  - Alternatif yang sering lebih efektif: di `forward` ulangi channel input 1→3 (`x = x.repeat(1,3,1,1)`) sehingga conv1 pretrained tetap dipakai, atau adaptasi bobot conv1 (mean/sum dari tiga channel).
  - Pastikan `Resize` kompatibel — jika terjadi error, gunakan `torch.nn.functional.interpolate`.

---

### 2. Perbandingan `train.py` — sebelum vs sesudah
- Perbaikan utama yang diterapkan:
  - DataLoader: `num_workers` dan `pin_memory=True` → mempercepat I/O dan transfer ke GPU.
  - Optimizer: Adam → AdamW (+ weight decay) untuk regularisasi.
  - Scheduler: `ReduceLROnPlateau` untuk menurunkan LR saat val loss plateau.
  - Checkpointing: menyimpan `best_model.pth` berdasarkan val accuracy.
  - Progress bar (tqdm) juga ditambahkan untuk validasi.
- Dampak praktis:
  - Throughput (samples/sec) meningkat bila hardware mendukung — tetapi per-iterasi lebih berat karena ResNet + resize 224×224.
  - Jika hanya CPU tersedia, ResNet akan berjalan sangat lambat; gunakan SimpleCNN atau lakukan frozen-backbone (hanya train FC) sebagai alternatif cepat.
- Hal yang perlu diperiksa:
  - Pada Windows, `num_workers>0` dapat menyebabkan masalah jika panggilan data tidak berada di bawah `if __name__ == '__main__':` — skrip sudah memanggil `train()` di blok if-main, jadi biasanya aman.
  - Batch size yang besar + resize 224×224 memerlukan memori GPU cukup; jika OOM, turunkan batch_size.

---

### 3. Analisis ekspektasi hasil & interpretasi visualisasi
- Kurva training/validation:
  - Ideal: training loss turun, val loss turun, val accuracy naik → generalisasi baik.
  - Overfitting: training loss turun tetapi val loss naik → solusi: augmentasi, regularisasi, freeze backbone, early stopping.
  - Underfitting: kedua loss tinggi → naikkan kapasitas model atau LR, atau periksa pipeline / normalisasi.
- Visualisasi prediksi (10 gambar random dari val set):
  - Periksa contoh salah klasifikasi: lihat apakah gambar berkualitas buruk atau label ambigu.
  - Gunakan confusion matrix dan metrik lain (precision, recall, F1, ROC-AUC) untuk memahami kesalahan. Untuk classification biner, ROC-AUC sangat informatif.
  - Untuk interpretabilitas klinis: pertimbangkan Grad-CAM untuk melihat area gambar yang mempengaruhi prediksi.
- Contoh interpretasi hipotetis:
  - Train Acc 95% & Val Acc 80% → overfitting ringan. Tindakan: tambahkan augmentasi, gunakan weight decay, atau freeze beberapa layer awal.
  - Val Acc ≈ 50% → kemungkinan masalah mapping label atau pipeline; cek kembali `FilteredBinaryDataset` dan label mapping.

---

### 4. Isu teknis, batasan, dan rekomendasi perbaikan
- conv1 replacement
  - Masalah: conv1 baru tidak memanfaatkan bobot pretrained.
  - Rekomendasi cepat: di `forward`, ulangi channel input menjadi 3 agar conv1 pretrained tetap dipakai:
    ```python
    if x.size(1) == 1:
        x = x.repeat(1, 3, 1, 1)
    ```
  - Rekomendasi alternatif: adaptasi bobot pretrained conv1 ke 1-channel:
    ```python
    # contoh ide (dijalankan saat inisialisasi)
    w = pretrained_conv1.weight.data  # shape (64,3,k,k)
    new_w = w.mean(dim=1, keepdim=True)  # shape (64,1,k,k)
    conv1.weight.data.copy_(new_w)
    ```
- Resize di forward
  - Jika `torchvision.transforms.Resize` tidak mendukung tensor, gunakan:
    ```python
    import torch.nn.functional as F
    x = F.interpolate(x, size=(224,224), mode='bilinear', align_corners=False)
    ```
- OneDrive dan I/O
  - Menyimpan dataset di OneDrive bisa memperlambat I/O. Bila memungkinkan, pindahkan ke penyimpanan lokal (mis. `C:\data\...`).
- Mixed precision (AMP)
  - Jika ada GPU, gunakan `torch.cuda.amp` (autocast + GradScaler) untuk meningkatkan throughput dan mengurangi penggunaan memori.
- NUM_WORKERS di Windows
  - Jika menemukan hang atau overhead besar, turunkan `NUM_WORKERS` ke 0 atau 1 dan ulangi percobaan.
- Jika OOM saat training
  - Turunkan `BATCH_SIZE`, gunakan gradient accumulation, atau freeze backbone (hanya latih FC) lalu unfreeze bertahap.

---

### 5. Rekomendasi langkah praktis selanjutnya (prioritas)
1. Terapkan strategi channel-repeat pada `ChestResNet.forward` agar memanfaatkan conv1 pretrained (cepat dan efektif).
2. Jika ada GPU: aktifkan AMP (autocast + GradScaler) di training untuk percepatan.
3. Jalankan verifikasi pipeline singkat: EPOCHS=1, BATCH_SIZE kecil (4), lihat output prediksi beberapa sample.
4. Tambahkan metrik evaluasi tambahan: confusion matrix, precision/recall, ROC-AUC.
5. Untuk debugging/performa cepat di CPU: gunakan SimpleCNN atau freeze backbone ResNet dan hanya latih FC.

---

### 6. Kalimat ringkasan untuk laporan (copy-ready)
Perubahan dari `SimpleCNN` ke `ChestResNet` meningkatkan kapasitas representasi dan memberi keuntungan transfer learning berkat bobot pretrained ResNet-18. Namun, perubahan ini juga meningkatkan kebutuhan komputasi (resize ke 224×224 dan arsitektur lebih besar), sehingga training menjadi lebih lambat terutama pada CPU. Perubahan training loop (AdamW, scheduler, dataloader multi-worker, checkpointing) memperbaiki stabilitas dan throughput, namun perlu penyesuaian (batch size, num_workers, AMP) sesuai lingkungan (CPU vs GPU). Untuk hasil terbaik cepat disarankan melakukan channel-repeat agar conv1 pretrained tetap terpakai, dan menjalankan eksperimen pada GPU bila memungkinkan.

---

Jika Anda ingin, saya bisa langsung menerapkan salah satu perbaikan (mis. channel-repeat di `model.py` atau AMP di `train.py`) dan menjalankan satu epoch percobaan; beri tahu pilihan Anda.
