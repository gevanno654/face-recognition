import cv2
import os
import numpy as np
import pandas as pd
from datetime import datetime
import time
import pickle
import warnings
warnings.filterwarnings('ignore')

class FaceRecognitionAttendance:
    def __init__(self):
        # Inisialisasi detektor wajah
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        if not os.path.exists(cascade_path):
            cascade_path = 'haarcascade_frontalface_default.xml'
        
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        
        if self.face_cascade.empty():
            print("Error: Haarcascade file tidak ditemukan!")
            return
        
        # Inisialisasi recognizer wajah dengan parameter khusus
        try:
            # Gunakan LBPH dengan parameter khusus untuk pencahayaan
            self.recognizer = cv2.face.LBPHFaceRecognizer_create(
                radius=2,  # Radius untuk LBP
                neighbors=16,  # Lebih banyak neighbors untuk lebih robust
                grid_x=8,  # Grid untuk normalisasi lokal
                grid_y=8,
                threshold=float('inf')  # Atur threshold secara dinamis
            )
        except AttributeError:
            print("Error: OpenCV-contrib tidak terinstal!")
            return
        
        # Data mahasiswa
        self.students = {}
        self.next_id = 0
        
        # File penyimpanan
        self.training_file = "face_trainer.yml"
        self.students_file = "students_data.pkl"
        self.attendance_file = "absensi_mahasiswa.csv"
        
        # Parameter adaptif
        self.base_confidence_threshold = 65  # Threshold dasar
        self.current_threshold = self.base_confidence_threshold
        self.min_face_size = 80
        
        # Histori pencahayaan untuk adaptasi
        self.lighting_history = []
        self.max_lighting_samples = 100
        
        # Load data
        self.load_students_data()
        self.load_training_data()
        self.initialize_attendance_file()

    def load_students_data(self):
        """Memuat data mahasiswa dari file"""
        if os.path.exists(self.students_file):
            with open(self.students_file, 'rb') as f:
                self.students = pickle.load(f)
            if self.students:
                self.next_id = max(self.students.keys()) + 1
            print(f"✓ Data {len(self.students)} mahasiswa berhasil dimuat!")
        else:
            print("Belum ada data mahasiswa.")

    def save_students_data(self):
        """Menyimpan data mahasiswa ke file"""
        with open(self.students_file, 'wb') as f:
            pickle.dump(self.students, f)

    def load_training_data(self):
        """Memuat data training dari file"""
        if os.path.exists(self.training_file):
            self.recognizer.read(self.training_file)
            print("✓ Data training berhasil dimuat!")
        else:
            print("Belum ada data training.")

    def initialize_attendance_file(self):
        """Menginisialisasi file absensi"""
        if not os.path.exists(self.attendance_file):
            df = pd.DataFrame(columns=["NIM", "Nama", "Waktu", "Status", "Confidence", "Lighting"])
            df.to_csv(self.attendance_file, index=False)

    def analyze_lighting_conditions(self, frame):
        """Analisis kondisi pencahayaan dari frame"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Hitung beberapa metrik pencahayaan
        brightness = np.mean(gray)
        contrast = np.std(gray)
        
        # Hitung histogram untuk analisis lebih lanjut
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist_normalized = hist / hist.sum()
        
        # Entropy dari histogram (ukuran variasi)
        entropy = -np.sum(hist_normalized * np.log2(hist_normalized + 1e-10))
        
        # Simpan histori pencahayaan
        lighting_info = {
            'brightness': brightness,
            'contrast': contrast,
            'entropy': entropy,
            'timestamp': datetime.now()
        }
        
        self.lighting_history.append(lighting_info)
        if len(self.lighting_history) > self.max_lighting_samples:
            self.lighting_history.pop(0)
        
        return lighting_info

    def adjust_threshold_based_on_lighting(self, lighting_info):
        """Sesuaikan threshold berdasarkan kondisi pencahayaan"""
        base_threshold = self.base_confidence_threshold
        
        # Adaptasi berdasarkan brightness
        brightness = lighting_info['brightness']
        if brightness < 50:  # Pencahayaan gelap
            adjustment = 15  # Lebih longgar
        elif brightness > 200:  # Pencahayaan sangat terang
            adjustment = 10  # Sedikit lebih longgar
        elif brightness > 150:  # Pencahayaan terang
            adjustment = 5
        else:  # Pencahayaan normal
            adjustment = 0
        
        # Adaptasi berdasarkan contrast
        contrast = lighting_info['contrast']
        if contrast < 20:  # Kontras rendah (pencahayaan buruk)
            adjustment += 10
        elif contrast > 60:  # Kontras tinggi (pencahayaan baik)
            adjustment -= 5
        
        # Batasi adjustment
        adjustment = max(-20, min(30, adjustment))
        
        self.current_threshold = base_threshold + adjustment
        return self.current_threshold

    def advanced_face_preprocessing(self, face_image, lighting_info):
        """Preprocessing wajah yang advanced untuk handling pencahayaan"""
        try:
            # 1. Normalize size
            face_image = cv2.resize(face_image, (200, 200))
            
            # 2. CLAHE (Contrast Limited Adaptive Histogram Equalization)
            # Lebih baik dari histogram equalization biasa
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            face_image = clahe.apply(face_image)
            
            # 3. Gamma correction berdasarkan brightness
            brightness = lighting_info['brightness']
            if brightness < 70:  # Terlalu gelap
                gamma = 0.7  # Brighten
                face_image = np.array(255 * (face_image / 255) ** gamma, dtype=np.uint8)
            elif brightness > 180:  # Terlalu terang
                gamma = 1.3  # Darken sedikit
                face_image = np.array(255 * (face_image / 255) ** gamma, dtype=np.uint8)
            
            # 4. Unsharp masking untuk meningkatkan detail
            gaussian = cv2.GaussianBlur(face_image, (0, 0), 2.0)
            face_image = cv2.addWeighted(face_image, 1.5, gaussian, -0.5, 0)
            
            # 5. Bilateral filter untuk mengurangi noise
            face_image = cv2.bilateralFilter(face_image, 9, 75, 75)
            
            # 6. Local Binary Pattern untuk illumination invariance
            # (LBP sudah digunakan oleh recognizer, tapi kita bisa enhance)
            
            return face_image
        except Exception as e:
            print(f"Warning in preprocessing: {e}")
            return cv2.resize(face_image, (200, 200))

    def augment_training_data(self, face_image):
        """Data augmentation untuk training - simulasi berbagai pencahayaan"""
        augmented_images = []
        
        # Original
        augmented_images.append(face_image)
        
        # Brightness variations
        for alpha in [0.7, 0.8, 0.9, 1.1, 1.2, 1.3]:
            bright_img = cv2.convertScaleAbs(face_image, alpha=alpha, beta=0)
            augmented_images.append(bright_img)
        
        # Contrast variations
        for contrast in [0.7, 0.8, 1.2, 1.3]:
            img_float = face_image.astype(np.float32)
            img_contrast = np.clip(img_float * contrast, 0, 255).astype(np.uint8)
            augmented_images.append(img_contrast)
        
        # Gamma correction variations
        for gamma in [0.6, 0.8, 1.2, 1.4]:
            table = np.array([((i / 255.0) ** gamma) * 255 for i in np.arange(0, 256)]).astype(np.uint8)
            gamma_img = cv2.LUT(face_image, table)
            augmented_images.append(gamma_img)
        
        # Gaussian blur variations (simulasi fokus berbeda)
        for ksize in [(3,3), (5,5)]:
            blur_img = cv2.GaussianBlur(face_image, ksize, 0)
            augmented_images.append(blur_img)
        
        return augmented_images

    def register_student_robust(self):
        """Pendaftaran dengan data augmentation untuk berbagai pencahayaan"""
        print("\n" + "="*50)
        print("PENDAFTARAN WAJAH - ROBUST UNTUK BERBAGAI PENCARCAHAYAAN")
        print("="*50)
        
        nim = input("Masukkan NIM mahasiswa: ").strip()
        name = input("Masukkan nama mahasiswa: ").strip()
        
        if not nim or not name:
            print("Error: NIM dan nama tidak boleh kosong!")
            return
        
        # Cek duplikasi
        for student_id, student_data in self.students.items():
            if student_data['nim'] == nim:
                print(f"Error: NIM {nim} sudah terdaftar!")
                return
        
        # Ambil sampel dengan berbagai pencahayaan
        print("\n📸 Silakan ambil posisi dengan pencahayaan berbeda-beda:")
        print("1. Hadap ke cahaya (jendela/lampu)")
        print("2. Posisi normal")
        print("3. Sedikit gelap")
        print("4. Sedikit silau")
        print("\nProgram akan mengambil sampel secara otomatis...")
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Tidak dapat mengakses kamera!")
            return
        
        face_samples = []
        count = 0
        max_samples = 40
        lighting_stages = ['terang', 'normal', 'gelap', 'silau']
        current_stage = 0
        
        print(f"\n🎯 Target: {max_samples} sampel dengan variasi pencahayaan")
        
        while count < max_samples:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Analisis pencahayaan
            lighting_info = self.analyze_lighting_conditions(frame)
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=7, minSize=(self.min_face_size, self.min_face_size)
            )
            
            for (x, y, w, h) in faces:
                if w < self.min_face_size:
                    continue
                
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                
                # Ambil dan preprocessing wajah
                face_roi = gray[y:y+h, x:x+w]
                processed_face = self.advanced_face_preprocessing(face_roi, lighting_info)
                
                # AUGMENTASI: Buat variasi dari satu wajah
                if count % 4 == 0:  # Setiap 4 sampel, lakukan augmentasi
                    augmented_faces = self.augment_training_data(processed_face)
                    face_samples.extend(augmented_faces)
                    count += len(augmented_faces)
                else:
                    face_samples.append(processed_face)
                    count += 1
                
                # Update stage setiap 10 sampel
                if count % 10 == 0:
                    current_stage = (current_stage + 1) % len(lighting_stages)
                
                # Tampilkan info
                cv2.putText(frame, f"Sampel: {count}/{max_samples}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Stage: {lighting_stages[current_stage]}", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                cv2.putText(frame, f"Brightness: {lighting_info['brightness']:.0f}", (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            cv2.imshow('Pendaftaran - Gerakkan kepala ke area pencahayaan berbeda', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        if len(face_samples) >= 30:  # Minimal 30 sampel (asli + augmented)
            # Simpan data mahasiswa
            self.students[self.next_id] = {"nim": nim, "name": name}
            
            # Training dengan semua sampel augmented
            labels = [self.next_id] * len(face_samples)
            
            print("\n⏳ Training model dengan data augmented...")
            if os.path.exists(self.training_file):
                self.recognizer.update(face_samples, np.array(labels))
            else:
                self.recognizer.train(face_samples, np.array(labels))
            
            self.recognizer.write(self.training_file)
            self.save_students_data()
            
            print(f"\n✅ {name} (NIM: {nim}) BERHASIL DIDAFTARKAN!")
            print(f"   Total sampel (asli+augmented): {len(face_samples)}")
            print(f"   ID: {self.next_id}")
            print(f"   Threshold adaptif akan digunakan")
            self.next_id += 1
        else:
            print("\n❌ Pendaftaran gagal! Sampel tidak cukup.")

    def take_attendance_adaptive(self):
        """Absensi dengan adaptive threshold berdasarkan pencahayaan"""
        if not self.students:
            print("Belum ada mahasiswa yang terdaftar!")
            return
        
        print("\n" + "="*50)
        print("ABSENSI ADAPTIF - OTOMATIS SESUAI PENCARCAHAYAAN")
        print("="*50)
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Tidak dapat mengakses kamera!")
            return
        
        attended_students = set()
        verification_count = {}
        frame_count = 0
        
        print("\n🔧 Sistem akan menyesuaikan threshold otomatis...")
        print("📊 Confidence threshold akan berubah sesuai pencahayaan")
        print("\nMemulai absensi adaptif...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # 1. ANALISIS PENCARCAHAYAAN
            lighting_info = self.analyze_lighting_conditions(frame)
            
            # 2. SESUAIKAN THRESHOLD OTOMATIS
            current_threshold = self.adjust_threshold_based_on_lighting(lighting_info)
            
            # 3. DETECT FACES
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=6, 
                minSize=(self.min_face_size, self.min_face_size)
            )
            
            current_detections = set()
            
            for (x, y, w, h) in faces:
                if w < self.min_face_size:
                    continue
                
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                
                # 4. PREPROCESSING DENGAN NORMALISASI PENCARCAHAYAAN
                face_roi = gray[y:y+h, x:x+w]
                processed_face = self.advanced_face_preprocessing(face_roi, lighting_info)
                
                # 5. PREDIKSI
                id_, confidence = self.recognizer.predict(processed_face)
                
                # 6. DECISION DENGAN THRESHOLD ADAPTIF
                if confidence < current_threshold and id_ in self.students:
                    student = self.students[id_]
                    student_key = student['nim']
                    current_detections.add(student_key)
                    
                    # Verification system
                    if student_key not in verification_count:
                        verification_count[student_key] = 0
                    verification_count[student_key] += 1
                    
                    # Tentukan warna dan status
                    if verification_count[student_key] >= 3:
                        color = (0, 255, 0)  # Hijau - terverifikasi
                        status_text = "TERVERIFIKASI"
                        
                        if student_key not in attended_students:
                            self.save_attendance_with_lighting(
                                student['nim'], student['name'], 
                                "Hadir", confidence, lighting_info
                            )
                            attended_students.add(student_key)
                            print(f"✓ {student['name']} - Confidence: {confidence:.1f} (Threshold: {current_threshold})")
                    else:
                        color = (0, 255, 255)  # Kuning - proses
                        status_text = f"VERIFIKASI {verification_count[student_key]}/3"
                    
                    # Tampilkan info di frame
                    cv2.putText(frame, f"{student['name']}", (x, y-60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    cv2.putText(frame, f"Conf: {confidence:.1f}", (x, y-35), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    cv2.putText(frame, f"Threshold: {current_threshold}", (x, y-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    
                else:
                    # Tidak dikenali
                    color = (0, 0, 255)
                    cv2.putText(frame, "Tidak Dikenal", (x, y-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    cv2.putText(frame, f"Conf: {confidence:.1f}", (x, y+h+20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Reset verification untuk wajah yang tidak terlihat
            for student_key in list(verification_count.keys()):
                if student_key not in current_detections:
                    verification_count[student_key] = 0
            
            # 7. TAMPILKAN INFORMASI PENCARCAHAYAAN
            info_y = 30
            # Info pencahayaan
            cv2.putText(frame, f"Pencahayaan: {lighting_info['brightness']:.0f}", 
                       (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            info_y += 25
            
            # Info threshold
            threshold_status = "NORMAL"
            if current_threshold > self.base_confidence_threshold + 5:
                threshold_status = "LONGGAR (cahaya kurang)"
            elif current_threshold < self.base_confidence_threshold - 5:
                threshold_status = "KETAT (cahaya baik)"
            
            cv2.putText(frame, f"Threshold: {current_threshold} ({threshold_status})", 
                       (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 0), 1)
            info_y += 20
            
            # Sudah absen
            if attended_students:
                cv2.putText(frame, f"Sudah absen: {len(attended_students)}", 
                           (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            cv2.imshow('Absensi Adaptif - Tekan Q untuk keluar', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Ringkasan
        print(f"\n{'='*40}")
        print("RINGKASAN ABSENSI")
        print(f"{'='*40}")
        print(f"Total hadir: {len(attended_students)}")
        if self.lighting_history:
            avg_brightness = np.mean([h['brightness'] for h in self.lighting_history[-10:]])
            print(f"Rata-rata pencahayaan: {avg_brightness:.0f}")
            print(f"Threshold rata-rata: {np.mean([h.get('threshold', self.base_confidence_threshold) for h in self.lighting_history[-10:]]):.0f}")

    def save_attendance_with_lighting(self, nim, name, status, confidence, lighting_info):
        """Simpan absensi dengan informasi pencahayaan"""
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        new_attendance = pd.DataFrame({
            "NIM": [nim],
            "Nama": [name],
            "Waktu": [current_time],
            "Status": [status],
            "Confidence": [f"{confidence:.1f}"],
            "Lighting": [f"{lighting_info['brightness']:.0f}"]
        })
        
        new_attendance.to_csv(self.attendance_file, mode='a', header=False, index=False)

    def view_attendance_with_stats(self):
        """Tampilkan absensi dengan statistik pencahayaan"""
        if os.path.exists(self.attendance_file):
            df = pd.read_csv(self.attendance_file)
            if not df.empty:
                print("\n" + "="*70)
                print("DATA ABSENSI DENGAN ANALISIS PENCARCAHAYAAN")
                print("="*70)
                
                # Tampilkan data
                print(df.to_string(index=False))
                
                # Statistik
                print(f"\n📊 STATISTIK:")
                print(f"Total records: {len(df)}")
                
                if 'Lighting' in df.columns:
                    try:
                        df['Lighting'] = pd.to_numeric(df['Lighting'], errors='coerce')
                        print(f"Pencahayaan rata-rata: {df['Lighting'].mean():.1f}")
                        print(f"Pencahayaan min: {df['Lighting'].min():.0f}")
                        print(f"Pencahayaan max: {df['Lighting'].max():.0f}")
                        
                        # Kategori pencahayaan
                        df['Lighting_Category'] = pd.cut(df['Lighting'], 
                            bins=[0, 50, 100, 150, 256],
                            labels=['Sangat Gelap', 'Gelap', 'Normal', 'Terang'])
                        
                        print("\n📈 Distribusi Pencahayaan:")
                        print(df['Lighting_Category'].value_counts())
                        
                    except:
                        print("(Data pencahayaan tidak tersedia)")
                
                # Hari ini
                today = datetime.now().strftime("%Y-%m-%d")
                today_attendance = df[df['Waktu'].str.contains(today)]
                if not today_attendance.empty:
                    print(f"\n📅 Absensi Hari Ini ({today}):")
                    print(f"Total hadir: {len(today_attendance)}")
                    
                    # Per mahasiswa hari ini
                    print("\nDetail per mahasiswa:")
                    for nim in today_attendance['NIM'].unique():
                        student_data = today_attendance[today_attendance['NIM'] == nim]
                        name = student_data['Nama'].iloc[0]
                        times = student_data['Waktu'].tolist()
                        print(f"  {name} ({nim}): {len(times)}x pada {', '.join([t.split()[1] for t in times])}")
            else:
                print("Belum ada data absensi.")
        else:
            print("File absensi tidak ditemukan.")

    def run(self):
        """Menu utama dengan sistem adaptif"""
        print("=" * 60)
        print("SISTEM ABSENSI WAJAH - ADAPTIF UNTUK SEMUA PENCARCAHAYAAN")
        print("=" * 60)
        print("Fitur:")
        print("  ✅ Adaptive threshold berdasarkan pencahayaan")
        print("  ✅ Data augmentation untuk berbagai kondisi cahaya")
        print("  ✅ Advanced preprocessing (CLAHE, Gamma, Unsharp)")
        print("  ✅ Verifikasi multi-frame untuk akurasi")
        print("  ✅ Analisis statistik pencahayaan")
        
        while True:
            print("\n" + "="*40)
            print("MENU UTAMA")
            print("="*40)
            print("1. Daftarkan Mahasiswa (ROBUST)")
            print("2. Ambil Absensi (ADAPTIF)")
            print("3. Lihat Data Absensi & Statistik")
            print("4. Lihat Data Mahasiswa")
            print("5. Test Pencahayaan Kamera")
            print("6. Atur Threshold Manual")
            print("7. Keluar")
            
            choice = input("\nPilih menu (1-7): ").strip()
            
            if choice == '1':
                self.register_student_robust()
            elif choice == '2':
                self.take_attendance_adaptive()
            elif choice == '3':
                self.view_attendance_with_stats()
            elif choice == '4':
                self.view_students()
            elif choice == '5':
                self.test_lighting_conditions()
            elif choice == '6':
                self.set_manual_threshold()
            elif choice == '7':
                print("\nTerima kasih! Sistem akan ditutup.")
                break
            else:
                print("Pilihan tidak valid!")

    def test_lighting_conditions(self):
        """Test real-time kondisi pencahayaan kamera"""
        print("\n🔦 TEST PENCARCAHAYAAN KAMERA")
        print("Arahkan kamera ke berbagai area untuk melihat nilai brightness")
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Kamera tidak dapat diakses!")
            return
        
        print("\nTekan 'q' untuk keluar dari test...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            lighting_info = self.analyze_lighting_conditions(frame)
            recommended_threshold = self.adjust_threshold_based_on_lighting(lighting_info)
            
            # Tampilkan info
            cv2.putText(frame, f"Brightness: {lighting_info['brightness']:.0f}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            cv2.putText(frame, f"Contrast: {lighting_info['contrast']:.1f}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            # Warna berdasarkan brightness
            if lighting_info['brightness'] < 50:
                color = (0, 0, 255)  # Merah - gelap
                status = "GELAP - Threshold akan dilonggarkan"
            elif lighting_info['brightness'] < 100:
                color = (0, 165, 255)  # Oranye - agak gelap
                status = "SEDIKIT GELAP"
            elif lighting_info['brightness'] < 150:
                color = (0, 255, 0)  # Hijau - normal
                status = "NORMAL - Optimal"
            else:
                color = (255, 255, 0)  # Cyan - terang
                status = "TERANG - Threshold akan dikencangkan"
            
            cv2.putText(frame, status, (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            cv2.putText(frame, f"Recommended Threshold: {recommended_threshold}", (10, 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow('Test Pencahayaan Kamera - Tekan Q untuk keluar', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        print("\nTest pencahayaan selesai!")

    def set_manual_threshold(self):
        """Atur threshold manual untuk kasus khusus"""
        print(f"\n🔧 Threshold saat ini: {self.base_confidence_threshold}")
        print("Rekomendasi:")
        print("  - 50-60: Sangat ketat (hanya pencahayaan sangat baik)")
        print("  - 60-70: Optimal untuk pencahayaan normal")
        print("  - 70-80: Longgar (untuk pencahayaan gelap)")
        print("  - 80+: Sangat longgar (hanya untuk testing)")
        
        try:
            new_threshold = int(input("\nMasukkan threshold baru (50-90): "))
            if 50 <= new_threshold <= 90:
                self.base_confidence_threshold = new_threshold
                print(f"✓ Threshold berhasil diubah menjadi {new_threshold}")
            else:
                print("❌ Threshold harus antara 50-90")
        except ValueError:
            print("❌ Input harus angka!")

    def view_students(self):
        """Tampilkan data mahasiswa"""
        if self.students:
            print("\n" + "="*50)
            print("DATA MAHASISWA TERDAFTAR")
            print("="*50)
            for student_id, data in self.students.items():
                print(f"ID: {student_id:3d} | NIM: {data['nim']:15s} | Nama: {data['name']}")
            print(f"\nTotal mahasiswa: {len(self.students)}")
        else:
            print("Belum ada mahasiswa yang terdaftar.")

# Main execution
if __name__ == "__main__":
    try:
        print("Memulai Sistem Absensi Wajah Adaptif...")
        system = FaceRecognitionAttendance()
        system.run()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        input("\nTekan Enter untuk keluar...")