import cv2
import os
import numpy as np
import pandas as pd
from datetime import datetime
import time
import pickle

class FaceRecognitionAttendance:
    def __init__(self):
        # Inisialisasi detektor wajah
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        if not os.path.exists(cascade_path):
            cascade_path = 'haarcascade_frontalface_default.xml'
        
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        
        # Cek apakah cascade classifier berhasil dimuat
        if self.face_cascade.empty():
            print("Error: Haarcascade file tidak ditemukan!")
            print("Pastikan file 'haarcascade_frontalface_default.xml' ada di folder yang sama")
            return
        
        # Inisialisasi recognizer wajah
        try:
            self.recognizer = cv2.face.LBPHFaceRecognizer_create()
            # Atur parameter recognizer untuk lebih ketat
            self.recognizer.setThreshold(80)  # Default adalah 80, kita buat lebih ketat
        except AttributeError:
            print("Error: OpenCV-contrib tidak terinstal dengan benar!")
            print("Jalankan: pip install opencv-contrib-python")
            return
        
        # Dictionary untuk menyimpan data mahasiswa
        self.students = {}
        self.next_id = 0
        
        # File untuk menyimpan data
        self.training_file = "face_trainer.yml"
        self.students_file = "students_data.pkl"
        self.attendance_file = "absensi_mahasiswa.csv"
        
        # Threshold untuk confidence
        self.confidence_threshold = 30  # Lebih ketat dari sebelumnya (70)
        self.min_face_size = 100  # Minimum ukuran wajah dalam pixels
        
        # Load data yang ada
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
            print(f"Data {len(self.students)} mahasiswa berhasil dimuat!")
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
            print("Data training berhasil dimuat!")
        else:
            print("Belum ada data training. Silakan daftarkan wajah mahasiswa terlebih dahulu.")

    def initialize_attendance_file(self):
        """Menginisialisasi file absensi jika belum ada"""
        if not os.path.exists(self.attendance_file):
            df = pd.DataFrame(columns=["NIM", "Nama", "Waktu", "Status", "Confidence"])
            df.to_csv(self.attendance_file, index=False)
            print("File absensi dibuat.")

    def preprocess_face(self, face_image):
        """Preprocessing wajah untuk meningkatkan akurasi"""
        try:
            # Resize ke ukuran konsisten
            face_image = cv2.resize(face_image, (200, 200))
            
            # Equalize histogram untuk meningkatkan kontras
            face_image = cv2.equalizeHist(face_image)
            
            # Gaussian blur untuk mengurangi noise
            face_image = cv2.GaussianBlur(face_image, (5, 5), 0)
            
            return face_image
        except:
            return face_image

    def register_student(self):
        """Fungsi untuk mendaftarkan wajah mahasiswa baru dengan kualitas lebih baik"""
        print("\n=== PENDAFTARAN WAJAH MAHASISWA ===")
        nim = input("Masukkan NIM mahasiswa: ").strip()
        name = input("Masukkan nama mahasiswa: ").strip()
        
        if not nim or not name:
            print("Error: NIM dan nama tidak boleh kosong!")
            return
        
        # Cek apakah NIM sudah terdaftar
        for student_id, student_data in self.students.items():
            if student_data['nim'] == nim:
                print(f"Error: NIM {nim} sudah terdaftar atas nama {student_data['name']}!")
                return
        
        # Inisialisasi webcam
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Tidak dapat mengakses kamera!")
            return
        
        face_samples = []
        count = 0
        max_samples = 50  # Increased samples for better training
        skipped_faces = 0
        
        print("\nPanduan Pendaftaran:")
        print("- Pastikan wajah terlihat jelas di kamera")
        print("- Pastikan pencahayaan cukup dan merata")
        print("- Posisikan wajah di tengah frame")
        print("- Jaga ekspresi wajah netral")
        print("- Hindari background yang ramai")
        print("- Tekan 'q' untuk menyelesaikan pendaftaran lebih awal")
        print(f"- Program akan mengambil {max_samples} sampel wajah berkualitas")
        
        print("\nMengambil sampel wajah...")
        
        while count < max_samples:
            ret, frame = cap.read()
            if not ret:
                print("Error: Tidak dapat membaca frame dari kamera!")
                break
                
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Deteksi wajah dengan parameter yang lebih ketat
            faces = self.face_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.1, 
                minNeighbors=8,  # Increased for better accuracy
                minSize=(self.min_face_size, self.min_face_size),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            for (x, y, w, h) in faces:
                # Cek kualitas wajah
                if w < self.min_face_size or h < self.min_face_size:
                    skipped_faces += 1
                    continue
                
                # Cek aspect ratio wajah (harus mendekati 1:1)
                aspect_ratio = w / h
                if aspect_ratio < 0.7 or aspect_ratio > 1.3:
                    skipped_faces += 1
                    continue
                
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                
                # Simpan region wajah dengan preprocessing
                face_roi = gray[y:y+h, x:x+w]
                processed_face = self.preprocess_face(face_roi)
                face_samples.append(processed_face)
                count += 1
                
                # Tampilkan count di frame
                cv2.putText(frame, f"Sampel: {count}/{max_samples}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Nama: {name}", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Skipped: {skipped_faces}", (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                cv2.putText(frame, "Tekan 'q' untuk keluar", (10, 120), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            
            cv2.imshow('Pendaftaran Wajah - Pastikan Wajah Jelas', frame)
            
            # Tekan 'q' untuk keluar lebih awal
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        if len(face_samples) >= 20:  # Minimal 20 sampel berkualitas
            # Simpan data mahasiswa
            self.students[self.next_id] = {"nim": nim, "name": name}
            
            # Training recognizer
            labels = [self.next_id] * len(face_samples)
            
            print("Melakukan training model...")
            # Jika sudah ada data training sebelumnya, gunakan update
            if os.path.exists(self.training_file):
                self.recognizer.update(face_samples, np.array(labels))
            else:
                self.recognizer.train(face_samples, np.array(labels))
            
            self.recognizer.write(self.training_file)
            self.save_students_data()
            
            print(f"\n✓ Mahasiswa {name} (NIM: {nim}) berhasil didaftarkan!")
            print(f"✓ {len(face_samples)} sampel wajah berkualitas berhasil diambil")
            print(f"✓ {skipped_faces} wajah tidak memenuhi syarat kualitas")
            print(f"✓ ID Mahasiswa: {self.next_id}")
            self.next_id += 1
        else:
            print(f"\n✗ Pendaftaran gagal! Hanya {len(face_samples)} sampel yang memenuhi syarat (minimal 20).")
            print("Penyebab mungkin:")
            print("- Pencahayaan kurang baik")
            print("- Wajah terlalu kecil/jauh dari kamera")
            print("- Background terlalu ramai")
            print("- Ekspresi wajah berubah-ubah")

    def take_attendance(self):
        """Fungsi untuk mengambil absensi dengan verifikasi ganda"""
        if not self.students:
            print("Belum ada mahasiswa yang terdaftar! Silakan daftarkan mahasiswa terlebih dahulu.")
            return
            
        print("\n=== PENGAMBILAN ABSENSI ===")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Tidak dapat mengakses kamera!")
            return
        
        # Dictionary untuk melacak mahasiswa yang sudah diabsen
        attended_students = set()
        # Dictionary untuk verifikasi ganda (harus terdeteksi beberapa kali)
        verification_count = {}
        
        print("\nPanduan Pengambilan Absensi:")
        print("- Arahkan wajah ke kamera dengan jelas")
        print("- Tunggu hingga nama terdeteksi dengan confidence tinggi")
        print("- Sistem membutuhkan verifikasi berulang untuk mencegah false positive")
        print("- Absensi akan tercatat otomatis setelah verifikasi")
        print("- Tekan 'q' untuk keluar")
        
        print("\nMemulai pengambilan absensi...")
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Tidak dapat membaca frame dari kamera!")
                break
                
            frame_count += 1
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Deteksi wajah dengan parameter ketat
            faces = self.face_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.1, 
                minNeighbors=8,
                minSize=(self.min_face_size, self.min_face_size),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            current_detections = set()
            
            for (x, y, w, h) in faces:
                # Hanya proses wajah dengan ukuran memadai
                if w < self.min_face_size or h < self.min_face_size:
                    continue
                
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                
                # Prediksi wajah dengan preprocessing
                face_roi = gray[y:y+h, x:x+w]
                processed_face = self.preprocess_face(face_roi)
                id_, confidence = self.recognizer.predict(processed_face)
                
                # Threshold yang lebih ketat dan verifikasi ganda
                if confidence < self.confidence_threshold and id_ in self.students:
                    student = self.students[id_]
                    student_key = student['nim']
                    current_detections.add(student_key)
                    
                    # Update verification count
                    if student_key not in verification_count:
                        verification_count[student_key] = 0
                    verification_count[student_key] += 1
                    
                    # Tentukan warna berdasarkan tingkat verifikasi
                    if verification_count[student_key] >= 3:  # Terdeteksi 3 kali
                        color = (0, 255, 0)  # Hijau - terverifikasi
                        status_text = "TERVERIFIKASI"
                        
                        # Simpan absensi jika belum tercatat dan sudah terverifikasi
                        if student_key not in attended_students:
                            self.save_attendance(student['nim'], student['name'], "Hadir", confidence)
                            attended_students.add(student_key)
                            print(f"✓ {student['name']} ({student['nim']}) - Hadir (Confidence: {confidence:.1f})")
                    else:
                        color = (0, 255, 255)  # Kuning - dalam proses verifikasi
                        status_text = f"VERIFIKASI {verification_count[student_key]}/3"
                    
                    # Tampilkan informasi di frame
                    cv2.putText(frame, f"{student['name']}", (x, y-60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    cv2.putText(frame, f"NIM: {student['nim']}", (x, y-35), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    cv2.putText(frame, status_text, (x, y-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    cv2.putText(frame, f"Confidence: {confidence:.1f}", (x, y+h+25), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    
                else:
                    # Wajah tidak dikenali atau confidence terlalu rendah
                    cv2.putText(frame, "Tidak Dikenal", (x, y-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.putText(frame, f"Confidence: {confidence:.1f}", (x, y+h+25), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            
            # Reset verification count untuk wajah yang tidak terdeteksi di frame ini
            for student_key in list(verification_count.keys()):
                if student_key not in current_detections:
                    verification_count[student_key] = 0
            
            # Tampilkan info mahasiswa yang sudah diabsen
            y_offset = 30
            cv2.putText(frame, "SUDAH ABSEN:", (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y_offset += 25
            
            for nim in list(attended_students)[:4]:  # Tampilkan max 4
                for sid, data in self.students.items():
                    if data['nim'] == nim:
                        cv2.putText(frame, f"- {data['name']}", (10, y_offset), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                        y_offset += 20
                        break
            
            # Tampilkan threshold info
            cv2.putText(frame, f"Threshold: {self.confidence_threshold}", (10, frame.shape[0] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            cv2.imshow('Pengambilan Absensi - Tekan Q untuk keluar', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        print(f"\nPengambilan absensi selesai!")
        print(f"Total mahasiswa yang hadir: {len(attended_students)}")

    def save_attendance(self, nim, name, status, confidence):
        """Menyimpan data absensi ke file CSV dengan confidence score"""
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        new_attendance = pd.DataFrame({
            "NIM": [nim],
            "Nama": [name],
            "Waktu": [current_time],
            "Status": [status],
            "Confidence": [f"{confidence:.1f}"]
        })
        
        # Append ke file CSV
        new_attendance.to_csv(self.attendance_file, mode='a', header=False, index=False)

    def view_attendance(self):
        """Menampilkan data absensi"""
        if os.path.exists(self.attendance_file):
            df = pd.read_csv(self.attendance_file)
            if not df.empty:
                print("\n" + "="*70)
                print("DATA ABSENSI MAHASISWA")
                print("="*70)
                print(df.to_string(index=False))
                print(f"\nTotal records: {len(df)}")
                
                # Tampilkan statistik
                today = datetime.now().strftime("%Y-%m-%d")
                today_attendance = df[df['Waktu'].str.contains(today)]
                if not today_attendance.empty:
                    print(f"\nAbsensi Hari Ini ({today}):")
                    print(f"Total hadir: {len(today_attendance)}")
            else:
                print("Belum ada data absensi.")
        else:
            print("File absensi tidak ditemukan.")

    def view_students(self):
        """Menampilkan data mahasiswa terdaftar"""
        if self.students:
            print("\n" + "="*50)
            print("DATA MAHASISWA TERDAFTAR")
            print("="*50)
            for student_id, data in self.students.items():
                print(f"ID: {student_id} | NIM: {data['nim']} | Nama: {data['name']}")
            print(f"\nTotal mahasiswa: {len(self.students)}")
        else:
            print("Belum ada mahasiswa yang terdaftar.")

    def adjust_threshold(self):
        """Fungsi untuk menyesuaikan threshold confidence"""
        print(f"\nThreshold confidence saat ini: {self.confidence_threshold}")
        print("Semakin rendah nilai, semakin ketat sistem (rekomendasi: 50-70)")
        try:
            new_threshold = int(input("Masukkan threshold baru: "))
            if 0 <= new_threshold <= 100:
                self.confidence_threshold = new_threshold
                print(f"Threshold berhasil diubah menjadi: {self.confidence_threshold}")
            else:
                print("Threshold harus antara 0-100")
        except ValueError:
            print("Input harus angka!")

    def run(self):
        """Menu utama program"""
        print("=" * 50)
        print("SISTEM ABSENSI WAJAH MAHASISWA")
        print("Menggunakan OpenCV dan Face Recognition")
        print("=" * 50)
        
        while True:
            print("\n=== MENU UTAMA ===")
            print("1. Daftarkan Mahasiswa Baru")
            print("2. Ambil Absensi")
            print("3. Lihat Data Absensi")
            print("4. Lihat Data Mahasiswa")
            print("5. Atur Threshold Confidence")
            print("6. Keluar")
            
            choice = input("Pilih menu (1-6): ").strip()
            
            if choice == '1':
                self.register_student()
            elif choice == '2':
                self.take_attendance()
            elif choice == '3':
                self.view_attendance()
            elif choice == '4':
                self.view_students()
            elif choice == '5':
                self.adjust_threshold()
            elif choice == '6':
                print("\nTerima kasih telah menggunakan sistem absensi!")
                break
            else:
                print("Pilihan tidak valid. Silakan pilih 1-6.")

# Jalankan program
if __name__ == "__main__":
    try:
        attendance_system = FaceRecognitionAttendance()
        attendance_system.run()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        print("\nTroubleshooting:")
        print("1. Pastikan kamera terpasang dengan benar")
        print("2. Jalankan: pip install opencv-contrib-python pandas")
        print("3. Pastikan file 'haarcascade_frontalface_default.xml' ada di folder yang sama")
        input("Tekan Enter untuk keluar...")