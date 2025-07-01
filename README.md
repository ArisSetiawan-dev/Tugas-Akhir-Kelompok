# Tugas-Akhir-Kelompok
from sklearn.datasets import load_iris
# Mengimpor dataset Iris dari library sklearn 
from sklearn.model_selection import train_test_split
# Mengimpor fungsi untuk membagi data menjadi train dan test 
from sklearn.metrics import confusion_matrix, classification_report
# Mengimpor metrik evaluasi model 
from sklearn.ensemble import RandomForestClassifier
# Mengimpor RandomForestClassifier dari data sklearn ensemble
# dir (ensemble) # (Opsional) Menampilkan dari fungsi dalam modul yang diambil yaitu (ensemble) 
data = load_iris()
# Memuat dataset Iris ke dalam variabel data 
X, y = load_iris(return_X_y=True
# Memisahkan fitur (X) dan target (y) langsung dari dataset Iris 
x = data.data
# Alternatif lain, mengambil fitur (X) dari atribut .data 
y = data.target
# Mengambil label target dari atribut .target 
# Untuk membagi dataset menjadi data latih dan data uji 
X_train, X_test, y_train, y_test = train_test_split(
# Memisahkan data: 70% untuk training, 30% untuk testing 
X, y, test_size=0.3, random_state=42
# Menggunakan random_state agar hasil baginya tetap konsisten 
) 
# Untuk membuat sebuah model Random Forest 
model = RandomForestClassifier( # Membuat suatu objek model Random Forest Classifier 
n_estimators=100, # Menggunakan 100 pohon dalam forest 
random_state=42 # Menetapkan seed random agar hasil replikasi konsisten 
) 
model.fit(X_train, y_train) # Melatih model dengan data training 
Y_predict = model.predict(X_test) # Melakukan prediksi terhadap data testing 
# Menampilkan Confusion Matrix 
print(confusion_matrix(y_test, Y_predict)) # Menampilkan matriks kebingungan sebagai evaluasi model 
# Menampilkan Classification Report 
print(classification_report(y_test, Y_predict)) # Menampilkan metrik evaluasi detail: precision, recall, f1-score 
Tujuan kode: Melakukan klasifikasi terhadap dataset Iris menggunakan Random Forest Classifier.
Langkah utama:
- Load Data
- Split Data
- Build Model
- Training
- Prediction
- Evaluation (Confusion Matrix + Classification Report)

