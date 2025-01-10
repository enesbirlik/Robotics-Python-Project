# Robot Manipulator Inverse Kinematics Solver Project

Bu proje, farklı robot manipülatörleri için çeşitli ters kinematik çözüm yöntemlerini uygulayan ve karşılaştıran bir Python uygulamasıdır.

## Proje Hakkında

Bu projede, robotik sistemlerde sıkça kullanılan beş farklı ters kinematik çözüm yöntemi uygulanmış ve performansları karşılaştırılmıştır:

- Newton-Raphson Metodu
- Damped Least Squares (DLS)
- Jacobian Bazlı Çözüm
- Cyclic Coordinate Descent (CCD)
- FABRIK (Forward And Backward Reaching Inverse Kinematics)

## Teknolojiler ve Gereksinimler
- Python 3.8+
- NumPy
- SciPy
- Robotics Toolbox
- CustomTkinter (GUI için)

## Kurulum

```bash
# Gerekli kütüphanelerin kurulumu
pip install numpy scipy roboticstoolbox-python customtkinter
```
## Proje Yapısı

Proje aşağıdaki ana modüllerden oluşmaktadır:

```plaintext
├── main.py                # Ana test programı
├── ik_solvers.py          # Ters kinematik çözücüler
├── robot_manipulator.py   # Robot manipülatör sınıfı
├── robot_config.py        # Robot konfigürasyonları
└── new_gui.py             # Grafiksel kullanıcı arayüzü
```

## Yapılandırma ve Kullanım
### Robot Konfigürasyonları
robot_config.py dosyasında dört farklı robot tipi tanımlanmıştır:

- Custom Robot (RRP): 
3 eklemli (2 döner, 1 prizmatik).
DH parametreleri ve eklem limitleri kullanıcıdan girilebilir.

- UR5: 
6 eklemli endüstriyel robot.
Standart UR5 parametreleri.

- SCARA: 
4 eklemli SCARA tipi robot.
Özelleştirilmiş SCARA parametreleri.

- KR6: 
6 eklemli KUKA robot.
Endüstriyel standartlara uygun parametreler.

### Ters Kinematik Çözücüler
ik_solvers.py dosyasında bulunan çözüm metodları:

- Newton-Raphson Metodu: 
Matematiksel iterasyon bazlı çözüm.
Yüksek hassasiyet ve hızlı yakınsama.
newton_raphson_solver() fonksiyonu ile çağrılır.
- Damped Least Squares (DLS):
Tekil durumlarda kararlı çözüm.
Ayarlanabilir sönümleme faktörü.
dls_solver() fonksiyonu ile çağrılır.
- Jacobian Bazlı Çözüm: 
Hız bazlı kinematik çözüm.
Basit ve etkili implementasyon.
jacobian_solver() fonksiyonu ile çağrılır.
- Cyclic Coordinate Descent (CCD):
[CCD](https://www.youtube.com/watch?v=xHN6PkfQVLY)
Eklem bazlı iteratif çözüm.
Düşük hesaplama maliyeti.
ccd_solver() fonksiyonu ile çağrılır.
- FABRIK: 
İleri-geri kinematik çözüm.
Hızlı yakınsama özelliği.
fabrik_solver() fonksiyonu ile çağrılır.

### Test Konumları ve Parametreler
robot_config.py dosyasında şu fonksiyonlarla ayarlanır:
```python
# Test pozisyonlarını ayarlama
def get_test_positions():
    return [
        [100, 20, 140],    # Başlangıç pozisyonu (mm)
        #[150, 0, 100]       # Hedef pozisyon (mm) İkinci nokta opsiyoneldir. Birden fazla nokta varken yörüngeyi çözer.
    ]

# Çözücü parametrelerini ayarlama  
def get_solver_parameters():
    return {
        'max_iter': 1000,    # Maksimum iterasyon sayısı
        'tolerance': 1e-2,   # Hata toleransı (mm)
        'lambda_val': 0.2,   # DLS sönümleme faktörü
        'alpha': 0.5         # Jacobian adım büyüklüğü
    }
```

### Program Çalıştırma
#### Temel Test İçin:
```bash
python main.py
```
Bu komut:
- Tüm çözücüleri test eder
- Performans metriklerini gösterir
- Sonuçları görselleştirir

## GUI Özellikleri ve Kullanımı

[Örnek Video](https://www.youtube.com/watch?v=PqQ_4BVTA54)

![GUI Görüntüsü](/images/img_gui1.png)
![GUI Görüntüsü 2](/images/img_gui2.png)

### Temel Bileşenler
### 1. Robot Konfigürasyon Paneli
- Robot Type: Robot tipi seçimi (SCARA, UR5, KR6, Custom)
- Number of Joints: Eklem sayısı seçimi (2-10 arası)
- DH Parametre Tablosu:
- Joint Type (R: Revolute, P: Prismatic)
- a, alpha, d, offset parametreleri
- q_min, q_max (eklem limitleri)
### 2. Kontrol Paneli
Solver: Çözücü algoritması seçimi
- jacobian
- dls (Damped Least Squares)
- ccd (Cyclic Coordinate Descent)
- fabrik
- newton
Target Position: Hedef konum girişi (X, Y, Z) mm cinsinden
### 3. Butonlar
- Solve IK: Seçili çözücü ile ters kinematik hesaplama
- Reset Robot: Robotu başlangıç pozisyonuna döndürme
- Add Point: Yörünge noktası ekleme
- Clear Points: Yörünge noktalarını temizleme
- Animate: Yörünge animasyonunu başlatma
### 4. Performans Metrikleri
- Iterations: Çözüm için gereken iterasyon sayısı
- Time: Çözüm süresi (saniye)
- Error: Konum hatası (mm)
### 5. Log Paneli
- Eklenen noktalar
- Bulunan çözümler
- Eklem açıları/konumları
- Hata mesajları
### 6. Yörünge Noktaları
- Eklenen noktaların listesi
- Her nokta için X, Y, Z koordinatları
![GUI Görüntüsü 3](/images/img_gui3.png)

### Örnek Kullanım
Temel Çözüm İşlemi:
```plaintext
1. Robot tipini seçin (örn: SCARA)
2. Hedef pozisyonu girin (X, Y, Z)
3. Çözücü seçin (örn: jacobian)
4. "Solve IK" butonuna tıklayın
```
Yörünge Oluşturma:
```plaintext
1. İlk hedef noktayı girin
2. "Add Point" ile ekleyin
3. İkinci noktayı girin ve ekleyin
4. "Animate" ile yörüngeyi canlandırın
```
Önemli Notlar
- Robot tipi değiştirildiğinde DH parametreleri otomatik güncellenir
- Prizmatik eklemler (P) için limit değerleri mm cinsindendir
- Döner eklemler (R) için limit değerleri radyan cinsindendir
- Hedef nokta robot çalışma uzayı dışındaysa uyarı verilir
- Yörünge noktaları arası otomatik interpolasyon yapılır
