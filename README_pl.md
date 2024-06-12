[EN](README.md) | [PL](README_pl.md)

# 🐦 Projekt: Wykrywanie Ptaków na Zdjęciach

Celem projektu było stworzenie modelu, który będzie sprawdzał, czy na zdjęciu występuje ptak, czy nie. Rozwiązanie zostało zastosowane w projekcie [Bird Species Recognition](https://github.com/bubiasz/team-project).

## 📊 1. Data Set

Wykorzystany zbiór danych to mix CIFAR-10, COCO, Birdsnap i CUB-200-2011, co dało nam zbiór zawierający około 150 tysięcy zdjęć w dwóch kategoriach: ptak i nie-ptak.

Największym problemem w wyszukaniu odpowiedniego zbioru było znalezienie odpowiednich datasetów, które będą jednocześnie różnorodne i nie będą zawierały ptaków. Wybór padł na CIFAR-10 i COCO, ponieważ są to bardzo duże zbiory podzielone na kategorie.

## 🚀 2. Data Loader

W celu przyspieszenia treningu modeli został stworzony data loader, który ma za zadanie rozdzielić oraz przetasować zdjęcia do folderów `train`, `val` oraz `test`. Po wykonaniu tej operacji następuje rescaling zdjęć do odpowiednich wymiarów.

## 🧠 3. Model

### MobileNetV2 (CNN, Convolutional Neural Network)

Zaprojektowany z myślą o urządzeniach mobilnych.

**Plusy:**
- Wydajność obliczeniowa
- Efektywność pamięciowa
- Wszechstronność
- Skalowalność

**Minusy:**
- Mniejsza dokładność niż duże modele
- Trudności z dostrojeniem

#### Proces przygotowania:

```python
def build_model(num_classes):
    # Ładowanie wstępnie wytrenowanego modelu MobileNetV2 bez górnych warstw (bez warstw klasyfikacyjnych)
    base_model = MobileNetV2(
        input_shape=(224, 224, 3),  # Definiowanie kształtu wejściowego obrazów (224x224x3)
        include_top=False,          # Pominięcie górnej warstwy klasyfikacyjnej modelu
        weights="imagenet"          # Użycie wag wstępnie wytrenowanych na zbiorze danych ImageNet
    )

    # Dodanie własnych warstw na szczycie modelu
    x = base_model.output  # Wyjście z wstępnie wytrenowanego modelu
    x = GlobalAveragePooling2D()(x)  # Dodanie globalnej warstwy średniego łączenia w celu zmniejszenia wymiarowości danych
    x = Dense(128, activation="relu")(x)  # Dodanie gęstej warstwy z 128 neuronami i funkcją aktywacji ReLU
    predictions = Dense(num_classes, activation="softmax")(x)  # Dodanie warstwy wyjściowej z funkcją aktywacji softmax (num_classes to liczba klas)

    # Stworzenie finalnego modelu z oryginalnym wejściem i nowym wyjściem
    model = Model(inputs=base_model.input, outputs=predictions)
    return model

# Definiowanie liczby klas (w tym przypadku 2: ptak i nie-ptak)
num_classes = 2 

# Budowanie modelu
model = build_model(num_classes)

# Wyświetlenie podsumowania modelu, aby zobaczyć jego architekturę
model.summary()
```

Podczas trenowania batch_size został ustawiony na 32 ze względu na małą moc obliczeniową (przy wyższych wartościach program się crashował).

### 🧩 Rezultat

W modelu od około 6 epoki zaczęła spadać skuteczność na zbiorze walidacyjnym. Ostatecznie po całym treningu wybrano najkorzystniejszą wersję, która uzyskała skuteczność na zbiorze testowym na poziomie 74%, co było niesatysfakcjonującym wynikiem.

**Błędy w tej iteracji:**
- Zbyt mały zbiór - wykorzystano 10% całego zbioru w celu zaoszczędzenia czasu na szkolenie.
- Trenowanie na CPU - brak zainstalowanych CUDA i cuDNN powodował trenowanie modelu na procesorze, co znacznie wydłużyło czas treningu.
- Brak regularyzacji - spadek efektywności mógł być spowodowany overfittingiem.
- Nierównoważone dane - zdjęć ptaków było kilkadziesiąt procent więcej, co mogło spowodować, że model preferował tę klasę bardziej.

## 🔄 Zmiana Podejścia

Po uzyskaniu wyników dużo poniżej oczekiwań, postanowiliśmy zmienić podejście.

### ResNet (Residual Neural Network) 
Rodzina głębokich sieci neuronowych

**Plusy:**
- Wysoka dokładność
- Efektywne trenowanie
- Łatwiejsze dostosowanie

**Minusy:**
- Złożoność obliczeniowa
- Większe ryzyko przeuczenia
- Złożoność architektury

Wybraliśmy pretrenowany model ResNet50 (wariant składający się z 50 warstw) jako kompromis pomiędzy złożonością a efektywnością.

**Poprawione błędy w stosunku do poprzedniego rozwiązania:**
- Trenowanie modelu na GPU w celu przyspieszenia procesu.
- Zwiększenie liczności zbioru do około 50%.
- Zbalansowanie liczności klas między bird a nonbird.

### 💡 Opis Rozwiązania

Klasa EarlyStopping służy do monitorowania procesu trenowania modelu i automatycznego zatrzymania treningu, jeśli wskaźnik przestaje się poprawiać. Jeśli strata walidacyjna się poprawia, model jest zapisywany, a najlepszy wynik i minimalna strata walidacyjna są aktualizowane. Po przejściu wszystkich 10 epok ostateczny model jest zapisywany w `saved_models`.

```python
class EarlyStopping:
    def __init__(self, patience=5, verbose=False, delta=0):
        self.patience = patience          # Liczba epok bez poprawy, po której trening zostanie zatrzymany
        self.verbose = verbose            # Flaga określająca, czy drukować komunikaty
        self.counter = 0                  # Licznik epok bez poprawy
        self.best_score = None            # Najlepszy wynik (najniższa strata walidacyjna)
        self.early_stop = False           # Flaga określająca, czy zatrzymać trening
        self.val_loss_min = np.Inf        # Minimalna strata walidacyjna (inicjalizowana jako nieskończoność)
        self.delta = delta                # Minimalna różnica wymagana do uznania poprawy

    def __call__(self, val_loss, model, path):
        score = -val_loss                 # Negatywna wartość straty walidacyjnej, aby minimalizacja była optymalizacją maksymalizacji

        if self.best_score is None:       # Pierwsze wywołanie
            self.best_score = score       # Ustawienie początkowego najlepszego wyniku
            self.save_checkpoint(val_loss, model, path)  # Zapis modelu
        elif score < self.best_score + self.delta:  # Brak poprawy
            self.counter += 1             # Zwiększenie licznika epok bez poprawy
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:  # Jeśli licznik przekroczy wartość patience
                self.early_stop = True    # Ustawienie flagi wczesnego zatrzymania
        else:                             # Poprawa wyniku
            self.best_score = score       # Aktualizacja najlepszego wyniku
            self.save_checkpoint(val_loss, model, path)  # Zapis modelu
            self.counter = 0              # Resetowanie licznika

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ..."
            )
        torch.save(model.state_dict(), path)  # Zapis stanu modelu do pliku
        self.val_loss_min = val_loss          # Aktualizacja minimalnej straty walidacyjnej
```

### 🔧 Przygotowanie Modelu ResNet50

```python 
# Załadowanie wstępnie wytrenowanego modelu ResNet50
resnet50 = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

# Zamrożenie wszystkich parametrów w modelu
for param in resnet50.parameters():
    param.requires_grad = False

# Modyfikacja ostatniej warstwy, aby dopasować ją do liczby klas
num_features = resnet50.fc.in_features
resnet50.fc = nn.Linear(num_features, 2)

```
## ❄️ Zamrożenie Wag
Zamrożenie wag wstępnie wytrenowanego modelu pozwala na zatrzymanie procesu aktualizacji tych wag podczas treningu, co zmniejsza ryzyko przeuczenia (overfitting) i przyspiesza trening. Skupiamy się jedynie na trenowaniu nowych lub zmodyfikowanych warstw modelu.

Oryginalna warstwa końcowa ResNet50 jest przystosowana do klasyfikacji na 1000 klas ImageNet. W naszym zadaniu mamy tylko dwie klasy (ptak i nie-ptak), dlatego musielismy dostosować ostatnią warstwę do tej liczby klas. Zastępując ją nową warstwą z dwoma neuronami, model może generować odpowiednie predykcje dla naszego specyficznego zadania.


## 🏋️‍♂️ Trening Modelu”
```python 
def train_model(
    model,                     # Model, który będzie trenowany
    dataloaders,               # Zbiór danych treningowych i walidacyjnych w postaci słownika
    criterion,                 # Funkcja straty (loss function)
    optimizer,                 # Optymalizator do aktualizacji wag modelu
    num_epochs=25,             # Liczba epok treningowych (domyślnie 25)
    patience=5,                # Liczba epok bez poprawy przed zatrzymaniem (early stopping)
    checkpoint_path="checkpoint.pth",  # Ścieżka do zapisu najlepszego modelu
):
    early_stopping = EarlyStopping(patience=patience, verbose=True)  # Inicjalizacja wczesnego zatrzymywania
    history = {"train_loss": [], "val_loss": [], "val_acc": []}       # Historia treningu

    print("Starting training...")  # Rozpoczęcie treningu

    for epoch in range(num_epochs):  # Pętla przez wszystkie epoki
        model.train()  # Ustawienie modelu w tryb treningowy
        running_loss = 0.0  # Inicjalizacja straty dla bieżącej epoki
        running_corrects = 0  # Inicjalizacja liczby poprawnych predykcji dla bieżącej epoki

        print(f"Epoch {epoch+1}/{num_epochs}")  # Informacja o bieżącej epoce

        for inputs, labels in dataloaders["train"]:  # Pętla przez wszystkie batch'e danych treningowych
            inputs = inputs.to(device)  # Przeniesienie danych wejściowych na urządzenie (CPU/GPU)
            labels = labels.to(device)  # Przeniesienie etykiet na urządzenie (CPU/GPU)

            optimizer.zero_grad()  # Wyzerowanie gradientów
            outputs = model(inputs)  # Przepuszczenie danych przez model (forward pass)
            loss = criterion(outputs, labels)  # Obliczenie straty
            loss.backward()  # Obliczenie gradientów (backward pass)
            optimizer.step()  # Aktualizacja wag modelu

            _, preds = torch.max(outputs, 1)  # Uzyskanie predykcji
            running_loss += loss.item() * inputs.size(0)  # Aktualizacja skumulowanej straty
            running_corrects += torch.sum(preds == labels.data)  # Aktualizacja liczby poprawnych predykcji

        epoch_loss = running_loss / len(dataloaders["train"].dataset)  # Obliczenie średniej straty dla epoki
        epoch_acc = running_corrects.double() / len(dataloaders["train"].dataset)  # Obliczenie dokładności dla epoki

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}")  # Informacja o wynikach epoki
        history["train_loss"].append(epoch_loss)  # Zapis straty treningowej do historii

        model.eval()  # Ustawienie modelu w tryb ewaluacji
        val_running_loss = 0.0  # Inicjalizacja straty walidacyjnej
        val_running_corrects = 0  # Inicjalizacja liczby poprawnych predykcji walidacyjnych

        with torch.no_grad():  # Wyłączenie obliczania gradientów podczas walidacji
            for inputs, labels in dataloaders["val"]:  # Pętla przez wszystkie batch'e danych walidacyjnych
                inputs = inputs.to(device)  # Przeniesienie danych wejściowych na urządzenie (CPU/GPU)
                labels = labels.to(device)  # Przeniesienie etykiet na urządzenie (CPU/GPU)

                outputs = model(inputs)  # Przepuszczenie danych przez model (forward pass)
                loss = criterion(outputs, labels)  # Obliczenie straty

                _, preds = torch.max(outputs, 1)  # Uzyskanie predykcji
                val_running_loss += loss.item() * inputs.size(0)  # Aktualizacja skumulowanej straty walidacyjnej
                val_running_corrects += torch.sum(preds == labels.data)  # Aktualizacja liczby poprawnych predykcji walidacyjnych

        val_epoch_loss = val_running_loss / len(dataloaders["val"].dataset)  # Obliczenie średniej straty walidacyjnej
        val_epoch_acc = val_running_corrects.double() / len(dataloaders["val"].dataset)  # Obliczenie dokładności walidacyjnej

        print(f"Validation Loss: {val_epoch_loss:.4f}, Validation Acc: {val_epoch_acc:.4f}")  # Informacja o wynikach walidacji
        history["val_loss"].append(val_epoch_loss)  # Zapis straty walidacyjnej do historii
        history["val_acc"].append(val_epoch_acc.item())  # Zapis dokładności walidacyjnej do historii

        # Wczesne zatrzymywanie i zapisywanie punktu kontrolnego
        early_stopping(val_epoch_loss, model, checkpoint_path)  # Sprawdzenie warunku wczesnego zatrzymywania
        if early_stopping.early_stop:
            print("Early stopping")  # Informacja o wczesnym zatrzymaniu
            break

    # Załadowanie najlepszych wag modelu
    model.load_state_dict(torch.load(checkpoint_path))
    return model, history  # Zwrócenie wytrenowanego modelu i historii treningu

# Przypisanie DataLoaderów do zmiennej 'dataloaders'
dataloaders = {"train": train_loader, "val": val_loader}

```

W celu obserwacji historii treningu zostało dodane zapisywanie wyników, a następnie ich wizualizacja za pomocą wykresu.

![image](https://github.com/sit3kk/Bird_Detector/assets/69002597/a0383c80-bb6d-404b-8e26-74f63fd8662d)


## 📈 Przebieg Treningu
Od samego początku skuteczność trenowania rosła (wykluczając jeden spadek), jednocześnie znacznie przyspieszył czas treningu (prawdopodobnie ze względu na użycie GPU). Ostatecznie udało się uzyskać skuteczność na zbiorze testowym 99%, co było wynikiem powyżej oczekiwań.

## 🏆 Ostateczna Decyzja

Wykorzystanie architektury ResNet znacząco przebiło rezultaty MobileNetV2 i zakończyło iterację w poszukiwaniu najlepszego rozwiązania

### Dlaczego tak się stało?

- Błędy metodologiczne w pierwszym modelu (niezbalansowane klasy, złe dostosowanie modelu).
- Głębsza architektura ResNet dzięki zastosowaniu bloków resztkowych, które lepiej sprawdzają się w uchwycaniu skomplikowanych wzorców danych (np. porównanie zdjęcia ptaka i nietoperza).
- ResNet używa większej liczby parametrów, co pozwala na modelowanie bardziej złożonych funkcji.

Model miał za zadanie działać po stronie serwera, więc większe użycie mocy obliczeniowej nie było aż tak istotne jak w przypadku aplikacji mobilnych.

## 🛠️ Użycie Modeli

W pliku `models_predictions.ipynb` można zobaczyć przykłady użycia modeli na zdjęciach znalezionych w internecie oraz ich czasy działania, co potwierdza skuteczność ostatecznego rozwiązania.

W folderze `utils` mamy zdefiniowane pliki `mn_utils.py` oraz `rn_utils.py`, z których możemy importować funkcje w celu predykcji zdjęcia (które na wejściu przyjmują ścieżkę).

## 📦 Instalacja

Aby uruchomić projekt, wykonaj poniższe kroki:

1. Klonowanie repozytorium:
    ```bash
    git clone https://github.com/sit3kk/Bird_Detector
    ```

2. Aktywacja wirtualnego środowiska:
    ```bash
    python -m venv venv
    source ./venv/bin/activate
    ```

3. Instalacja wymaganych bibliotek:
    ```bash
    pip install -r requirements.txt
    ```
