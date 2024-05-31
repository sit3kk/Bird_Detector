Opis projektu
Celem projektu bylo stworzenie modelu ktory bedzie sprawdzal czy na zdjeciu wystepuje ptak czy nie. Rozwiazanie zostal zastosowane w projekcie https://github.com/bubiasz/team-project



1. Data set

Wykorzystany DatSet to mix CIFAR-10, COCO oraz Birdsnap i CUB-200-2011 co dalo nam zbior zawierjacy okolo 150 tysiecy zdjec w dwoch kategoria ptak i nieptak,

Najwiekszym problemem w wyszukaniu odpowiedniego zbioru bylo znalezienia odpowiednich data setow na ktorym nie znajduja sie ptaki ktore beda jednoczesnie roznorodne, wybor padl na CIFAR-10, COCO gdyz sa ta bardzo duze zbiory podzielone na kategorie.

2. Data loader

W celu przyspiesznia treningu modeli zostal stowrzony data loader ktory ma za zadanie rozdzielic oraz przetasowac zdjecia do folderow train, val oraz test. Po wykonaniu tej operacji nastepuje rescalling zdjec do odpowiednich wymiarow


3. Model

W celu uzyskania optymalnych wynikow postanowilismy przetestowac kilka podejsc i sprawdzic ktore jest najlepsze jednoczesnie poprawiajac bledy.
Uzycie pretrenowanych modeli mialo zaoszczedzic czas, moc obliczeniowa oraz ulatwic wykonanie zadania.

MobileNetV2 (CNN, Convoluational Neural Network) zaprojektowany z mysla o urzadzeniach mobilnych.

Plusy
+ Wydajnosc obliczeniowa
+ Efektywnosc pamieciowa
+ Wszechstronnosc
+ Skalowalnosc

Minusy
- Mniejsza dokladnosc niz duze modele
- Trudnosci z dostrojeniem


Proces przygotowania
```
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


Podczas trenowania batch_size zostal ustawiony na 32 co bylo spodwodowane mala moca obliczeniowa(Przy wyzszych wartosciach crashowalo program)

W celu zabezpieczenia potenacjnych przerwan treningu zostaly ustawione checkpointy po kazdej epoce zapisywane w folderze oraz funkcja ktora bedzie przywracala nasz trening do najwczescniejszej epoki.
Testowanie na zbiorze walidacyjnym mialo za zadanie mozliwosc obserwacji live statysytk modelu co mialo zapobiegac overfittingowi..

Rezultat
W modelu od okolo 6 epoki zaczala spadac skutecznosc na zbiorze walidacyjnym. Ostatecznie po calym treningu zostala wybrana najkorzystniejsza wersja ktora uzyskala skutecznosc na zbiorze testowym na poziomie 74% co bylo niesatysfakcjonujacym wynikiem.

Bledy w danej iteracji
Zbyt maly zbior - wykorzystane zostalo 10% calego zbioru w celu zaoszczedzenia czasu na szkolenie.
Trenowanie na CPU - brak sterownikow CUDA, cuDNN powodowaly trenowanie modelu na procesorze co prawdopobnie znacznie wydluzylo czas treningu.
Brak regularyzcji - spadek efektynowsci mogl byc spodowoany overfittingiem.
Nierownowazone dane - zdjec ptakow bylo kilkadzesiat procent wiecej co moglo spododowac ze model preferowal te klase bardziej


Po otrzymaniu wyniku duzo ponizej oczekiwan postanowilismy na zmiana podejscia.

ResNet (Residual neural network)

Plusy
+ Wysoka dokladnosc
+ Efektyne trenowanie
+ Latwiejsze dostosowanie

Minus
- Zlozonosc obliczeniowa
- Wieksze ryzyko przeuczenia
- Zlozonsc architektury

Wybor padl na uzycie pretrenowanego modelu ResNet50(wariant skladajacy sie z 50 wartstw) wydawal sie komporpmisem pomiedzy zlozonscia a efektynwoscia


Poprawione bledy w stosunu do poprzedniego rozwiazania

Trenowanie modelu na GPU w celu przyspieszenia procesu
Zwiekszenie licznosci zbioru do okolo 50%
Zbalansowanie licznosci klas miedzy bird a nonbird


Opis rozwiazania

Klasa EarlyStopping sluzy do monitorowania procesu trenowania modelu i automatczniego zatrzymania treningu jesli wskaznik przestaje sie poprawiac.
Jesli strata walidacyjna sie poprawia model jest zapisywane a najlepszy wynik i minimalna strata walidacyjna sa aktualizowane.
Po przejsciu wszystkich 10 epok ostateczny model jest zapisywane w saved_models

```
import numpy as np
import torch

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


```# Załadowanie wstępnie wytrenowanego modelu ResNet50
resnet50 = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

# Zamrożenie wszystkich parametrów w modelu
for param in resnet50.parameters():
    param.requires_grad = False

# Modyfikacja ostatniej warstwy, aby dopasować ją do liczby klas
num_features = resnet50.fc.in_features
resnet50.fc = nn.Linear(num_features, 2)
```
Zamrożenie wag wstępnie wytrenowanego modelu pozwala na zatrzymanie procesu aktualizacji tych wag podczas treningu, co zmniejsza ryzyko przeuczenia (overfitting) i przyspiesza trening. Skupiamy się jedynie na trenowaniu nowych lub zmodyfikowanych warstw modelu.

Oryginalna warstwa końcowa ResNet50 jest przystosowana do klasyfikacji na 1000 klas ImageNet. W naszym zadaniu mamy tylko dwie klasy (ptak i nie-ptak), dlatego musielismy dostosować ostatnią warstwę do tej liczby klas. Zastępując ją nową warstwą z dwoma neuronami, model może generować odpowiednie predykcje dla naszego specyficznego zadania.


```def train_model(
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


W celu obserwacji histori treningu zostalo dodane zapisywane wynikow a nastepnie ich wizualizacja za pomoca wykresu

Przebieg treningu
Od samego poczatku skutecznosc trenowania rosla (wykluczajac jeden spadek), jednoczesnie znacznie przyspieszyl czas treningu (prawodpobonie ze wzgledu na uzycie GPU). Ostatecznie udalo sie uzyskac skutecznosc na zbiorze testowym 99% co bylo wynikiem powyzej oczekiwan



Ostateczna decyzja

Wykorzystanie archiektury ResNet znaczaco przebilo rezultaty MobiletNetV2 i zakonczylo iteracje w poszukiwaniu najlepszego rozwiaznia.

Dlaczego tak sie stalo?
Bledy metodologiczne w pierszym modelu (Niezbalansowane klasy, zle dostosowanie modelu)

Glebsza architektura ResNet dzieki zastosowaniu blokow resztkowych ktora lepiej sie sprtawdza w uchwycaniu skomplikowanych wzorcow danych (np .p orownniae zdjecia ptaka i nietoperza)

ResNet uzywa wiekszej liczby paramewtrow co pozwala na modelowania bardziej zlozonych funkcji. 

Model mial za zaadnie dzialac po stronie servera wiec wieksze uzycie mocy obliczeniowej nie bylo az tak istotne jak w przypadku aplikacji mobilnych.



Uzycie modeli

W models_predictions.ipynb mozemy zobaczyc przyklady uzycia modeli na zdjeciach znalezionych w internecie oraz ich czasy dzialania co potwierdza skutecznosc ostatecznego rozwiazania.

w folderze utils mamy zdefiniowane pliki mn_utils.py oraz rn_utils.py z ktorych mozemy importowac funkcje w celu predykcji zdjecia(ktore na wejsciu przyjmuja sciezke)


Instalacja

#Aktywacja wirtualnego srodowiska
git clone https://github.com/sit3kk/Bird_Detector
python -m venv venv
source ./venv/bin/activate
pip install -r requiremnets.txt