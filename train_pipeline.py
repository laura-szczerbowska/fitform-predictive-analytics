import random
import pandas as pd
from faker import Faker
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.model_selection import cross_validate, GroupKFold
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.linear_model import LinearRegression, ElasticNet, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
import os
import logging
from sqlalchemy import create_engine
from dotenv import load_dotenv
import matplotlib.pyplot as plt

# Konfiguracja logowania
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("fitform_backend.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)

load_dotenv()
db_url = os.getenv("FITFORM_DB_URL")

try:
    logging.info("Tworzenie połączenia SQLAlchemy z bazą danych Supabase.")

    if not db_url:
        raise ValueError("Brak zmiennej FITFORM_DB_URL w pliku .env!")

    engine = create_engine(db_url)
    nazwa_tabeli = "daily_logs"
    query = f"SELECT * FROM {nazwa_tabeli};"

    df = pd.read_sql_query(query, engine)
    logging.info(f"Pobrano {len(df)} rekordów.")

except Exception as error:
    logging.error(f"Błąd połączenia: {error}")
    exit()

os.makedirs("../static", exist_ok=True)

# Sortowanie danych po użytkowniku i dacie
df = df.sort_values(by=['user_id', 'data_wpisu'])

# Uzupełnienie braków w wadze poprzednimi i kolejnymi wpisami użytkownika
df['waga_czczo'] = df.groupby('user_id')['waga_czczo'].ffill()
df['waga_czczo'] = df.groupby('user_id')['waga_czczo'].bfill()

# Zmiana formatu daty na datetime
df['data_wpisu'] = pd.to_datetime(df['data_wpisu'])

# Tworzenie nowych cech
# Bilans kaloryczny
df['bilans_kcal'] = df['zjedzone_kcal'] - df['spalone_kcal']
df['bilans_pct'] = df['bilans_kcal'] / df['spalone_kcal'].replace(0, np.nan)

# Przeliczenie kalorii i białka na masę ciała
df['kcal_na_kg'] = df['zjedzone_kcal'] / df['waga_czczo'].replace(0, np.nan)
df['bialko_na_kg'] = df['bialko_g'] / df['waga_czczo'].replace(0, np.nan)

# Cechy związane z treningiem i aktywnością
df['spalone_z_silowym'] = df['spalone_kcal'] * df['trening_silowy']
df['cardio_i_silowy'] = df['cardio_min'] * df['trening_silowy']
df['aktywnosc_total'] = df['cardio_min'] + df['kroki'] / 100

# Flaga czy dany dzień to weekend
df['dzien_tygodnia'] = df['data_wpisu'].dt.dayofweek
df['weekend'] = (df['dzien_tygodnia'] >= 5).astype(int)

# Średnie z 3 i 7 ostatnich dni dla każdego użytkownika
for col in ['zjedzone_kcal', 'spalone_kcal', 'bilans_kcal']:
    df[f'{col}_ma3'] = df.groupby('user_id')[col].transform(lambda x: x.rolling(3, min_periods=1).mean())
    df[f'{col}_ma7'] = df.groupby('user_id')[col].transform(lambda x: x.rolling(7, min_periods=1).mean())

logging.info("Dodano cechy kontekstowe.")

# Pobranie wagi i daty z kolejnego wpisu
df['waga_jutro'] = df.groupby('user_id')['waga_czczo'].shift(-1)
df['data_nastepna'] = df.groupby('user_id')['data_wpisu'].shift(-1)

df['data_nastepna'] = pd.to_datetime(df['data_nastepna'])

# Obliczenie ile dni minęło między wpisami
df['dni_miedzy_wpisami'] = (df['data_nastepna'] - df['data_wpisu']).dt.days

# Usunięcie ostatnich wpisów, gdzie nie ma jeszcze kolejnego pomiaru wagi
df_clean = df.dropna(subset=['waga_jutro']).copy()

# Zamiana nieskończoności na braki danych
df_clean = df_clean.replace([np.inf, -np.inf], np.nan)

# Dobowa zmiana wagi (uśredniona przez liczbę dni między pomiarami), bo w prawdziwych danych waga jest co 2-3 dni, a w generowanych codziennie
df_clean['roznica_wagi'] = (df_clean['waga_jutro'] - df_clean['waga_czczo']) / df_clean['dni_miedzy_wpisami']

# Obcięcie skrajnych 5% wartości z góry i dołu, żeby usunąć błędy wpisów
Q_low = df_clean['roznica_wagi'].quantile(0.05)
Q_high = df_clean['roznica_wagi'].quantile(0.95)
df_clean = df_clean[(df_clean['roznica_wagi'] >= Q_low) & (df_clean['roznica_wagi'] <= Q_high)]

# Przygotowanie i podział danych
# Zastosowano taki podział, ponieważ konieczne jest uwzględnienie user_id 1-5 zarówno w teście, jak i treningu.
# Są to jedyne dane od rzeczywistych osób i ważne jest, aby uwzględnić je w nauce modelu, unikając przeuczenia na danych generowanych.
kolumny_do_usuniecia = [
    'data_wpisu', 'waga_czczo', 'waga_jutro', 'data_nastepna', 'dni_miedzy_wpisami', 'roznica_wagi', 'id',
    'dzien_tygodnia', 'bilans_pct', 'spalone_z_silowym', 'cardio_i_silowy', 'aktywnosc_total',
    'zjedzone_kcal_ma3', 'spalone_kcal_ma3', 'bilans_kcal_ma3', 'zjedzone_kcal_ma7', 'spalone_kcal_ma7'
]
X_surowe = df_clean.drop(kolumny_do_usuniecia, axis=1, errors='ignore')
y_surowe = df_clean['roznica_wagi']

# Usunięcie wierszy z brakami danych
czyste_indeksy = X_surowe.dropna().index.intersection(y_surowe.dropna().index)
X_pelne = X_surowe.loc[czyste_indeksy].copy()
y_pelne = y_surowe.loc[czyste_indeksy]

wszyscy_uzytkownicy = df_clean.loc[czyste_indeksy, 'user_id'].unique().tolist()
kandydaci_1_5 = [uid for uid in wszyscy_uzytkownicy if uid in [1, 2, 3, 4, 5]]

# Losowanie 1 użytkownika z user_id 1-5 do testu, reszta idzie do treningu
random.seed(42)
if kandydaci_1_5:
    uzytkownik_testowy_wymuszony = random.choice(kandydaci_1_5)
    uzytkownicy_test_wymuszeni = [uzytkownik_testowy_wymuszony]
    uzytkownicy_trening_wymuszeni = [uid for uid in kandydaci_1_5 if uid != uzytkownik_testowy_wymuszony]
else:
    uzytkownicy_test_wymuszeni = []
    uzytkownicy_trening_wymuszeni = []

pozostali_uzytkownicy = [uid for uid in wszyscy_uzytkownicy if uid not in kandydaci_1_5]

# Podział użytkowników na trening i test w proporcji 80/20
docelowa_liczba_test = max(1, int(len(wszyscy_uzytkownicy) * 0.20))
brakujaca_liczba_test = max(0, docelowa_liczba_test - len(uzytkownicy_test_wymuszeni))

uzytkownicy_test = uzytkownicy_test_wymuszeni + pozostali_uzytkownicy[:brakujaca_liczba_test]
uzytkownicy_trening = uzytkownicy_trening_wymuszeni + pozostali_uzytkownicy[brakujaca_liczba_test:]

logging.info(f"Użytkownicy w zbiorze treningowym (80%): {sorted(uzytkownicy_trening)}")
logging.info(f"Użytkownicy w zbiorze testowym (20%): {sorted(uzytkownicy_test)}")

indeksy_trening = df_clean.loc[czyste_indeksy][df_clean.loc[czyste_indeksy, 'user_id'].isin(uzytkownicy_trening)].index
indeksy_test = df_clean.loc[czyste_indeksy][df_clean.loc[czyste_indeksy, 'user_id'].isin(uzytkownicy_test)].index

kolumny_X = [col for col in X_pelne.columns if col != 'user_id']

# Czytelne nazwy kolumn do wykresów i tabel
ETYKIETY_CECH = {
    'bilans_kcal': 'Dobowy bilans kaloryczny',
    'bilans_kcal_ma7': 'Średni bilans kcal (7 dni)',
    'zjedzone_kcal': 'Spożyte kalorie (kcal)',
    'spalone_kcal': 'Spalone kalorie (kcal)',
    'kcal_na_kg': 'Kalorie / kg masy ciała',
    'bialko_na_kg': 'Białko / kg masy ciała',
    'bialko_g': 'Spożycie białka (g)',
    'trening_silowy': 'Trening siłowy',
    'cardio_min': 'Czas cardio (min)',
    'kroki': 'Liczba kroków',
    'weekend': 'Dzień weekendowy'
}

# Zmiana typów danych na float32 dla stabilności modeli
X_train = X_pelne.loc[indeksy_trening, kolumny_X].astype(np.float32)
y_train = y_pelne.loc[indeksy_trening].astype(np.float32)
user_groups = df_clean.loc[indeksy_trening, 'user_id']

X_test_user = X_pelne.loc[indeksy_test, kolumny_X].astype(np.float32)
y_test_user = y_pelne.loc[indeksy_test].astype(np.float32)

# Walidacja krzyżowa według grup użytkowników
if len(uzytkownicy_trening) >= 3:
    group_cv = GroupKFold(n_splits=3)
else:
    from sklearn.model_selection import KFold
    group_cv = KFold(n_splits=2, shuffle=True, random_state=42)


# Definicja 10 modeli do porównania
def przygotowanie_modeli():
    return {
        'XGBoost': xgb.XGBRegressor(n_estimators=150, max_depth=4, learning_rate=0.05, objective='reg:squarederror',
                                    reg_alpha=0.3, reg_lambda=2.0, random_state=42),

        'LightGBM': lgb.LGBMRegressor(n_estimators=150, max_depth=4, learning_rate=0.05, num_leaves=15,
                                      objective='regression', reg_alpha=0.3, reg_lambda=2.0, verbose=-1,
                                      random_state=42),

        'CatBoost': CatBoostRegressor(iterations=350, depth=4, learning_rate=0.05, loss_function='RMSE', verbose=0,
                                      random_state=42),

        'Random Forest': RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1),

        'Linear Regression': Pipeline([('skaler', StandardScaler()), ('model_lr', LinearRegression())]),

        'Elastic Net': Pipeline(
            [('skaler', StandardScaler()), ('model_en', ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=42))]),

        'Ridge Regression': Pipeline([('skaler', StandardScaler()), ('model_ridge', Ridge(alpha=1.0))]),

        'SVM (RBF)': Pipeline([('skaler', StandardScaler()), ('model_svm', SVR(kernel='rbf', C=5.0, epsilon=0.05))]),

        'KNN': Pipeline(
            [('skaler', StandardScaler()), ('model_knn', KNeighborsRegressor(n_neighbors=5, weights='distance'))]),

        'Decision Tree': DecisionTreeRegressor(max_depth=5, min_samples_split=10, random_state=42)
    }


# Trening wszystkich modeli bazowych
logging.info("Trenowanie modeli końcowych...")
modele_wytrenowane = przygotowanie_modeli()
for nazwa, model in modele_wytrenowane.items():
    model.fit(X_train, y_train)

# Model Ensemble łączący XGBoost, LightGBM i CatBoost ze średnią ważoną
model_ensemble = VotingRegressor(
    estimators=[
        ('xgb',
         xgb.XGBRegressor(n_estimators=150, max_depth=4, learning_rate=0.05, random_state=42).fit(X_train, y_train)),
        ('lgb',
         lgb.LGBMRegressor(n_estimators=150, max_depth=4, learning_rate=0.05, verbose=-1, random_state=42).fit(X_train,
                                                                                                               y_train)),
        ('cat', CatBoostRegressor(iterations=350, depth=4, verbose=0, random_state=42).fit(X_train, y_train))
    ],
    weights=[0.4, 0.3, 0.3]
).fit(X_train, y_train)
modele_wytrenowane['Ensemble'] = model_ensemble


# Symulacja wagi dzień po dniu na kolejne dni
def symulacja_wagi(model, dni, start_weight, kcal, bialko, spalone_kcal, cardio, silowy, kroki):
    lista_wag = [start_weight]
    obecna_waga = start_weight
    bilans = kcal - spalone_kcal

    for d in range(1, dni + 1):
        dane_wiersza = {
            'zjedzone_kcal': kcal, 'bialko_g': bialko, 'spalone_kcal': spalone_kcal,
            'cardio_min': cardio, 'trening_silowy': silowy, 'kroki': kroki,
            'bilans_kcal': bilans,
            'kcal_na_kg': kcal / obecna_waga if obecna_waga > 0 else 0,
            'bialko_na_kg': bialko / obecna_waga if obecna_waga > 0 else 0,
            'weekend': 1 if (d % 7) >= 5 else 0,
            'bilans_kcal_ma7': bilans,
        }

        wejscie = pd.DataFrame([dane_wiersza])[kolumny_X].astype(np.float32)
        roznica = model.predict(wejscie)[0]
        obecna_waga = obecna_waga + roznica
        lista_wag.append(round(obecna_waga, 2))

    return lista_wag


# Parametry wejściowe do 30-dniowej symulacji
dni_prognozy = 30
waga_start = 85.0
kalk_kcal = 1600
kalk_bialko = 130
kalk_spalone = 2600
kalk_cardio = 45
kalk_silowy = 1
kalk_kroki = 14000

# Obliczenie prognozy dla każdego modelu
wyniki_xgb = symulacja_wagi(modele_wytrenowane['XGBoost'], dni_prognozy, waga_start, kalk_kcal, kalk_bialko,
                            kalk_spalone, kalk_cardio, kalk_silowy, kalk_kroki)
wyniki_lgb = symulacja_wagi(modele_wytrenowane['LightGBM'], dni_prognozy, waga_start, kalk_kcal, kalk_bialko,
                            kalk_spalone, kalk_cardio, kalk_silowy, kalk_kroki)
wyniki_cat = symulacja_wagi(modele_wytrenowane['CatBoost'], dni_prognozy, waga_start, kalk_kcal, kalk_bialko,
                            kalk_spalone, kalk_cardio, kalk_silowy, kalk_kroki)
wyniki_rf = symulacja_wagi(modele_wytrenowane['Random Forest'], dni_prognozy, waga_start, kalk_kcal, kalk_bialko,
                           kalk_spalone, kalk_cardio, kalk_silowy, kalk_kroki)
wyniki_lr = symulacja_wagi(modele_wytrenowane['Linear Regression'], dni_prognozy, waga_start, kalk_kcal, kalk_bialko,
                           kalk_spalone, kalk_cardio, kalk_silowy, kalk_kroki)
wyniki_en = symulacja_wagi(modele_wytrenowane['Elastic Net'], dni_prognozy, waga_start, kalk_kcal, kalk_bialko,
                           kalk_spalone, kalk_cardio, kalk_silowy, kalk_kroki)
wyniki_ridge = symulacja_wagi(modele_wytrenowane['Ridge Regression'], dni_prognozy, waga_start, kalk_kcal, kalk_bialko,
                              kalk_spalone, kalk_cardio, kalk_silowy, kalk_kroki)
wyniki_svm = symulacja_wagi(modele_wytrenowane['SVM (RBF)'], dni_prognozy, waga_start, kalk_kcal, kalk_bialko,
                            kalk_spalone, kalk_cardio, kalk_silowy, kalk_kroki)
wyniki_knn = symulacja_wagi(modele_wytrenowane['KNN'], dni_prognozy, waga_start, kalk_kcal, kalk_bialko, kalk_spalone,
                            kalk_cardio, kalk_silowy, kalk_kroki)
wyniki_dt = symulacja_wagi(modele_wytrenowane['Decision Tree'], dni_prognozy, waga_start, kalk_kcal, kalk_bialko,
                           kalk_spalone, kalk_cardio, kalk_silowy, kalk_kroki)
wyniki_ensemble = symulacja_wagi(modele_wytrenowane['Ensemble'], dni_prognozy, waga_start, kalk_kcal, kalk_bialko,
                                 kalk_spalone, kalk_cardio, kalk_silowy, kalk_kroki)

# Porównanie wyników po 30 dniach w logach
logging.info("------ PORÓWNANIE KOŃCOWE ------")
logging.info(f"Waga startowa: {waga_start:.2f} kg")
logging.info(
    f"XGBOOST po {dni_prognozy} dniach: {wyniki_xgb[-1]:.2f} kg | Różnica: {(wyniki_xgb[-1] - waga_start):.2f} kg")
logging.info(
    f"LIGHTGBM po {dni_prognozy} dniach: {wyniki_lgb[-1]:.2f} kg | Różnica: {(wyniki_lgb[-1] - waga_start):.2f} kg")
logging.info(
    f"CATBOOST po {dni_prognozy} dniach: {wyniki_cat[-1]:.2f} kg | Różnica: {(wyniki_cat[-1] - waga_start):.2f} kg")
logging.info(
    f"RANDOMFOREST po {dni_prognozy} dniach: {wyniki_rf[-1]:.2f} kg | Różnica: {(wyniki_rf[-1] - waga_start):.2f} kg")
logging.info(
    f"LINEAR REGRESSION po {dni_prognozy} dniach: {wyniki_lr[-1]:.2f} kg | Różnica: {(wyniki_lr[-1] - waga_start):.2f} kg")
logging.info(
    f"ELASTIC NET po {dni_prognozy} dniach: {wyniki_en[-1]:.2f} kg | Różnica: {(wyniki_en[-1] - waga_start):.2f} kg")
logging.info(
    f"RIDGE REGRESSION po {dni_prognozy} dniach: {wyniki_ridge[-1]:.2f} kg | Różnica: {(wyniki_ridge[-1] - waga_start):.2f} kg")
logging.info(f"SVM po {dni_prognozy} dniach: {wyniki_svm[-1]:.2f} kg | Różnica: {(wyniki_svm[-1] - waga_start):.2f} kg")
logging.info(f"KNN po {dni_prognozy} dniach: {wyniki_knn[-1]:.2f} kg | Różnica: {(wyniki_knn[-1] - waga_start):.2f} kg")
logging.info(
    f"DECISION TREE po {dni_prognozy} dniach: {wyniki_dt[-1]:.2f} kg | Różnica: {(wyniki_dt[-1] - waga_start):.2f} kg")
logging.info(
    f"ENSEMBLE po {dni_prognozy} dniach: {wyniki_ensemble[-1]:.2f} kg | Różnica: {(wyniki_ensemble[-1] - waga_start):.2f} kg")

# Tabela z metrykami jakości modeli
try:
    logging.info("Generowanie tabeli...")

    scoring_metrics = {
        'rmse': 'neg_root_mean_squared_error',
        'mae': 'neg_mean_absolute_error',
        'r2': 'r2'
    }

    modele_do_cv = przygotowanie_modeli()
    modele_do_cv['Ensemble'] = VotingRegressor(
        estimators=[('xgb', przygotowanie_modeli()['XGBoost']), ('lgb', przygotowanie_modeli()['LightGBM']),
                    ('cat', przygotowanie_modeli()['CatBoost'])],
        weights=[0.4, 0.3, 0.3]
    )

    wiersze = []

    # Zastosowano walidację GroupKFold po user_id zamiast zwykłego podziału - przy małym zbiorze danych byłby on zbyt losowy.
    # Cała historia jednego użytkownika trafia albo do treningu, albo do testu - sprawdzanie modelu na osobach, których wcześniej nie widział.
    for nazwa, model_cv in modele_do_cv.items():
        cv = cross_validate(model_cv, X_train, y_train, cv=group_cv, groups=user_groups, scoring=scoring_metrics,
                            n_jobs=1)

        rmse = -cv['test_rmse']
        mae = -cv['test_mae']
        r2 = cv['test_r2']

        # Wyniki na wydzielonym zbiorze testowym
        m_wytrenowany = modele_wytrenowane[nazwa]
        if len(X_test_user) > 0:
            predykcje_testowe = m_wytrenowany.predict(X_test_user)
            test_rmse = np.sqrt(np.mean((y_test_user - predykcje_testowe) ** 2))
            test_mae = np.mean(np.abs(y_test_user - predykcje_testowe))

            wariancja_calkowita = np.sum((y_test_user - np.mean(y_test_user)) ** 2)
            if wariancja_calkowita > 0:
                test_r2 = 1 - (np.sum((y_test_user - predykcje_testowe) ** 2) / wariancja_calkowita)
            else:
                test_r2 = 0.0
        else:
            test_rmse, test_mae, test_r2 = np.nan, np.nan, np.nan

        wiersze.append({
            'Model': nazwa,
            'RMSE mean CV': round(np.mean(rmse), 4),
            'RMSE std CV': round(np.std(rmse), 4),
            'MAE mean CV': round(np.mean(mae), 4),
            'R2 mean CV': round(np.mean(r2), 4),
            'Test RMSE': round(float(test_rmse), 4) if not np.isnan(test_rmse) else "Brak danych",
            'Test MAE': round(float(test_mae), 4) if not np.isnan(test_mae) else "Brak danych",
            'Test R2': round(float(test_r2), 4) if not np.isnan(test_r2) else "Brak danych"
        })

    df_wyniki = pd.DataFrame(wiersze)
    df_wyniki = df_wyniki.sort_values(by='MAE mean CV', ascending=True).reset_index(drop=True)

    logging.info("Tabela porównawcza gotowa.")
    print("\n TABELA WYNIKÓW ")
    print(df_wyniki.to_string(index=False))

    sciezka_csv = "static/models_comparison.csv"
    df_wyniki.to_csv(sciezka_csv, index=False, encoding='utf-8-sig')
    logging.info(f"Zapisano tabelę w: {sciezka_csv}")

except Exception as e:
    logging.error(f"Błąd podczas generowania tabeli: {e}")

# Zapis wytrenowanych modeli do plików pickle dla backendu
import pickle

try:
    logging.info("Rozpoczynanie eksportu modeli LightGBM oraz Ensemble do plików pickle...")

    sciezka_lgb = "static/model_lightgbm.pkl"
    with open(sciezka_lgb, "wb") as plik_lgb:
        pickle.dump(modele_wytrenowane['LightGBM'], plik_lgb)
    logging.info(f"Pomyślnie zapisano model LightGBM w: {sciezka_lgb}")

    sciezka_ensemble = "static/model_ensemble.pkl"
    with open(sciezka_ensemble, "wb") as plik_ens:
        pickle.dump(modele_wytrenowane['Ensemble'], plik_ens)
    logging.info(f"Pomyślnie zapisano model Ensemble w: {sciezka_ensemble}")

    print("\npliki pickle gotowe")

except Exception as error_pickle:
    logging.error(f"Błąd podczas zapisu plików pickle: {error_pickle}")

# Wykres porównujący dopasowanie modeli do prawdziwej wagi użytkownika
try:
    logging.info("Generowanie wykresu dopasowania dla wszystkich 10 modeli...")

    # Wybór użytkownika z największą zmiennością wagi
    wariancje = df_clean.groupby('user_id')['waga_czczo'].std()
    if (wariancje > 0).any():
        najczestszy_user = wariancje[wariancje > 0].idxmax()
    else:
        najczestszy_user = df_clean['user_id'].value_counts().index[0]

    df_user = df_clean[df_clean['user_id'] == najczestszy_user].sort_values(by='data_wpisu').copy()

    if len(df_user) > 1:
        X_user = df_user[kolumny_X].astype(np.float32)
        delta = df_user['roznica_wagi'].values
        waga_start_user = df_user['waga_czczo'].iloc[0]

        historie_wag = {
            'Rzeczywista': [waga_start_user],
            'XGBoost': [waga_start_user],
            'LightGBM': [waga_start_user],
            'CatBoost': [waga_start_user],
            'Random Forest': [waga_start_user],
            'Linear Regression': [waga_start_user],
            'Elastic Net': [waga_start_user],
            'Ridge Regression': [waga_start_user],
            'SVM (RBF)': [waga_start_user],
            'KNN': [waga_start_user],
            'Decision Tree': [waga_start_user],
            'Ensemble': [waga_start_user]
        }

        delty_modele = {
            nazwa: model.predict(X_user) for nazwa, model in modele_wytrenowane.items()
        }

        for i in range(len(df_user)):
            historie_wag['Rzeczywista'].append(historie_wag['Rzeczywista'][-1] + delta[i])
            for nazwa_modelu, predykcje_delty in delty_modele.items():
                historie_wag[nazwa_modelu].append(historie_wag[nazwa_modelu][-1] + predykcje_delty[i])

        dni_user = list(range(0, len(historie_wag['Rzeczywista'])))

        plt.figure(figsize=(14, 8))

        plt.plot(dni_user, historie_wag['Rzeczywista'], label='WARTOŚĆ RZECZYWISTA',
                 color='black', linewidth=3.5, marker='o', zorder=5)

        # Style linii dla modeli
        plt.plot(dni_user, historie_wag['XGBoost'], label='XGBoost', linewidth=1.8, linestyle='-')
        plt.plot(dni_user, historie_wag['LightGBM'], label='LightGBM', linewidth=1.8, linestyle='-')
        plt.plot(dni_user, historie_wag['CatBoost'], label='CatBoost', linewidth=1.8, linestyle='-')
        plt.plot(dni_user, historie_wag['Ensemble'], label='Ensemble', linewidth=2.0, linestyle='-')
        plt.plot(dni_user, historie_wag['Random Forest'], label='Random Forest', linewidth=1.5, linestyle='--')
        plt.plot(dni_user, historie_wag['Linear Regression'], label='Linear Regression', linewidth=1.5, linestyle=':')
        plt.plot(dni_user, historie_wag['Elastic Net'], label='Elastic Net', linewidth=1.5, linestyle=':')
        plt.plot(dni_user, historie_wag['Ridge Regression'], label='Ridge Regression', linewidth=1.5, linestyle=':')
        plt.plot(dni_user, historie_wag['SVM (RBF)'], label='SVM (RBF)', linewidth=1.5, linestyle='-.')
        plt.plot(dni_user, historie_wag['KNN'], label='KNN', linewidth=1.5, linestyle='-.')
        plt.plot(dni_user, historie_wag['Decision Tree'], label='Decision Tree', linewidth=1.5, linestyle='--')

        plt.title("Analiza porównawcza modeli", fontsize=14, fontweight='bold', pad=15)
        plt.xlabel("Dni", fontsize=11)
        plt.ylabel("Masa ciała (kg)", fontsize=11)
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=10)

        sciezka = "static/models_timeline_comparison.png"
        plt.tight_layout()
        plt.savefig(sciezka, dpi=150, bbox_inches='tight')
        plt.close()

        logging.info(f"Zapisanie wykresu w: {sciezka}")
    else:
        logging.warning("Za mało danych, aby wygenerować pełną linię czasu.")

except Exception as e:
    logging.error(f"Błąd podczas tworzenia wykresu {e}")

# Analiza ważności cech (PFI oraz SHAP)
from sklearn.inspection import permutation_importance
import shap

try:
    logging.info("Rozpoczęcie analizy PFI i SHAP dla modelu LightGBM...")

    model_lgb = modele_wytrenowane['LightGBM']

    # Obliczenie ważności cech metodą permutacji (PFI)
    logging.info("Obliczanie PFI...")
    dane_do_pfi_X = X_test_user if len(X_test_user) > 0 else X_train
    dane_do_pfi_y = y_test_user if len(y_test_user) > 0 else y_train

    pfi_wynik = permutation_importance(model_lgb, dane_do_pfi_X, dane_do_pfi_y, scoring='neg_mean_absolute_error',
                                       n_repeats=15, random_state=42, n_jobs=-1)

    waznosci_pfi_srednie = np.maximum(0, pfi_wynik.importances_mean)
    df_pfi = pd.DataFrame({
        'Cecha': [ETYKIETY_CECH.get(col, col) for col in kolumny_X],
        'Klucz_techniczny': kolumny_X,
        'Wzrost błędu MAE (kg)': np.round(waznosci_pfi_srednie, 5),
        'Odchylenie std (kg)': np.round(pfi_wynik.importances_std, 5)
    }).sort_values(by='Wzrost błędu MAE (kg)', ascending=False).reset_index(drop=True)

    sciezka_pfi_csv = "static/tabela_pfi_lightgbm.csv"
    df_pfi.to_csv(sciezka_pfi_csv, index=False, encoding='utf-8-sig')
    logging.info(f"Zapisano tabelę PFI w: {sciezka_pfi_csv}")

    indeksy_posortowanej_waznosci = np.argsort(waznosci_pfi_srednie)
    cechy_sortowane = [ETYKIETY_CECH.get(kolumny_X[i], kolumny_X[i]) for i in indeksy_posortowanej_waznosci]
    waznosc_sortowana = waznosci_pfi_srednie[indeksy_posortowanej_waznosci]

    plt.figure(figsize=(11, 6.5))
    plt.barh(cechy_sortowane, waznosc_sortowana, color='#008080', edgecolor='black', alpha=0.85)
    plt.xlabel("Wzrost błędu dobowej zmiany wagi (MAE [kg])", fontsize=11)
    plt.title("PFI (Permutation Feature Importance)\nModel: LightGBM", fontsize=13,
              fontweight='bold', pad=15)
    plt.grid(True, linestyle=':', alpha=0.6, axis='x')

    sciezka_pfi = "static/pfi_lightgbm.png"
    plt.tight_layout()
    plt.savefig(sciezka_pfi, dpi=150, bbox_inches='tight')
    plt.close()
    logging.info(f"Zapisano wykres PFI w: {sciezka_pfi}")

    # Analiza SHAP - wpływ cech na wynik modelu
    logging.info("Obliczanie wartości SHAP...")

    X_train_etykiety = X_train.rename(columns=ETYKIETY_CECH)

    explainer = shap.TreeExplainer(model_lgb, data=X_train)
    shap_values = explainer(X_train)

    shap_values.feature_names = [ETYKIETY_CECH.get(col, col) for col in kolumny_X]

    # Globalny wykres SHAP (podsumowanie wpływu wszystkich cech)
    plt.figure(figsize=(11, 6.5))
    shap.summary_plot(shap_values, X_train_etykiety, show=False)
    plt.title("Wpływ zmiennych na dobową zmianę wagi", fontsize=13, fontweight='bold', pad=25)

    sciezka_shap_summary = "static/shap_summary.png"
    plt.tight_layout()
    plt.savefig(sciezka_shap_summary, dpi=150, bbox_inches='tight')
    plt.close()
    logging.info(f"Zapisano globalny wykres SHAP Summary w: {sciezka_shap_summary}")

    # Lokalny wykres SHAP Waterfall dla wybranego dnia
    idx_konkretnego_dnia = np.abs(shap_values.values.sum(axis=1)).argmax()

    plt.figure(figsize=(11, 6.5))
    shap.plots.waterfall(shap_values[idx_konkretnego_dnia], show=False)
    plt.title("Lokalna interpretacja pojedynczej predykcji", fontsize=12, fontweight='bold', pad=25)

    sciezka_shap_waterfall = "static/shap_waterfall.png"
    plt.tight_layout()
    plt.savefig(sciezka_shap_waterfall, dpi=150, bbox_inches='tight')
    plt.close()
    logging.info(f"Zapisano lokalny wykres SHAP Waterfall w: {sciezka_shap_waterfall}")

    logging.info("Wszystkie analizy XAI zostały pomyślnie wygenerowane i zapisane w folderze static.")

except Exception as e:
    logging.error(f"Błąd podczas analizy PFI/SHAP: {e}")