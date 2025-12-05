import os
import pandas as pd

#Ścieżka do folderu z plikami CSV
folder_path = "Dane"

#Lista plików CSV w folderze
csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

#Lista zostawionych zmiennych
columns_to_keep = [
    "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR",
    "HTHG", "HTAG", "HTR", "HS", "AS", "HST", "AST", "HC", "AC",
    "HF", "AF", "HY", "AY", "HR", "AR", "B365H", "B365D", "B365A",
    "BFH", "BFD", "BFA", "BSH", "BSD", "BSA", "BWH", "BWD", "BWA",
    "GBH", "GBD", "GBA", "IWH", "IWD", "IWA", "LBH", "LBD", "LBA",
    "PSH", "PSD", "PSA", "SOH", "SOD", "SOA", "SBH", "SBD", "SBA",
    "SJH", "SJD", "SJA", "SYH", "SYD", "SYA", "VCH", "VCD", "VCA",
    "WHH", "WHD", "WHA"
]

#Pusta lista na dane
all_data = []

#Przetwarzanie każdego pliku CSV
for file in csv_files:
    file_path = os.path.join(folder_path, file)

    try:
        #Wczytywanie pliku CSV z odpowiednim separatorem
        df = pd.read_csv(file_path, sep=",", encoding="ISO-8859-1", engine="python")

        #Sprawdzenie, czy dane są w jednej kolumnie
        if df.shape[1] == 1:
            df = pd.read_csv(file_path, sep=",", encoding="ISO-8859-1", engine="python", header=0)

        #Sprawdzenie, które kolumny są dostępne w danym pliku
        available_columns = df.columns.tolist()

        #Dodanie brakujących kolumn oraz wypełnienie ich wartością NaN
        for col in columns_to_keep:
            if col not in available_columns:
                df[col] = pd.NA

        #Wybranie tylko wymaganych kolumn
        df = df[columns_to_keep]

        #Dodanie przetworzonego pliku do listy
        all_data.append(df)

    except Exception as e:
        print(f"Błąd w pliku {file}: {e}")


#Połącznie wszystkich danych w jedną tabelę
if all_data:
    final_df = pd.concat(all_data, ignore_index=True)

    #Listy kolumn z kursami na każdy wynik
    home_odds_cols = [col for col in final_df.columns if col.endswith('H')]
    draw_odds_cols = [col for col in final_df.columns if col.endswith('D')]
    away_odds_cols = [col for col in final_df.columns if col.endswith('A')]

    #Średnie kursy
    final_df['AvgHomeOdds'] = final_df[home_odds_cols].astype(float).mean(axis=1, skipna=True).round(2)
    final_df['AvgDrawOdds'] = final_df[draw_odds_cols].astype(float).mean(axis=1, skipna=True).round(2)
    final_df['AvgAwayOdds'] = final_df[away_odds_cols].astype(float).mean(axis=1, skipna=True).round(2)

    #Usunięcie oryginalnych kolumn z kursami
    final_df = final_df.drop(columns=home_odds_cols + draw_odds_cols + away_odds_cols)

    final_df.to_csv("PremierLeagueMatches.csv", index=False)
    print("Dane zostały połączone i zapisane jako 'PremierLeagueMatches.csv'.")
else:
    print("Brak poprawnych danych do zapisania.")
