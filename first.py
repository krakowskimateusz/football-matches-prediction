import pandas as pd

#Wczytanie danych
df = pd.read_csv("PremierLeagueMatches.csv")

print("\nPierwsze 5 wierszy:")
print(df.head())

print("\nStatystyki opisowe:")
print(df.describe())

#Liczba brakujących wartości w każdej kolumnie
print("\nBrakujące wartości w kolumnach:")
print(df.isna().sum())

#Wiersze z brakującymi wartościami
print("\nWiersze z brakującymi wartościami:")
print(df[df.isna().any(axis=1)])

#Usunięcie wierszy z brakującymi wartościami
data = df.dropna()

#Full time result
#liczba wygranych meczów u gospodarzy, remisów i wygranych meczów gości
print(data["FTR"].value_counts())

#Half time result
#Stan meczu do połowy
#H - gospodarze prowadzą, D - remis, A - goście prowadzą
print(data['HTR'].value_counts())



import numpy as np
from scipy.stats import poisson


def predict_match_outcomes_poisson(home_avg, away_avg, max_goals=10):
    home_win = 0
    draw = 0
    away_win = 0
    for home_goals in range(0, max_goals+1):
        for away_goals in range(0, max_goals+1):
            prob = poisson.pmf(home_goals, home_avg) * poisson.pmf(away_goals, away_avg)
            if home_goals > away_goals:
                home_win += prob
            elif home_goals == away_goals:
                draw += prob
            else:
                away_win += prob
    total = home_win + draw + away_win
    print("\nPrawdopodobieństwo wygranej gospodarzy: {:.2%}".format(home_win/total))
    print("Prawdopodobieństwo remisu: {:.2%}".format(draw/total))
    print("Prawdopodobieństwo wygranej gości: {:.2%}".format(away_win/total))

predict_match_outcomes_poisson(avg_home_goals, avg_away_goals)