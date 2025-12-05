import pandas as pd

# Wczytaj dane
df = pd.read_csv('PremierLeagueMatches.csv')

# Posortuj po dacie
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')

# Funkcja do obliczania formy drużyny
def calculate_team_form(df, team, date, n=5):
    # Filtruj mecze tej drużyny przed daną datą
    mask = ((df['HomeTeam'] == team) | (df['AwayTeam'] == team)) & (df['Date'] < date)
    last_matches = df[mask].tail(n)
    # Liczba zwycięstw
    wins = ((last_matches['HomeTeam'] == team) & (last_matches['FTR'] == 'H')).sum() + \
           ((last_matches['AwayTeam'] == team) & (last_matches['FTR'] == 'A')).sum()
    # Liczba remisów
    draws = ((last_matches['FTR'] == 'D')).sum()
    # Liczba porażek
    losses = n - wins - draws
    return wins, draws, losses

# Przykład użycia dla jednego meczu
row = df.iloc[10]
home_form = calculate_team_form(df, row['HomeTeam'], row['Date'])
away_form = calculate_team_form(df, row['AwayTeam'], row['Date'])
print('Forma gospodarzy:', home_form)
print('Forma gości:', away_form)