# Organize KenPom .csv files from behind the paywall
# Data represents regular season

import pandas as pd

year = input("Year: ")
year = str(year)
filestring = "KenPom"+year+"_pt_raw.csv"
kenpom_df = pd.read_csv(filestring)

'''
for i in range(len(kenpom_df['TeamName'])):
    if not ((kenpom_df['seed'][i] < 1) and (kenpom_df['seed'][i] > 16)):
        kenpom_df = kenpom_df.drop([i])
'''
kenpom_df = kenpom_df.dropna()

kenpom_df = kenpom_df.reset_index(drop=True)

kenpom_df = kenpom_df.drop("RankTempo",axis=1)
kenpom_df = kenpom_df.drop("RankAdjTempo",axis=1)
kenpom_df = kenpom_df.drop("RankOE",axis=1)
kenpom_df = kenpom_df.drop("RankAdjOE",axis=1)
kenpom_df = kenpom_df.drop("RankDE",axis=1)
kenpom_df = kenpom_df.drop("RankAdjDE",axis=1)
kenpom_df = kenpom_df.drop("RankAdjEM",axis=1)

kenpom_df = kenpom_df.sort_values(by=['AdjEM'],ascending=False)
kenpom_df = kenpom_df.reset_index(drop=True)

print(kenpom_df.head(70))

choice = input("Look good (y/n)? ")

if choice == "y":
    filename = "KenPom"+year+"_pt.csv"
    kenpom_df.to_csv(filename)
    print(filename, "written\n")
else:
    print("File not written\n")
    
    
    
    