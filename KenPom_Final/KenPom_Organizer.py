# -*- coding: utf-8 -*-
"""
Script to organize raw KenPom data,
Pulled from kenpom.com

To prepare for this step:
1) Delete excess rows at the top
2) Line up columns properly
3) Delete excess columns
"""

import pandas as pd

year = input("Year: ")
year = str(year)
filestring = "KenPom"+year+"_raw.csv"
kenpom_df = pd.read_csv(filestring)

droplist = []
    
for i in range(len(kenpom_df['Team'])):
    if not kenpom_df['Team'][i][-1].isdigit():
        kenpom_df = kenpom_df.drop([i])

kenpom_df = kenpom_df.reset_index(drop=True)
kenpom_df["Seed"] = 0

for i in range(len(kenpom_df['Team'])):
    team = kenpom_df['Team'][i]
    if team[-1].isdigit() and team[-2].isdigit():
        seed = int(team[-2:])
        kenpom_df['Team'][i] = team[:-2]
    elif team[-1].isdigit():
        seed = int(team[-1])
        kenpom_df['Team'][i] = team[:-1]
    else:
        continue
    kenpom_df['Seed'][i] = seed
    
kenpom_df['W'] = 0
kenpom_df['L'] = 0
kenpom_df['Win%'] = 1.0

for i in range(len(kenpom_df['Team'])):
    kenpom_df['W'][i] = int(kenpom_df['W-L'][i][:2])
    if not kenpom_df['W-L'][i][-2].isdigit(): # >10 losses
        kenpom_df['L'][i] = int(kenpom_df['W-L'][i][-1])
    else:
        kenpom_df['L'][i] = int(kenpom_df['W-L'][i][-2:])
    kenpom_df['Win%'][i] = kenpom_df['W'][i]/(kenpom_df['W'][i]+kenpom_df['L'][i])
    
kenpom_df = kenpom_df.drop("W-L",axis=1)

print(kenpom_df.head(70))

choice = input("Look good (y/n)? ")

if choice == "y":
    filename = "KenPom"+year+".csv"
    kenpom_df.to_csv(filename)
    print(filename, "written\n")
else:
    print("File not written\n")
 