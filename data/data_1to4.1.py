# 1 -> 2
import pickle
import os
import re
import pandas as pd


def extract_chords(text):
    chords = re.findall(r'[A-G](?:#|b)?(?:m|M|dim|aug)?(?:maj|min)?\d?(?:sus\d)?(?:/[A-G](?:#|b)?)?', text)
    return chords


def split_chord(chord):
    if len(chord) > 1 and (chord[1] == '#' or chord[1] == 'b'):
        return chord[:2], chord[2:]
    else:
        return chord[0], chord[1:]


here = os.path.dirname(os.path.abspath(__file__))

with open(here + '\data-1.pkl', 'rb') as f:
    data = pickle.load(f)


chords = []
for song in data['chords']:
    chords.append([split_chord(chord) for chord in song])

df = pd.DataFrame({"chords" : chords, "genres" : data['genres']})

print(df)

with open(here + '\data-2.pkl', 'wb') as f:
    pickle.dump(df, f)


# 2 -> 3


def split_slash(chord):
    splited = chord[1].split('/')
    if len(splited) == 1:
        return chord[0], splited, ''
    return chord[0], splited[0], splited[1]


here = os.path.dirname(os.path.abspath(__file__))

with open(here + '\data-2.pkl', 'rb') as f:
    data = pickle.load(f)

chords = []
for song in data['chords']:
    tmp = []
    for chord in song:
        sp = chord[1].split('/')
        if len(sp) == 1:
            tmp.append([chord[0], sp[0], ""])
        else:
            tmp.append([chord[0], sp[0], sp[1]])
    chords.append(tmp)
        

df = pd.DataFrame({"chords" : chords, "genres" : data['genres']})

print(df)

with open(here + '\data-3.pkl', 'wb') as f:
    pickle.dump(df, f)


# 3 -> 4


here = os.path.dirname(os.path.abspath(__file__))

with open(here + '\data-3.pkl', 'rb') as f:
    data = pickle.load(f)

code_dict = {
    'C': 1, 'B#' : 1,
    'C#': 2, 'Db': 2,
    'D': 3,
    'D#': 4, 'Eb': 4,
    'E': 5, 'Fb' : 5, 
    'E#' : 6, 'F': 6,
    'F#': 7, 'Gb': 7,
    'G': 8,
    'G#': 9, 'Ab': 9,
    'A': 10,
    'A#': 11, 'Bb': 11,
    'B': 12, 'Cb' : 12
}

chords = []
for song in data['chords']:
    tmp = []
    for chord in song:
        if len(chord[2]) > 0:
            converted = [code_dict.get(chord[0]), chord[1], code_dict.get(chord[2])]
        else:
            converted = [code_dict.get(chord[0]), chord[1], chord[2]]
        tmp.append(converted)
    chords.append(tmp)
        

df = pd.DataFrame({"chords" : chords, "genres" : data['genres']})

print(df)

with open(here + '\data-4.pkl', 'wb') as f:
    pickle.dump(df, f)


# 4 -> 4.1


here = os.path.dirname(os.path.abspath(__file__))
with open(here + '\data-4.pkl', 'rb') as f:
    data = pickle.load(f)

df = data[data['chords'].str.len() != 0]

print(df)

df = df.reset_index(drop=True)

print(df)

with open(here + '\data-4.1.pkl', 'wb') as f:
    pickle.dump(df, f)
