import csv
import math

#############
# FUNCTIONS #
#############

def translateTimeSig(ts):
    if ts == '0/4':
        return 0, 0
    elif ts == '1/4':
        return -1, 0
    elif ts == '2/4':
        return -1/2, math.sqrt(3)/2
    elif ts == '3/4':
        return 1/2, math.sqrt(3)/2
    elif ts == '4/4':
        return 1, 0
    else:
        return 0, -1

# 'C#', 'D#', 'C', 'D', 'G#', 'F#', 'B', 'A', 'G', 'E', 'A#', 'F'
def translateKey(key, mode, alpha):
    if mode == 'Major':
        if key == 'C':
            return 0,1
        elif key == 'G':
            return 1/2,math.sqrt(3)/2
        elif key == 'D':
            return math.sqrt(3)/2,1/2
        elif key == 'A':
            return 1,0
        elif key == 'E':
            return math.sqrt(3)/2,-1/2
        elif key == 'B' or key == 'Cb':
            return 1/2,-math.sqrt(3)/2
        elif key == 'F#' or key == 'Gb':
            return 0,-1
        elif key == 'C#' or key == 'Db':
            return -1/2,-math.sqrt(3)/2
        elif key == 'Ab' or key == 'G#':
            return -math.sqrt(3)/2,-1/2
        elif key == 'Eb' or key == 'D#':
            return 0,-1
        elif key == 'Bb' or key == 'A#':
            return -math.sqrt(3)/2,1/2
        elif key == 'F':
            return -1/2,math.sqrt(3)/2
        else:
            print("ERROR", key, mode)
            return 0,0
    else:
        ret = 0,0
        if key == 'A':
            ret = 0,1
        elif key == 'E':
            ret = 1/2,math.sqrt(3)/2
        elif key == 'B':
            ret = math.sqrt(3) / 2, 1 / 2
        elif key == 'F#':
            ret = 1,0
        elif key == 'C#':
            ret = math.sqrt(3)/2,-1/2
        elif key == 'G#' or key == 'Ab':
            ret = 1/2,-math.sqrt(3)/2
        elif key == 'D#' or key == 'Eb':
            ret = 0,-1
        elif key == 'A#' or key == 'Bb':
            ret = -1/2,-math.sqrt(3)/2
        elif key == 'F':
            ret = -math.sqrt(3) / 2, -1 / 2
        elif key == 'C':
            ret = 0,-1
        elif key == 'G':
            ret = -math.sqrt(3) / 2, 1 / 2
        elif key == 'D':
            ret = -1/2,math.sqrt(3)/2
        else:
            print("ERROR", key, mode)

        x, y = ret
        x *= alpha
        y *= alpha
        return ret


#############
# MAIN CODE #
#############

filename = "data/SpotifyFeatures.csv"
rows = []

finalFields = ['artist_name','track_name','track_id','popularity',
                'acousticness','danceability','duration_ms','energy',
                'instrumentalness','liveness','loudness',
                'speechiness','tempo','valence',
                'Opera','A Capella','Alternative','Blues','Dance',
                'Pop','Electronic','R&B','Children’s Music','Folk',
                'Anime','Rap','Classical','Reggae','Hip-Hop','Comedy',
                'Country','Reggaeton','Ska','Indie','Rock','Soul',
                'Soundtrack','Jazz','World','Movie','time_sig_x',
               'time_sig_y','key_x','key_y']
twentysixzeros = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

with open(filename, 'r', encoding='utf-8-sig') as fd:
    csvreader = csv.reader(fd)

    fields = next(csvreader)
    print(fields)
    genres = []
    ids = []
    i = 0
    for row in csvreader:
        if i % 10000 == 0:
            print('i:',i)
        i += 1
        if row[fields.index('genre')] in ['Opera', 'A Capella']:
            continue
        if len(rows) % 10000 == 0:
            print('rows:',len(rows))
        alreadyFound = False
        # if row[fields.index('track_id')] == '001gDjxhKGDSx4sMMAgS9R':
        #     print("Kingdom", ids)
        if row[fields.index('track_id')] in ids:
            genre = row[fields.index('genre')]
            for j in rows:
                if j[finalFields.index('track_id')] == row[fields.index('track_id')]:
                    j[finalFields.index(genre)] = 1
        else:
            timeSig = row.pop(fields.index('time_signature'))
            mode = row.pop(fields.index('mode'))
            key = row.pop(fields.index('key'))
            genre = row.pop(fields.index('genre'))


            row.extend(twentysixzeros)
            row[finalFields.index(genre)] = 1

            row.extend([0,0])
            timeSigX, timeSigY = translateTimeSig(timeSig)
            row[finalFields.index('time_sig_x')] = timeSigX
            row[finalFields.index('time_sig_y')] = timeSigY

            row.extend([0,0])
            keyX, keyY = translateKey(key, mode, 0.5)
            row[finalFields.index('key_x')] = keyX
            row[finalFields.index('key_y')] = keyY

            rows.append(row)
            ids.append(row[finalFields.index('track_id')])
        # if key not in genres:
        #     genres.append(key)
    print(genres)

with open('data/engineered-track-data.csv', 'w', encoding='utf-8') as fd:
    csvwriter = csv.writer(fd)
    csvwriter.writerow(finalFields)
    csvwriter.writerows(rows)
