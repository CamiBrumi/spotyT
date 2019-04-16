import csv

filename = "data/data.csv"
rows = []

countries = ['ec', 'ar', 'cl', 'br','pe','bo','uy','co','py','us','sv',
             'cr','hn','gt','ca','mx','pa','do','is','fr','fi','no','it',
             'lt','ee','de','ch','hu','be','dk','pl','at','se','pt','es',
             'cz','ie','nl','sk','lu','gb','lv','gr','ph','tw','tr','jp',
             'my','sg','id','hk','au','nz','global']

countryArray = []
for c in countries:
    countryArray.append([])

with open(filename, 'r', encoding='utf-8') as fd:
    csvreader = csv.reader(fd)

    fields = next(csvreader)

    for r in csvreader:
        # rows.append(r)
        found = False
        region = r[fields.index("Region")]
        i = countries.index(region)
        if i < 100:
            if len(countryArray[i]) == 0:
                countryArray[i].append(r)
            else:
                length = len(countryArray[i])
                for j in range(length):
                    if not found and countryArray[i][j][fields.index('URL')] == r[fields.index('URL')]:
                        found = True
                        if int(countryArray[i][j][fields.index('Position')]) > int(r[fields.index('Position')]):
                            countryArray[i][countryArray[i].index(countryArray[i][j])] = r
                        break
                if not found:
                    countryArray[i].append(r)
        # if row[fields.index("")] not in countries:
        #     countries.append(row[fields.index("Region")])

print(", ".join(field for field in fields))

print("FINISHED ANALYSIS")
rows = []
for c in countryArray:
    rows.extend(c)
    countryArray.remove(c)

with open('data/engineered-data.csv', 'w', encoding='utf-8') as fd:
    csvwriter = csv.writer(fd)
    csvwriter.writerow(fields)
    csvwriter.writerows(rows)
    # csvwriter.writerows(countryArray[1])
