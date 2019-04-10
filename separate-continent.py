import csv

filename = "data/data.csv"
fields = []
rows = []
countries = []

sa = ['ec', 'ar', 'cl', 'br','pe','bo','uy','co','py']
saRows = []
na = ['us','sv','cr','hn','gt','ca','mx','pa','do','is']
naRows = []
eu = ['fr','fi','no','it','lt','ee','de','ch','hu','be',
      'dk','pl','at','se','pt','es','cz','ie','nl','sk',
      'lu','gb','lv','gr']
euRows = []
ai = ['ph','tw','tr','jp','my','sg','id','hk']
aiRows = []
au = ['au','nz']
auRows = []
gl = 'global'
glRows = []

with open(filename, 'r', encoding='utf-8') as fd:
    csvreader = csv.reader(fd)

    fields = next(csvreader)

    for row in csvreader:
        rows.append(row)
        # if row[fields.index("Region")] not in countries:
        #     countries.append(row[fields.index("Region")])

print(", ".join(field for field in fields))
print(countries)
print(len(countries))

regionCol = fields.index("Region")
for row in rows:
    if row[regionCol] in na:
        row[regionCol] = 'NA'
    elif row[regionCol] in sa:
        row[regionCol] = 'SA'
    elif row[regionCol] in eu:
        row[regionCol] = 'EU'
    elif row[regionCol] in ai:
        row[regionCol] = 'AI'
    elif row[regionCol] in au:
        row[regionCol] = 'AU'
    else:
        row[regionCol] = 'GL'

with open('data/engineered-data.csv', 'w', encoding='utf-8') as fd:
    csvwriter = csv.writer(fd)
    csvwriter.writerow(fields)
    csvwriter.writerows(rows)