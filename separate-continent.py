import csv

filename = "data/data.csv"
fields = []
rows = []
countries = []

with open(filename, 'r', encoding='utf-8') as fd:
    csvreader = csv.reader(fd)

    fields = next(csvreader)

    for row in csvreader:
        rows.append(row)
        if row[fields.index("Region")] not in countries:
            countries.append(row[fields.index("Region")])

print(", ".join(field for field in fields))
print(countries)
print(len(countries))

# sa = ['ec', 'ar', 'cl', '']

# ['ec', Ecuador
#  'fr',France
#  'ar', Argentina
#  'fi', Finland
#  'no', Norway
#  'it', Italy
#  'lt', Lithuania
#  'ph', Philippines
#  'tw', Taiwan
#  'nz', New Zealand
#  'ee', Estonia
#  'tr', Turkey
#  'us', United States
#  'sv', El Salvador
#  'cr', Costa Rica
#  'de', Germany
#  'cl', Chile
#  'jp', Japan
#  'br', Brazil
#  'hn', Honduras
#  'gt', Guatemala
#  'ch', Switzerland
#  'hu', Hungary
#  'ca', Canada
#  'pe', Peru
#  'be', Belgium
#  'my', Malaysia
#  'dk', Denmark
#  'bo', Bolivia
#  'pl', Poland
#  'at', Austria
#  'pt', Portugal
#  'se', Sweden
#  'mx', Mexico
#  'pa', Panama
#  'uy', Uruguay
#  'is', Iceland
#  'es', Spain
#  'cz', Czech
#  'ie', Ireland
#  'nl', Netherlands
#  'sk', Slovakia
#  'co', Colombia
#  'sg', Singapore
#  'id', Indonesia
#  'do', Dominican Republic
#  'lu', Luxembourg
#  'gb', UK
#  'global', Global
#  'py', Paraguay
#  'au', Australia
#  'lv', Latvia
#  'gr', Greece
#  'hk'  Hong Kong]