from dataUtilities import *

print("Starting")
df = getStartData(False)
print("I have the data")
train, test = splitData(df, 9970)
print("Test",test)
bstest = bootstrap(test)
print("Bootstrap",bstest)