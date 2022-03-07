
mydict = {}


mydict["hallo"] = {}
mydict["hallo"]["hallo_sub"] = 1

mydict["hallo2"] = {}
mydict["hallo2"]["hallo2_sub"] = 2

print(mydict.items())

mydict2 = {}
mydict2["hallo"] = 1
mydict2["hallo2"] = 2

print(mydict2.items())

print()
for key, item in mydict.items():
    print(key)
    print(item.items())

for key, item in mydict2.items():
    print(key)