from Levenshtein import distance
import time


list1 = "Add every keyword with Capital first letters and with all Capitals to keyword list"

list2 = ["add", "ever", "capit", "letto"]

newlist = []

start_time = time.time()

for string1 in list1.split():
    for string2 in list2:
        if distance(string1, string2) <= 3:
            newlist.append(string1)

newlist = list(set(newlist))
print(newlist)

print(time.time() - start_time, "seconds")
