str = input("please enter any string")
res = ""
storage = [][2]
l = 0
for i in range(len(str)):
    if str[i] not in res:
        res += str[i]
    else:
        storage[l][0] = res
        storage[l][1] = len(res)
        res = ""
        res += str[i]
        i -= 1
        l += 1

for i in range(len(storage)):
    print(storage[i])
