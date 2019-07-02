str = input("please enter any string")
res = ""
storage = []
lstorage = []
F_index = 0
L_index = 0
temp = 0

for i in range(len(str)):
    if str[i] not in res:
        res += str[i]
    else:
        storage.append(res)
        lstorage.append(len(res))
        res = str[i]
        temp = F_index
        F_index = i
        i = temp

storage.append(res)
lstorage.append(len(res))

for i in range(len(storage)):
    print(storage[i],lstorage[i])
