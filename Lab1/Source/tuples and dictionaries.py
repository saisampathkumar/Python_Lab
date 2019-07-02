data = (("John", ("Physics", 80)),
        ("Daniel", ("Science", 90)),
        ("John", ("Science", 95)),
        ("Mark", ("Maths", 100)),
        ("Daniel", ("History", 75)),
        ("Mark", ("Social", 95)),
        ("John", ("Maths", 180)))

dict = {}

for i in data:
    if i[0] not in dict:
        temp = [i[1]]
        dict[i[0]] = temp
    else:
        val1 = dict.get(i[0])
        val1.append(i[1])
        key = {i[0]: val1}
        dict.update(key)

for i,j in dict.items():
    print(i,j)