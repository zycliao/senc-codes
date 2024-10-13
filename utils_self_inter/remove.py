def remove(l, element):
    type_ele = type(element)
    for i in range(len(l)):
        if type(l[i]) == type_ele:
            if l[i] == element:
                l.pop(i)
                return             
        else:
            continue