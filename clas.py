import random


class Student:
    def __init__(self, id, cid):
        self.id = id
        self.cid = cid
        self.name = "wrb" + str(random.randint(0, 80))


class Clas:
    def __init__(self, cid):
        self.cid = cid

class1 = Clas(1)
student1 = Student(1, class1.cid)
student2 = Student(2, class1.cid)
print("Class: " + str(class1.cid) + ", student: " + student1.name + " " + student2.name)
print("Class: {class1}, student: {stu}".format(class1=str(class1.cid), stu=student1.name + " " + student2.name))

for i in range(10):
    class1 = Clas(i)
    stu = []
    for si in range(20):
        student = Student(si, class1.cid)
        stu.append(student)
    stu_s = ""
    for s in stu:
        stu_s += s.name + " "
    print("Class: {class1}, student: {stu}".format(class1=str(class1.cid), stu=stu_s))

# Class: cno, student: names
# class2 = Clas(2)
# class3 = Clas(3)
#
# nums = []
# for cid in range(3):
#     for id in range(20):
#         nums.append(Student(id, cid))



