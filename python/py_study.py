# -*- coding: utf-8 -*-

"""
私有属性

在实例变量前加单个下划线，以“保护”变量，变量并不是真正的受保护，这只是Python开发者的约定，
在看到单个前置下划线的变量时，并不会尝试访问和修改它。双前置下划线会让Python解释器重写属性的名称，以达到保护变量的目的。
使用双前下划线可以很好地保护类中的资源，但访问类中资源的通道并不是完全关闭的，经过修改名称后，仍可以访问。
"""


class Car(object):
    def __init__(self,age,color,brand="奥迪"): #

        self.__brand = brand  #两个下划线表示
        self._age = age #实例属性
        self.color = color #实例属性
car = Car(9,"red")
print(car.color)
print(car._age)
#print(car.__brand)  无法访问
print(car._Car__brand)  #修改后可以访问

"""
私有方法
私有方法采用双前置下划线的形式表示，只能在类内调用，与私有属性同理，不建议修改
"""

class cat:
    def __init__(self,age,color,name="xiao"): #构造方法
        self.name = name #实例属性
        self.age = age #实例属性s
        self.color = color #实例属性
        
    def __voice(self): # 私有方法
        print("我的名字是{}，我{}岁了！".format(self.name,self.age))
x = cat(6,"red") 
#x.__voice()   无法访问
x._cat__voice()


"""

函数sorted(iterable, cmp=None, key=None, reverse=False)

args:
iterable -- 可迭代对象。
cmp -- 比较的函数，这个具有两个参数，参数的值都是从可迭代对象中取出，此函数必须遵守的规则为，大于则返回1，小于则返回-1，等于则返回0。
key -- 主要是用来进行比较的元素，只有一个参数，具体的函数的参数就是取自于可迭代对象中，指定可迭代对象中的一个元素来进行排序。
reverse -- 排序规则，reverse = True 降序 ， reverse = False 升序（默认）。

return：
排序后的对象

"""

a = [5,7,6,3,4,1,2]
b = sorted(a)       # 保留原列表
print(a)
print(b)
 
L=[('b',2),('a',1),('c',3),('d',4)]
sorted(L, cmp=lambda x,y:cmp(x[1],y[1]))   # 利用cmp函数
sorted(L, key=lambda x:x[1])               # 利用key

 
 
students = [('john', 'A', 15), ('jane', 'B', 12), ('dave', 'B', 10)]
sorted(students, key=lambda s: s[2])            # 按年龄排序
sorted(students, key=lambda s: s[2], reverse=True)       # 按降序
