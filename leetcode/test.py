# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 02:15:47 2023

@author: myf
"""
class TreeNode():#二叉树节点
    def __init__(self,val,lchild=None,rchild=None):
        self.val=val		#二叉树的节点值
        self.lchild=lchild		#左孩子
        self.rchild=rchild		#右孩子
"""
class fibnaqi(object):
    def __init__(self,cache=None):
        if cache is not None:
            self.cache = cache
        else:
            self.cache = {}
    def caculate(self,n):
        if n in self.cache:
            return self.cache[n]
        elif n <= 2:
            return 1
        else:
            #print("caculate")
            self.cache[n]=self.caculate(n-1) + self.caculate(n-2)
            return self.caculate(n-1) + self.caculate(n-2)
cache = {}
fib1 = fibnaqi(cache)
fib1.caculate(10)
print(cache)
fib2 = fibnaqi(cache)


print("_________________________")
#print(fib2.caculate(10))
#print(fib2.cache)


class myf(object):
    def __init__(self,cache=None):
        if cache is not None:
            self.cache = cache
        else:
            self.cache = []
    def test(self):
        self.cache.append(1)

cache = [8]
myf1 = myf(cache)
print(cache)
myf1.test()
print(cache)


class yyx(object):
    def __init__(self,cache=None):
        if cache is not None:
            self.cache = cache
        else:
            self.cache = []
    def test(self):
        self.cache = 2


cache = 8
yyx1 = yyx(cache)
print(cache)
yyx1.test()
print(cache)

x=[None]
print(x)
"""

def Creat_Tree(Root,vals):
    if len(vals)==0:#终止条件：val用完了
        return Root
    if vals[0]!='#':#本层需要干的就是构建Root、Root.lchild、Root.rchild三个节点。
        Root = TreeNode(vals[0])
        vals.pop(0)
        Root.lchild = Creat_Tree(Root.lchild,vals)
        Root.rchild = Creat_Tree(Root.rchild,vals)
        return Root#本次递归要返回给上一次的本层构造好的树的根节点
    else:
        Root=None
        vals.pop(0)
        return Root#本次递归要返回给上一次的本层构造好的树的根节点

if __name__ == '__main__':
    Root = None
    strs="abc##d##e##"#前序遍历扩展的二叉树序列
    vals= list(strs)
    Roots=Creat_Tree(Root,vals)#Roots就是我们要的二叉树的根节点。
