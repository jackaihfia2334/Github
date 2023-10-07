
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 17 22:52:17 2023

@author: myf
"""
from typing import List
import collections

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next\

def buildlist(input:List)-> ListNode:

    dummy = ListNode(-1)
    p = dummy
    if input is not None:
        for i in range(len(input)):
            p.next = ListNode(val=input[i])
            p = p.next
        return dummy.next
    else:
        return None

def traverselist(l1:ListNode)-> List:
    
    if l1 is not None:
        res = []
        p = l1
        while p:
            res.append(p.val)
            p = p.next
        return res
    else:
        return None

def mergeTwoLists(l1: ListNode, l2: ListNode) -> ListNode:
    # 虚拟头结点
    dummy = ListNode(-1)
    p = dummy  #为什么要引入p呢 因为最后我们需要返回结果链表的头节点啊，所以不能直接移动dummy
    p1 = l1
    p2 = l2

    #双指针移动
    while p1 and p2:

        if p1.val>p2.val:
            p.next = p2.val
            p2 = p2.next
        else:
            p.next  = p1.val
            p1 = p1.next
        p = p.next
    
    if p1:
        p.next = p1
    if p2:
        p.next = p2

        return dummy.next


"""
代码中还用到一个链表的算法题中是很常见的「虚拟头结点」技巧，也就是 dummy 节点。
如果不使用 dummy 虚拟节点，代码会复杂一些，需要额外处理指针 p 为空的情况。
而有了 dummy 节点这个占位符，可以避免处理空指针的情况，降低代码的复杂性
"""
#x = traverselist(buildlist([]))
#print(x)
x = []
print(x is not None)
