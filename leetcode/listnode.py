
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 17 22:52:17 2023

@author: myf
"""
from typing import List
from typing import Optional
import collections

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def buildlist(input:List)-> ListNode:
    dummy = ListNode(-1)
    p = dummy
    if input:
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
        if p1.val > p2.val:
            p.next = p2
            p2 = p2.next
        else:
            p.next  = p1
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

"""
测试用例
[1,2,4] [1,3,6]
[] []
[0] []
[] [1]

x = traverselist(buildlist([]))
print(x)
"""


# 分割链表
def partition(head: ListNode, x: int) -> ListNode:
    dummy1 = ListNode(-1)
    dummy2 = ListNode(-1)
    p1 = dummy1
    p2 = dummy2

    while head:
        if head.val<x:
            p1.next = ListNode(head.val)
            #p1.next = head
            p1 = p1.next
        else:
            p2.next = ListNode(head.val)
            #p2.next = head
            p2 = p2.next

        head = head.next
        #temp = head.next
        #head.next = None
        #head = temp
    
    p1.next  = dummy2.next
    return dummy1.next
    


"""
如果我们需要把原链表的节点接到新链表上，而不是 new 新节点来组成新链表的话，
那么断开节点和原链表之间的链接可能是必要的。那其实我们可以养成一个好习惯，
但凡遇到这种情况，就把原链表的节点断开，这样就不会出错了。
"""

"""
x = buildlist([1,0,2,4,3,4,2,4])
y = partition(x,3)
print(traverselist(y))
"""

#返回链表中间节点
class Solution:
    def middleNode(self, head: Optional[ListNode]) -> Optional[ListNode]:
        if head is None:
            return None
        
        p1 = head
        p2 = head

        while p1.next:
            if p1.next.next:
                p1 = p1.next.next
            else:
                p1 = p1.next
            p2 = p2.next
        return p2
    
#返回链表倒数第K个节点
class Solution:
    def trainingPlan(self, head: Optional[ListNode], cnt: int) -> Optional[ListNode]:
        p1 = head
        p2 = head
        # p1 先走 k 步
        for i in range(cnt):
            p1 = p1.next

        # p1 和 p2 同时走 n - k 步
        while p1 != None:
            p2 = p2.next
            p1 = p1.next
        # p2 现在指向第 n - k + 1 个节点，即倒数第 k 个节点
        return p2

#环形链表
class Solution:
    def hasCycle(self, head: Optional[ListNode]) -> bool:
        if head is None:
            return False
  
        # 快慢指针初始化指向 head
        slow, fast = head, head
        # 快指针走到末尾时停止
        while fast.next:
            # 慢指针走一步，快指针走两步
            if fast.next.next:
                fast = fast.next.next
            else:
                return False
            slow = slow.next
            # 快慢指针相遇，说明含有环
            if slow == fast:
                return True
        # 不包含环
        return False
    
    """
    另一种简洁的写法是 while p and p.next
    奇数个node时相同,偶数个node时不同,一个指针停在None,一个停在最后一个node
    对于需要两步两步前进,都推荐使用while p and p.next作为循环条件
    """

from typing import List
import heapq

ListNode.__lt__ = lambda a, b: a.val < b.val  #使得能够用<比较两个ListNode
class Solution:
    def mergeKLists(self, lists: List[Optional[ListNode]]) -> Optional[ListNode]:
        if not lists:
            return None
        # 虚拟头结点
        dummy = ListNode(-1)
        p = dummy
        # 优先级队列，最小堆
        pq = []
        for head in lists:
            if head:
                heapq.heappush(pq, (head.val, head))

        while pq:
            # 获取最小节点，接到结果链表中
            node = heapq.heappop(pq)[1]  #0是val值 1是节点对象
            p.next = node
            if node.next:
                heapq.heappush(pq, (node.next.val, node.next))
            # p 指针不断前进
            p = p.next
        return dummy.next

#进阶 寻找环的起点
class Solution:
    def detectCycle(self,head: ListNode) -> ListNode:
        fast, slow = head, head
        #首先使快慢指针相遇，此时假设走了k和2k步，k必是环长度的倍数
        while fast and fast.next:
            fast = fast.next.next
            slow = slow.next
            if fast == slow:
                break 
        #不存在环
        if not fast or not fast.next:
            return None
        #慢指针从出发开始
        slow = head 
        while slow != fast:
            fast = fast.next
            slow = slow.next
        return slow


#返回两个相交链表的交汇点 
    def getIntersectionNode(self,headA: ListNode, headB: ListNode) -> ListNode: 
        # p1 指向 A 链表头结点，p2 指向 B 链表头结点
        p1, p2 = headA, headB
        while p1 != p2:
            # p1 走一步，如果走到 A 链表末尾，转到 B 链表
            if p1 == None:
                p1 = headB
            else:
                p1 = p1.next
            # p2 走一步，如果走到 B 链表末尾，转到 A 链表
            if p2 == None:
                p2 = headA
            else:
                p2 = p2.next
        return p1

    def getIntersectionNode1_2(self,headA: ListNode, headB: ListNode) -> ListNode: 
        # p1 指向 A 链表头结点，p2 指向 B 链表头结点
        p1, p2 = headA, headB
        while p1 != p2:
            if p1 and p2:
                p1 = p1.next
                p2 = p2.next
            if p1 is None and p2 is not None:
                p1 = headB
            if p1 is not None and p2 is None:
                p2 = headA
            #print(p1.val)
        return p1
    
    def getIntersectionNode1_3(self,headA: ListNode, headB: ListNode) -> ListNode: 
        # p1 指向 A 链表头结点，p2 指向 B 链表头结点
        p1, p2 = headA, headB
        while p1 != p2:
            if p1 and p2:
                p1 = p1.next
                p2 = p2.next
            if p1 is None and p2 is not None:
                p1 = headB
            if p1 is not None and p2 is None:
                p2 = headA
            #print(p1.val)
        return p1
    
    def getIntersectionNode2(self,headA: ListNode, headB: ListNode) -> ListNode: 
        # p1 指向 A 链表头结点，p2 指向 B 链表头结点
        p1, p2 = headA, headB
        while p1.next:
            p1 = p1.next
        p1.next = headB
        res = self.detectCycle(headA)
        p1.next = None
        return res
        


    def getIntersectionNode3(self,headA: ListNode, headB: ListNode) -> ListNode:
        lenA, lenB = 0, 0
        # 计算两条链表的长度
        p1, p2 = headA, headB
        while p1:
            lenA += 1
            p1 = p1.next
        while p2:
            lenB += 1
            p2 = p2.next
        # 让 p1 和 p2 到达尾部的距离相同
        p1, p2 = headA, headB
        if lenA > lenB:
            for i in range(lenA - lenB):
                p1 = p1.next
        else:
            for i in range(lenB - lenA):
                p2 = p2.next
    # 看两个指针是否会相同，p1 == p2 时有两种情况：
    # 1、要么是两条链表不相交，他俩同时走到尾部空指针
    # 2、要么是两条链表相交，他俩走到两条链表的相交点
        while p1 != p2:
            p1 = p1.next
            p2 = p2.next
        return p1

"""
x1 =ListNode(1)
x2 = ListNode(2)
x3  =ListNode(3)
x4 = ListNode(4)
x5 = ListNode(5)
x1.next = x2
x2.next = x3
x3.next = x4
x4.next = x5

y1 = ListNode(-2)
y2 = ListNode(-3)
y1.next = y2
y2.next = x4

solu= Solution()
res = solu.getIntersectionNode2(x1,y1)
print(res.val)
"""
class Solution:
    def reverseList1(self, head: Optional[ListNode]) -> Optional[ListNode]:
        #迭代法——双指针法
        pre = None
        cur = head

        while cur is not None:
            tmp = cur.next
            cur.next = pre
            pre = cur
            cur = tmp
        return pre
    
    def reverseList2_1(self, head: Optional[ListNode]) -> Optional[ListNode]:
        #递归法 实现1
        if head is None or head.next is None:
            return head

        last = self.reverseList2_1(head.next)
        head.next.next = head
        head.next = None
        return last

    def reverseList2_2(self, head: Optional[ListNode]) -> Optional[ListNode]:
        #递归法 实现2
        def reverse(pre: Optional[ListNode],cur: Optional[ListNode]):
            #input  要逆转的子链表的头节点的前驱节点（可能是None）;要逆转的子链表的头节点
            if cur is None:
                return pre
            tmp = cur.next
            cur.next = pre
            return reverse(cur,tmp)
        return reverse(None,head)
    
    def reverseList2_3(self, head: Optional[ListNode]) -> Optional[ListNode]:
        #递归法 实现3
        def reverse(pre: Optional[ListNode],cur: Optional[ListNode]):
            #input  要逆转的子链表的头节点的前驱节点（可能是None）;要逆转的子链表的头节点
            if cur is None:
                return pre
            last = reverse(cur,cur.next)
            cur.next = pre
            return last
        return reverse(None,head)
    
    #迭代法(双指针)
    def reverseN1(self,head: Optional[ListNode], n: int) -> Optional[ListNode]:
        pre = None
        cur = head
        for i  in range(n):
            tmp = cur.next
            cur.next = pre
            pre = cur
            cur = tmp
        head.next = cur
        return pre

    # 递归法
    def reverseN2(self,head: Optional[ListNode], n: int) -> Optional[ListNode]:
        global post
        if n ==1:
            post = head.next
            return head
        last = self.reverseN2(head.next,n-1)
        head.next.next = head
        head.next = post
        return last
    
    def reverseBetween1(self, head: Optional[ListNode], left: int, right: int) -> Optional[ListNode]:
        if left == 1:
        # 相当于反转前 n 个元素
            return self.reverseN2(head, right)
        else:
            cur = head
            pre = None
            for i in range(left-1):
                pre = cur 
                cur = cur.next
            last = self.reverseN2(cur, right-left+1)
            pre.next = last
            return head
        
    #递归实现
    def reverseBetween2(self, head: Optional[ListNode], left: int, right: int) -> Optional[ListNode]:
        if left == 1:
        # 相当于反转前 n 个元素
            return self.reverseN2(head, right)
        last =  self.reverseBetween2(head.next.left-1,right-1)
        head.next = last
        return head
    
    #K个一组反转链表
    def reverseKGroup1_1(self, head: Optional[ListNode], k: int) -> Optional[ListNode]:
        p = head
        length=0
        while p:
            p = p.next
            length = length+1
        if length< k:
            return head
        res1 = self.reverseN1(head,k)
        pre = None
        cur = res1
        for i in range(k):
            pre = cur
            cur = cur.next
        res2 = self.reverseKGroup1_1(cur,k)
    """
    错误写法：
    def reverseKGroup(self, head: Optional[ListNode], k: int) -> Optional[ListNode]:
        res1 = self.reverseN1(head.k)
        pre = None
        cur = head
        for i in range(k):
            pre = cur
            cur = cur.next
        res2 = self.reverseKGroup(cur,k)

        pre.next = res2
        return res1

        错误原因:没有递归过程中链表节点数考虑不足k个的情况(初始链表是默认长度大于等于k的),即没有递归出口
    """

    #优化——判断是否长度不足k,不需要求链表长度，只需让头节点往后移动k步,看是不是变成None即可，循环次数更少
    def reverseKGroup1_2(self, head: Optional[ListNode], k: int) -> Optional[ListNode]:
        p = head
        length=0
        for i in range(k):
            if p is None:
                return head
            else:
                p = p.next

        res1 = self.reverseN1(head,k)
        pre = None
        cur = res1
        for i in range(k):
            pre = cur
            cur = cur.next
        res2 = self.reverseKGroup1_2(cur,k)
        pre.next = res2
        return res1


class Solution:
    def reverse(self,a:ListNode, b:ListNode) -> ListNode:
        pre, cur, nxt = None, a, a
        # while  终止的条件改一下就行了
        while cur != b:
            nxt = cur.next
            cur.next = pre
            pre = cur
            cur = nxt
        # 返回反转后的头结点
        return pre

    def reverseKGroup(self,head: ListNode, k: int) -> ListNode:
        if not head:
            return None
        # 区间 [a, b) 包含 k 个待反转元素
        a, b = head, head
        for i in range(k):
            # 不足 k 个，不需要反转，base case
            if not b:
                return head
            b = b.next
        # 反转前 k 个元素
        new_head = self.reverse(a, b)
        # 递归反转后续链表并连接起来
        a.next = self.reverseKGroup(b, k)
 
        return new_head