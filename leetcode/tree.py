# -*- coding: utf-8 -*-
"""
Created on Sun Sep 17 22:52:17 2023

@author: myf
"""
from typing import List
import collections

# 节点代码
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

# 树生成代码
def generate_tree(vals):
    if len(vals) == 0:
        return None
    que = [] # 定义队列
    fill_left = True # 由于无法通过是否为 None 来判断该节点的左儿子是否可以填充，用一个记号判断是否需要填充左节点
    for val in vals:
        node = TreeNode(val) if val is not None else None # 非空值返回节点类，否则返回 None
        if len(que)==0:
            root = node # 队列为空的话，用 root 记录根结点，用来返回
            que.append(node)
        elif fill_left:
            que[0].left = node
            fill_left = False # 填充过左儿子后，改变记号状态
            if node: # 非 None 值才进入队列
                que.append(node)
        else:
            que[0].right = node
            if node:
                que.append(node)
            que.pop(0) # 填充完右儿子，弹出节点
            fill_left = True # 
    return root




# 定义一个bfs打印层序遍历
def bfs(node):
    que = []
    que.append(node)
    while que:
        l = len(que)
        for _ in range(l):
            tmp = que.pop(0)
            print(tmp.val, end=' ')
            if tmp.left:
                que.append(tmp.left)
            if tmp.right:
                que.append(tmp.right)
        print('|', end=' ')
        

class Solution:
    def buildTree(self, preorder: List[int], inorder: List[int]) -> TreeNode:
        def build(preorder: List[int], preStart: int, preEnd: int,inorder: List[int], inStart: int, inEnd: int) -> TreeNode:
            if preStart>preEnd or inStart>inEnd:
                return None
            root = TreeNode(val=preorder[preStart])
            
            index = 0
            for i in range(inStart,inEnd+1):
                if (inorder[i]==preorder[preStart]):
                    index = i
                    break
            leftsize= index-inStart
            root.left =  build(preorder, preStart+1, preStart+leftsize,inorder, inStart, index-1)
            root.right = build(preorder, preStart+leftsize+1, preEnd,inorder, index+1, inEnd)
            return root
        n = len(preorder)
        return build(preorder,0,n-1,inorder,0,n-1)

def preorder_tree(root:TreeNode)->List:
    result = []
    #print(root.val)
    if root==None:return
    else:
        result.append(root.val)
    if root.left !=None:
        result.extend(preorder_tree(root.left))
    if root.right !=None:
        result.extend(preorder_tree(root.right))
    return result

# 树生成代码-递归(输入需是前序遍历)
def Creat_Tree(vals):
    if len(vals)==0:#终止条件：val用完了
        return None
    if vals[0] is not None:   #本层需要干的就是构建Root、Root.lchild、Root.rchild三个节点。
        Root = TreeNode(vals[0])
        vals.pop(0)
        Root.left = Creat_Tree(vals)
        Root.right = Creat_Tree(vals)
        return Root#本次递归要返回给上一次的本层构造好的树的根节点
    else:
        Root=None
        vals.pop(0)
    return Root#本次递归要返回给上一次的本层构造好的树的根节点

def inorder_tree(root:TreeNode)->List:
    result = []
    if root==None:
        return
    if root.left !=None:
        result.extend(inorder_tree(root.left))
    result.append(root.val)   
    if root.right !=None:
        result.extend(inorder_tree(root.right))
        
    return result

def posorder_tree(root:TreeNode)->List:
    result = []
    if root==None:
        return
    
    if root.left !=None:
        result.extend(posorder_tree(root.left))   
    if root.right !=None:
        result.extend(posorder_tree(root.right))
    result.append(root.val)

    return result


def levelOrder(root:TreeNode) -> List[List[int]]:
    if not root: 
        return []
    res, queue = [],[]
    queue.append(root)
    while queue:
        tmp = []
        for _ in range(len(queue)):
            node = queue.pop(0)
            tmp.append(node.val)
            if node.left: 
                queue.append(node.left)
            if node.right: 
                queue.append(node.right)
        res.append(tmp)  #append输出形式为[[3], [9, 20], [15, 7], [8]]   extend输出形式为[3, 9, 20, 15, 7, 8]
    return res

def mylevelOrder(root:TreeNode) -> List[List[int]]:
    if not root: 
        return []
    res, queue = [],[]
    queue.append(root)
    while queue:
        temp = []
        length = len(queue)
        for i in range(length):
            if queue[0].left:
                queue.append(queue[0].left)
            if queue[0].right:
                queue.append(queue[0].right)
            temp.append(queue.pop(0).val)
        res.append(temp)
    
    return res



#深度优先搜索(迭代实现)  
def dfs1(node):
    if not node:
        return []
    res, stack,visited = [],[],[]
    res.append(node.val)
    stack.append(node)
    visited.append(node)
    node_tmp = node
    while stack: 
        node_tmp = stack[-1]
        if node_tmp.left and node_tmp.left not in visited:
            res.append(node_tmp.left.val) 
            stack.append(node_tmp.left)
            visited.append(node_tmp.left) 

        elif node_tmp.right and node_tmp.right not in visited:
            res.append(node_tmp.right.val) 
            stack.append(node_tmp.right)
            visited.append(node_tmp.right)

        else:
            node_tmp = stack.pop()
    return res


#深度优先搜索(迭代实现)
# 先遍历右边
def dfs2(node):
    if not node:
        return []
    res, stack,visited = [],[],[]
    stack.append(node)
    visited.append(node)
    
    while stack: 
        node_tmp = stack.pop()
        res.append(node_tmp.val)
        if node_tmp.left and node_tmp.left not in visited:
            #res.append(node_tmp.left.val) 
            stack.append(node_tmp.left)
            visited.append(node_tmp.left) 

        if node_tmp.right and node_tmp.right not in visited:
            #res.append(node_tmp.right.val) 
            stack.append(node_tmp.right)
            visited.append(node_tmp.right)
    return res
            

#深度优先搜索(递归实现)
def dfs_recursive (node,res=[]):
    stack,visited = [],[]
    res.append(node.val)
    stack.append(node)
    visited.append(node)
    while stack:
        node_tmp = stack.pop()
        if node_tmp.left and node_tmp.left not in visited:
            dfs_recursive(node_tmp.left,res)
        if node_tmp.right and node_tmp.right not in visited:
            dfs_recursive(node_tmp.right,res)   
    return res

""" 
solution = Solution()
preorder = [3,9,8,7,20,6]
inorder = [7,8,9,20,3,6]

result  =  solution.buildTree(preorder, inorder)
#result_list1 = dfs1(result)
result_list2 = posorder_tree(result)
#print(result_list1)
print(result_list2)
"""

node = Creat_Tree([3,9,20,None,None,15,7])
print(levelOrder(node))