### 1.删除文件

在Git中，删除也是一个修改操作

实战一下，先添加一个新文件`test.txt`到Git并且提交：

```
$ git add test.txt
```

一般情况下，你通常直接在文件管理器中把没用的文件删了，或者用`rm`命令删了：

```
$ rm test.txt
```

这个时候，Git知道你删除了文件，因此，工作区和版本库就不一致了，`git status`命令会立刻告诉你哪些文件被删除了：

```
$ git status
```

现在你有两个选择，一是确实要从版本库中删除该文件，那就用命令`git rm`删掉，并且`git commit`：

```
$ git rm test.txt
rm 'test.txt'

$ git commit -m "remove test.txt"
[master d46f35e] remove test.txt
 1 file changed, 1 deletion(-)
 delete mode 100644 test.txt
```



### 2.分支管理

首先，我们创建`dev`分支，然后切换到`dev`分支：

```
$ git checkout -b dev
Switched to a new branch 'dev'
```

`git checkout`命令加上`-b`参数表示创建并切换，相当于以下两条命令：

```
$ git branch dev
$ git checkout dev
Switched to branch 'dev'
```

然后，用`git branch`命令查看当前分支：

```
$ git branch
* dev
  master
```

`git branch`命令会列出所有分支，当前分支前面会标一个`*`号。

然后，我们就可以在`dev`分支上正常提交，比如对`readme.txt`做个修改，加上一行：

```
Creating a new branch is quick.
```

然后提交：

```
$ git add readme.txt 
$ git commit -m "branch test"
[dev b17d20e] branch test
 1 file changed, 1 insertion(+)
```

现在，`dev`分支的工作完成，我们就可以切换回`master`分支：

```
$ git checkout master
Switched to branch 'master'
```

切换回`master`分支后，再查看一个`readme.txt`文件，刚才添加的内容不见了！因为那个提交是在`dev`分支上，而`master`分支此刻的提交点并没有变。

![git-br-on-master](https://www.liaoxuefeng.com/files/attachments/919022533080576/0)







### 3.git 拉取不同分支



##### git拉取代码最常用的方式为：

```bash
git clone http://gitslab.yiqing.com/declare/about.git
```

这种方式没有指定分支，当代码有多个分支时，拉取的分支不一定就是master

##### 使用git拉代码时可以使用 -b 指定分支

指定拉 master 分支代码

```bash
git clone -b master http://gitslab.yiqing.com/declare/about.git
```

##### 查看当前项目拉的是哪个分支的代码：

进入项目根目录， 然后执行 git branch 命令

git branch
![img](https://img-blog.csdnimg.cn/20200131172839235.png)

##### 查看分支上的递交情况:

进入项目根目录，执行 git show-branch

```bash
git show-branch
```


