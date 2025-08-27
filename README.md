# MGM_Mask2Former

## github设置

### 设置github代理

```bash
git config --global https.proxy https://127.0.0.1:7890
git config --global http.proxy http://127.0.0.1:7890
```

### 设置github用户名和邮箱

```bash
git config --global user.name "YOUR_NAME"
git config --global user.email "YOUR_EMAIL"
```

### 关联github仓库

#### 如果本地仓库没有git初始化，初始化，关联仓库

```bash
git init
git remote add origin https://github.com/BUAAZhangHaonan/mgm_mask2former.git
```

#### 如果本地仓库已经git初始化，删除已有.git文件夹，重新初始化，关联仓库

```bash
cd workspace
rm -rf ./mask2former/.git
cd ..
git remote add origin https://github.com/BUAAZhangHaonan/mgm_mask2former.git
```

#### 新建git分支

```bash
git checkout -b 分支名
```

### 从远程拉取代码仓库到本地

#### 将本地代码更改保存，解决冲突

```bash
git add .
git commit -m "save current changes"
git pull --no-rebase origin master # <-要修改pull的分支
```

之后在文档管理器中有冲突的文件会标红显示，在文档管理器中进行merge以完成不同版本commit的合并

#### 合并完成后保存文件，将合并后代码上传

```bash
git add .
git commit -m "commit message"
```

### 如果想让本地分支的内容与远程分支同步（用远程完全覆盖本地）

```bash
git fetch origin
git reset --hard origin/master
```

### 配置中科大镜像源

```bash
python -m pip install --upgrade pip
pip config set global.index-url https://mirrors.ustc.edu.cn/pypi/web/simple/
```

## 提交类型说明

| 类型     | 描述                                                   |
| -------- | ------------------------------------------------------ |
| feat     | 新功能（feature）                                      |
| fix      | 修复 bug                                               |
| docs     | 文档变更                                               |
| style    | 代码格式（不影响功能，如空格、分号等）                 |
| refactor | 代码重构（既不是修复 bug 也不是添加新功能）            |
| perf     | 性能优化                                               |
| test     | 添加或修改测试用例                                     |
| build    | 构建相关（打包、CI 工作流等）                          |
| ci       | 持续集成配置修改                                       |
| chore    | 其他修改（不影响源代码或测试，如构建脚本、依赖更新等） |
| revert   | 回滚到上一个版本                                       |