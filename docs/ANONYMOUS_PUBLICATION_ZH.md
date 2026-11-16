# 匿名发布操作指南

## 先确认要上传哪一份

只上传生成的 `anonymous_release/` 目录内容。不要上传原工作目录，不要推送原 `.git`
历史，也不要把本机 checkpoints、features、模型权重、日志、PDF/图片或编辑器配置一起上传。
发布目录已经包含主实验、审稿补充实验入口、依赖、说明和经过筛选的结果出处。

```bash
cd anonymous_release
python -m reproducibility.verify_release .
```

应当看到 `"passed": true`。若要手动修改发布文件，回到工作目录修改源文件并导出到一个
新的目录，再重新检查；校验器会拒绝在审核之后新增或改变的文件。

## 方案 A：独立匿名 GitHub 账号

这是直接给审稿人 GitHub 链接的方式。**用真实姓名的账号新建仓库，仍然会暴露账号身份。**

1. 准备一个与真实姓名、学校、实验室、个人常用昵称无关的账号和邮箱；头像、简介、网站
   也不要填身份线索。创建时按 GitHub 的实际页面完成验证。
2. 在这个匿名账号下新建一个空仓库，取中性名称。不要 fork 原仓库；不要自动添加 README
   或 license，让远程仓库保持为空。
3. 在终端使用匿名账号完成 GitHub 认证。密码、token、私钥只在本地认证流程中使用，
   不写进代码、仓库地址或聊天。
4. 进入发布目录，运行下面的 helper；把地址换成匿名账号下的真实仓库 HTTPS 地址：

   ```bash
   bash tools/publish_anonymous.sh https://github.com/ANONYMOUS_ACCOUNT/REPOSITORY.git
   ```

   这段 helper 先验证文件与匿名扫描，再初始化全新的 Git 历史，使用 `Anonymous Authors`
   和 `anonymous@example.invalid` 作为提交身份，关闭签名，再推送到你提供的远程地址。
   它会拒绝已有 `.git` 的目录，因此不能误把原作者提交历史带过去。
5. 用退出登录的浏览器窗口打开仓库，核对账号、提交页、README、Actions 日志及下载内容。
   审稿人克隆代码后，按 README 在自己的 CUDA 环境运行；GitHub 的 CPU 检查本身不运行大模型。

仓库链接形式就是 `https://github.com/匿名账号/仓库名`。仓库归属账号仍然可见，匿名程度
取决于这个账号是否与个人身份有关联。提交显示名称匿名不等于账号归属匿名。

## 方案 B：Anonymous GitHub 审稿镜像

也可以使用 https://anonymous.4open.science/ 提供审稿访问链接。其上游项目是
https://github.com/tdurieux/anonymous_github 。

先上传**已经本地清理过的发布目录**，再按服务的当前界面创建匿名镜像、设置替换项和
有效期；不依赖网页替换功能去补救代码里的密钥或个人路径。生成后退出登录检查 README、
文件浏览和下载包中的内容。论文只填服务生成的匿名链接，不填真实账号的原始仓库地址。
第三方服务管理员仍可能知道来源；它不能提供对平台管理员的绝对匿名保证。

如果完全不希望公开仓库账号与个人身份相连，优先先用独立匿名账号保存清理后的代码，再
决定是否增加审稿镜像。不要把敏感凭据交给第三方服务。

## 结果与尚缺材料

发布不等于完成了所有 GPU 实验。`artifacts/` 保存原有证据和未解决项：

- 36.68、48.29 和审稿快照中的 BiPO 60.18，仍需原始运行记录核对。
- Q2 的 fold-0 同设置运行已经提供，尚未在本次整理中跑完 GPU 评测。
- MMsafe 的原始 prompts、判分器和生成记录不在当前项目中，不能冒充已复现历史比例。
- 保留第三方 license 和公开上游来源属于正确的软件归属，不应删除上游作者的版权声明。

内容扫描和清理不能保证绝对匿名。代码风格、模型文件来源、账号行为以及与公开仓库相同的
实现仍可能形成关联；发布前应当检查最终可公开访问的实际页面和下载包。

参考官方资料：

- GitHub commit email: https://docs.github.com/en/account-and-profile/how-tos/email-preferences/setting-your-commit-email-address
- Anonymous GitHub: https://anonymous.4open.science/
- 上游实现与说明: https://github.com/tdurieux/anonymous_github
