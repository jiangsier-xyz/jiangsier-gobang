# jiangsier-gobang

[English](https://github.com/jiangsier-xyz/jiangsier-gobang/blob/main/README.md) | 中文版

## 介绍
jiangsier-gobang 是一个基于强化学习的五子棋人工智能游戏。它支持多个 AI 玩家（每个使用不同的模型结构）。这些 AI 玩家可以通过自我对弈或相互对战来学习。

## 训练
您可以使用蒙特卡洛树从头开始进行训练。另外，您也可以使用约 5000 局人类对局的棋谱（Smart Game Format 格式）进行训练，以获得更好的初始参数。

它支持多个AI玩家。模型复杂度：
Ace < Fox < Baker < Casey < Darling < Ellis

特殊玩家：
- **Gill**：由 Ace、Baker、Casey、Darling、Fox 以不同权重投票决定。权重设置参考了 AI 玩家之间相互对弈的结果。测试表明，其游戏实力高于单模型的 AI 玩家。
- **Monkey**：使用随机移动作为先验，并使用蒙特卡洛树计算下一步。
- **Teacher**：基于策略的角色。

## 快照
### 控制台
![Console Snapshot](/data/images/console.jpg)
### 网页
![Web Snapshot 1](/data/images/web_1.jpg)
![Web Snapshot 2](/data/images/web_2.jpg)