# jiangsier-gobang
English | [中文版](https://github.com/jiangsier-xyz/jiangsier-gobang/blob/main/README.zh-CN.md)

## Introduction
jiangsier-gobang is a reinforcement-learning-based AI Gomoku game. It supports multiple AI players (each using different model structures). These AI players can learn through self-play or playing against each other.

## Training
You can start from scratch using the Monte Carlo tree for training. Alternatively, you can use game records from approximately 5000 human games (in Smart Game Format) for training to obtain better initial parameters.

It supports multiple AI players. Model complexity:
Ace < Fox < Baker < Casey < Darling < Ellis

Special players:
- **Gill**: Combines the predictions from Ace, Baker, Casey, Darling, and Fox with different weights. The weight settings are based on the results of games played among the AI players. Tests show that its playing strength is higher than that of individual model AI player.
- **Monkey**: Uses random moves as priors and employs the Monte Carlo tree to calculate the next move.
- **Teacher**: A role based on strategy.

## Snapshots
### On Console
![Console Snapshot](/data/images/console.jpg)
### On Web
![Web Snapshot 1](/data/images/web_1.jpg)
![Web Snapshot 2](/data/images/web_2.jpg)
