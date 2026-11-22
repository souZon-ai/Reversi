"""
cnn_model.py

オセロの盤面(2チャンネル: 自分の石・相手の石)を入力として受け取り、
各マス(0〜63)に置いたときのQ値を出力するCNNモデル。

画像認識で使われるCNNと異なり、盤面ゲームではマスの位置そのものに意味があるため、
プーリング層は使わない(位置情報を保ったまま処理する)。
"""

import torch
import torch.nn as nn


class OthelloCNN(nn.Module):
    def __init__(self, board_size=8, in_channels=2, num_filters=64, num_conv_layers=4):
        super().__init__() #モデルの初期化など
        self.board_size = board_size #オセロのボードサイズ

        # ===== 畳み込み層(プーリングなし) =====
        conv_layers = []
        channels = in_channels #差し手と相手の２チャネル
        for _ in range(num_conv_layers): #４層の畳み込み層を経る。これは、3->5->7->9となり、盤面全体を見渡せる数。
            conv_layers.append(nn.Conv2d(channels, num_filters, kernel_size=3, padding=1)) #入力チャネル数、出力チャネル数、3*3の畳み込み、padding行うので、結果も8*8で出力される。
            conv_layers.append(nn.BatchNorm2d(num_filters)) #各チャネルで、平均０，分散１に整え、学習が安定しやすくする。
            conv_layers.append(nn.ReLU(inplace=True)) #活性化関数
            channels = num_filters #2->64->64->64
        self.conv_layers = nn.Sequential(*conv_layers) #順伝播させる

        # ===== 出力層(全結合) =====
        # 畳み込みの出力(num_filters × board_size × board_size)を1次元に潰してから、
        # 各マスに対応するQ値(board_size*board_size個)を出力する
        flat_size = num_filters * board_size * board_size #演算結果を一次元に直すイメージ
        self.fc = nn.Sequential(
            nn.Linear(flat_size, 256), #全結合で、256まで圧縮
            nn.ReLU(inplace=True),
            nn.Linear(256, board_size * board_size), #ます一つ一つに値を与える
        )

    def forward(self, x):
        """
        x: (batch, 2, board_size, board_size) のテンソル
        戻り値: (batch, board_size*board_size) の各マスに対するQ値
        """
        features = self.conv_layers(x) #モデルに入力、畳み込みの出力を得る
        features = features.view(features.size(0), -1)  # 1次元に潰す(flatten)、もともとbatch, channel, 8, 8の四次元。これをbatch, 4096の二次元に直す。-1は、残りを自動で計算する引数
        q_values = self.fc(features) #最後の全結合によって出力結果を受けtる
        return q_values


if __name__ == "__main__":
    # 簡易動作確認: ダミーの入力を1つ流してみて、出力の形が正しいか確認する
    model = OthelloCNN(board_size=8)
    dummy_input = torch.zeros((1, 2, 8, 8))  # バッチサイズ1、2チャンネル、8x8盤面
    output = model(dummy_input)
    print("出力の形:", output.shape, " (1, 64)になっているはず")
    print("パラメータ数:", sum(p.numel() for p in model.parameters()))