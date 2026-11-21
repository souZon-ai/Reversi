"""
深層強化学習の学習ループ用に、pygameに依存しないオセロの環境(ルール・盤面管理)。

まずは通常ルールにおけるAIを作成する。理由は：
・正解の分かる題材なら、「ランダム相手に勝てるようになったか」で学習の成否を判断できます。
  4色版でいきなり始めると、学習が進まないときにバグかルールの癖かを切り分けられません。
・習ループ、経験再生、Target Network、評価の仕組みは、そのまま4色版で使えます。

盤面は「黒石・白石」ではなく「自分の石・相手の石」という手番視点で扱う設計にしている。
これにより、1つのモデルで黒番・白番どちらの手番でも同じ入力形式で扱える。
"""

import numpy as np

BOARD_SIZE = 8

EMPTY = 0
BLACK = 1
WHITE = -1  

DIRECTIONS = [(-1, 1),  (0, 1),  (1, 1),
              (-1, 0),           (1, 0),
              (-1, -1), (0, -1), (1, -1)]


class OthelloEnv:
    """
    強化学習用のオセロ環境。
    典型的な強化学習のインターフェース(reset / step / legal_actions)を持つ。
    """

    def __init__(self):
        self.board = None
        self.current_player = None  #現在のさし手を指定(BLACK or WHITE)
        self.done = False #両者とも置けない盤面になるとTrue
        self.reset()

    def reset(self): #初期盤面作成。先手はBLACK
        self.board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
        self.board[3][3] = WHITE
        self.board[3][4] = BLACK
        self.board[4][3] = BLACK
        self.board[4][4] = WHITE
        self.current_player = BLACK
        self.done = False
        return self.get_state() #学習に渡すための入力([自分, 相手]の2チャネルを作成)

    def _get_flips(self, row, col, player):
        '''
        置いた場合、ひっくり返す駒の座標を返す
        '''
        if self.board[row][col] != EMPTY: #キャンセル
            return []

        opponent = -player
        flips_total = []

        for dr, dc in DIRECTIONS: #全方位走査
            r, c = row + dr, col + dc
            flips_in_this_direction = []

            while 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and self.board[r][c] == opponent: #相手の色だったら保存
                flips_in_this_direction.append((r, c))
                r += dr
                c += dc

            if flips_in_this_direction and 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and \
                    self.board[r][c] == player:
                flips_total.extend(flips_in_this_direction) #相手の駒が自分の駒にはさまれているならば、保存して返す。

        return flips_total

    def legal_actions(self, player=None): #盤面すべて走査し、おける位置を確かめ、パスかどうかを調べる
        """
        playerが今置ける場所を返す。
        戻り値は行動インデックス(0〜63)のリスト。row*BOARD_SIZE+colで表す。
        """
        if player is None:
            player = self.current_player

        actions = []
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                if self._get_flips(row, col, player): #ひっくり返す情報が返ってきたなら、その場所はおける
                    actions.append(row * BOARD_SIZE + col) #戻り値は行動インデックス(0〜63).盤面の位置ごとにそれぞれ数字を割り振っている
        return actions 

    def step(self, action): #強化深層学習に重要な部分。ここを理解する
        """
        actionは0〜63の整数(row*BOARD_SIZE+col)。
        戻り値: (次の状態, 報酬, 終了フラグ, 追加情報)
        報酬は「勝った/負けた/引き分け」が確定した時にだけ、手を打ったプレイヤー視点で与える
        (+1: 勝ち, -1: 負け, 0: それ以外・引き分け)。中間報酬は今回は与えないシンプルな設計。
        """
        if self.done: #両者とも置けなくなった場合に、エラーを返す
            raise ValueError("ゲームは既に終了しています。reset()を呼んでください。")

        row, col = divmod(action, BOARD_SIZE) #割り算の商と余りを返す変数.インデックスからrow colに戻す. actionがインデックス.
        flips = self._get_flips(row, col, self.current_player) #おいた場合と、ひっくり返る駒を返す
        if not flips:
            raise ValueError(f"({row},{col})は合法手ではありません。")

        mover = self.current_player
        self.board[row][col] = mover #row colに置いた
        for r, c in flips:
            self.board[r][c] = mover #すべてひっくり返す

        # 手番を交代。ただし相手に合法手がなければスキップして自分の番のまま
        self.current_player = -mover
        if not self.legal_actions(self.current_player):
            self.current_player = mover
            if not self.legal_actions(self.current_player):
                # 両者とも置けない → 終了
                self.done = True

        reward = 0.0
        info = {}
        if self.done: #両者とも置けない
            black_count = int(np.sum(self.board == BLACK)) #駒の数をカウントする
            white_count = int(np.sum(self.board == WHITE))
            if black_count > white_count: #勝敗判定
                winner = BLACK
            elif white_count > black_count:
                winner = WHITE
            else:
                winner = EMPTY  # 引き分け
            info["winner"] = winner
            # moverから見た報酬(勝ち+1, 負け-1, 引き分け0)
            if winner == EMPTY: #報酬設定
                reward = 0.0
            elif winner == mover:
                reward = 1.0
            else:
                reward = -1.0

        return self.get_state(), reward, self.done, info #学習に使うチャネル、報酬、終わり判定、どちらが勝ったかの情報

    def get_state(self): #学習に使う(入力に使う)チャネル(色ではなく、[自分, 相手]を作成
        my_stones = (self.board == self.current_player).astype(np.float32) #差し手が変わるたびに、my_stonesも変わる？差し手視点から常に学習している？
        opp_stones = (self.board == -self.current_player).astype(np.float32)
        return np.stack([my_stones, opp_stones], axis=0) #axis = 0に、チャネルを二枚重ねる

    def render_text(self):
        """デバッグ用に盤面をテキストで表示する。"""
        symbols = {EMPTY: "・", BLACK: "黒", WHITE: "白"}
        lines = []
        for row in range(BOARD_SIZE):
            lines.append(" ".join(symbols[self.board[row][col]] for col in range(BOARD_SIZE)))
        return "\n".join(lines)