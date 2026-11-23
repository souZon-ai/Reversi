"""
dqn_train.py

通常オセロ(8x8)を、DQN(Deep Q-Network)による自己対戦で学習するスクリプト。

  - 1つのネットワークが黒番・白番の両方を打つ(入力が手番視点なので共有できる)
  - 経験再生(Replay Buffer)、Target Network、ε-greedy、合法手マスクを使う
  - 報酬は終局時のみ(勝ち+1 / 負け-1 / 引き分け0)
  - 定期的に「ランダムに打つ相手」との勝率を測り、CSVに記録する
  - 放置運転向けに、チェックポイントの保存と再開(--resume)に対応

使い方:
  python dqn_train.py --smoke      # CPUでの試運転(数分。動作確認用)
  python dqn_train.py              # 本番(既定は5000ゲーム)
  python dqn_train.py --resume     # 前回の続きから再開
"""

import argparse
import csv
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from standard_othello_env import OthelloEnv, BOARD_SIZE, BLACK, WHITE
from standard_cnn_model import OthelloCNN

NUM_ACTIONS = BOARD_SIZE * BOARD_SIZE


# =====================================================================
# 経験再生バッファ
# =====================================================================
class ReplayBuffer:
    """
    過去の1手ぶんのデータ(遷移)を貯めておき、ランダムに取り出して学習に使う。
    直近の対局だけで学習すると偏るので、過去のデータも混ぜるための仕組み。
    """

    '''
    理解の修正
    ・一手毎にバッファに保存、四手毎に学習を行う。
    ・それぞれのデータにおけるQ値と目標値を比較して損失を計算する
    ・目標値はtargets = rewards + (1.0 - dones) * signs * gamma * max_next_qとして計算する'''

    def __init__(self, capacity):
        self.capacity = capacity #経験再生バッファに貯めておく手の上限
        self.states = np.zeros((capacity, 2, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, 2, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        # 次の局面が「同じ側の視点」か「相手側の視点」か(+1: 同じ側が続く, -1: 相手に交代)
        self.signs = np.zeros(capacity, dtype=np.float32)
        # 次の局面での合法手(Trueのマスだけが選べる)
        self.next_masks = np.zeros((capacity, NUM_ACTIONS), dtype=np.bool_)
        self.pos = 0
        self.size = 0

    def add(self, state, action, reward, next_state, done, sign, next_mask): #経験バッファに保存。上限超えると、古いデータから上書き
        i = self.pos
        self.states[i] = state
        self.actions[i] = action
        self.rewards[i] = reward
        self.next_states[i] = next_state
        self.dones[i] = float(done)
        self.signs[i] = sign
        self.next_masks[i] = next_mask
        self.pos = (self.pos + 1) % self.capacity  # 満杯になったら古いものから上書き
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size): #ランダムにバッチサイズ分だけデータを取り出す
        idx = np.random.randint(0, self.size, size=batch_size)
        return (self.states[idx], self.actions[idx], self.rewards[idx],
                self.next_states[idx], self.dones[idx], self.signs[idx],
                self.next_masks[idx])
#class終わり

# =====================================================================
# ゲーム進行まわり(PyTorchに依存しない部分)
# =====================================================================
def legal_mask(env):
    """現在の手番プレイヤーが置けるマスだけTrueの、長さ64の配列を返す。"""
    mask = np.zeros(NUM_ACTIONS, dtype=np.bool_)
    for a in env.legal_actions(): #置けるマスの情報
        mask[a] = True
    return mask #置けるマスをハイライト


def play_self_play_game(env, buffer, choose_action, after_step=None):
    """
    1ゲームを最後まで打ち、1手ごとのデータをバッファに追加する。

    choose_action(env) -> 行動(0〜63)  : 手の選び方(学習中はε-greedy)
    after_step()                        : 1手打つたびに呼ばれる関数(学習ステップ用)
    """
    state = env.reset() #初期盤面セット
    moves = 0
    while not env.done: #どちらか一方が置けるなら、実行
        mover = env.current_player
        action = choose_action(env) #手を選ぶ。手の選び方(学習中はε-greedy)
        next_state, reward, done, _ = env.step(action) ##学習に使うチャネル、報酬、終わり判定、どちらが勝ったかの情報

        # 次の局面が誰の視点か。相手に置ける場所がなく、同じ側が続けて打つ場合は
        # current_playerが替わらないので、符号を反転してはいけない
        sign = 1.0 if env.current_player == mover else -1.0

        buffer.add(state, action, reward, next_state, done, sign, legal_mask(env))
        state = next_state
        moves += 1
        if after_step is not None:
            after_step()
    return moves #試合の合計手数(?)


# =====================================================================
# 手の選び方
# =====================================================================
def select_action(model, env, epsilon, device): #どこに置くかを決める関数
    """
    ε-greedy: 確率epsilonでランダムな合法手、それ以外はQ値が最大の合法手を選ぶ。
    合法手以外は、Q値がどれだけ高くても選ばない(マスク)。
    """
    legal = env.legal_actions() #置けるマスを判定する関数。
    if random.random() < epsilon: #ε-greedyで、ランダムの場合
        return random.choice(legal)

    state = torch.from_numpy(env.get_state()).unsqueeze(0).to(device)  #盤面をテンソルに変換－＞先頭に1をつける.unsqueeze(0)－＞モデルに渡す.to(device) (1, 2, 8, 8)
    model.eval()  # BatchNormを推論モードにする モデルの推論を行う
    with torch.no_grad(): #一時的に勾配計算を無効にする。手を選ぶために計算を行うので、学習には必要ない。
        q = model(state)[0].cpu().numpy()  #計算結果をcpuに渡してからnumpyに変換

    masked = np.full(NUM_ACTIONS, -np.inf, dtype=np.float32) #マス毎にランダムに小さいQ値を与える。
    masked[legal] = q[legal] #置ける位置にだけ正常なＱ値を与える
    return int(np.argmax(masked)) #Ｑ値が最大なマスのインデックスを返す


# =====================================================================
# 学習ステップ(DQNの更新式)
# =====================================================================
def train_step(model, target_model, optimizer, buffer, batch_size, gamma, device):
    (states, actions, rewards, next_states,
     dones, signs, next_masks) = buffer.sample(batch_size) #ランダムにバッチサイズ分だけデータを取り出す

    states = torch.from_numpy(states).to(device) #全てGPU(CPU)に渡す作業
    actions = torch.from_numpy(actions).to(device)
    rewards = torch.from_numpy(rewards).to(device)
    next_states = torch.from_numpy(next_states).to(device)
    dones = torch.from_numpy(dones).to(device)
    signs = torch.from_numpy(signs).to(device)
    next_masks = torch.from_numpy(next_masks).to(device)

    # 現在のネットワークが出した「実際に打った手」のQ値
    model.train() #学習モード
    q_all = model(states) #推論を行う                                       # (B, 64)
    q_sa = q_all.gather(1, actions.unsqueeze(1)).squeeze(1)      # (B,)

    # 目標値: 終局なら報酬そのもの。そうでなければ、次の局面の最大Q値(合法手のみ)を
    # 符号付きで足す。次の局面は相手視点なので符号を反転する(-1)。
    with torch.no_grad():
        next_q = target_model(next_states)                       # (B, 64)
        next_q = next_q.masked_fill(~next_masks, -1e9)           # 非合法手は選ばれないように
        max_next_q = next_q.max(dim=1).values                    # (B,)
        targets = rewards + (1.0 - dones) * signs * gamma * max_next_q #学習時の目標値

    loss = F.smooth_l1_loss(q_sa, targets)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)  # 勾配が暴れすぎないように
    optimizer.step()
    return loss.item()


# =====================================================================
# 評価: ランダムに打つ相手との勝率
# =====================================================================
def evaluate_vs_random(model, device, num_games):
    """
    学習中のAI(ε=0、常に最良手)が、ランダムに打つ相手と対戦したときの
    勝ち・引き分け・負けの割合を返す。黒番・白番を交互に担当する。
    """
    env = OthelloEnv()
    wins = draws = losses = 0
    for g in range(num_games):
        agent_color = BLACK if g % 2 == 0 else WHITE
        env.reset()
        while not env.done:
            if env.current_player == agent_color:
                action = select_action(model, env, 0.0, device)
            else:
                action = random.choice(env.legal_actions())
            env.step(action)

        black = int(np.sum(env.board == BLACK))
        white = int(np.sum(env.board == WHITE))
        margin = (black - white) * agent_color  # AI側から見た石数の差(BLACK=1, WHITE=-1)
        if margin > 0:
            wins += 1
        elif margin < 0:
            losses += 1
        else:
            draws += 1
    return wins / num_games, draws / num_games, losses / num_games


# =====================================================================
# メインの学習ループ
# =====================================================================
def make_model(args, device):
    return OthelloCNN(board_size=BOARD_SIZE, in_channels=2,
                      num_filters=args.filters, num_conv_layers=args.conv_layers).to(device)


def train(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"使用デバイス: {device}")

    os.makedirs(args.out_dir, exist_ok=True)
    latest_path = os.path.join(args.out_dir, "latest.pt")
    best_path = os.path.join(args.out_dir, "best.pt")
    log_path = os.path.join(args.out_dir, "training_log.csv")

    model = make_model(args, device)
    target_model = make_model(args, device)
    target_model.load_state_dict(model.state_dict())
    target_model.eval()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    buffer = ReplayBuffer(args.buffer_size)
    env = OthelloEnv()

    start_game = 0
    train_steps = 0
    best_win_rate = -1.0

    if args.resume and os.path.exists(latest_path):
        ckpt = torch.load(latest_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        target_model.load_state_dict(ckpt["target_model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_game = ckpt["game"]
        train_steps = ckpt["train_steps"]
        best_win_rate = ckpt["best_win_rate"]
        print(f"チェックポイントから再開: {start_game}ゲーム目から")

    counters = {"moves": 0, "train_steps": train_steps}
    recent_losses = []
    t0 = time.time()

    new_log = not os.path.exists(log_path)
    with open(log_path, "a", newline="", encoding="utf-8") as log_file:
        writer = csv.writer(log_file)
        if new_log:
            writer.writerow(["games", "epsilon", "avg_loss", "win_rate",
                             "draw_rate", "loss_rate", "elapsed_sec"])

        def evaluate_and_log(games_done, epsilon):
            nonlocal best_win_rate
            win, draw, lose = evaluate_vs_random(model, device, args.eval_games)
            avg_loss = float(np.mean(recent_losses)) if recent_losses else float("nan")
            recent_losses.clear()
            elapsed = time.time() - t0
            print(f"[{games_done:>6}ゲーム] ε={epsilon:.3f}  loss={avg_loss:.4f}  "
                  f"対ランダム 勝ち={win:.2f} 引分={draw:.2f} 負け={lose:.2f}  "
                  f"経過={elapsed / 60:.1f}分")
            writer.writerow([games_done, f"{epsilon:.4f}", f"{avg_loss:.6f}",
                             f"{win:.4f}", f"{draw:.4f}", f"{lose:.4f}", f"{elapsed:.1f}"])
            log_file.flush()

            torch.save({
                "model": model.state_dict(),
                "target_model": target_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "game": games_done,
                "train_steps": counters["train_steps"],
                "best_win_rate": best_win_rate,
            }, latest_path)
            if win > best_win_rate:
                best_win_rate = win
                torch.save(model.state_dict(), best_path)

        def after_step():
            counters["moves"] += 1
            if buffer.size >= args.learn_start and counters["moves"] % args.train_every == 0:
                loss = train_step(model, target_model, optimizer, buffer,
                                  args.batch_size, args.gamma, device)
                counters["train_steps"] += 1
                recent_losses.append(loss)
                if counters["train_steps"] % args.target_update == 0:
                    target_model.load_state_dict(model.state_dict())

        # 学習を始める前の強さ(ランダム相手に約50%のはず)を記録しておく
        if start_game == 0:
            evaluate_and_log(0, args.eps_start)

        for game in range(start_game, args.games):
            # εを1.0からeps_endまで直線的に減らす(最初は探索、徐々に活用へ)
            frac = min(1.0, game / args.eps_decay_games)
            epsilon = args.eps_start + (args.eps_end - args.eps_start) * frac

            play_self_play_game(
                env, buffer,
                choose_action=lambda e: select_action(model, e, epsilon, device),
                after_step=after_step,
            )

            if (game + 1) % args.eval_every == 0:
                evaluate_and_log(game + 1, epsilon)

    print("学習終了")


def parse_args():
    p = argparse.ArgumentParser(description="オセロのDQN自己対戦学習")
    p.add_argument("--games", type=int, default=5000, help="自己対戦のゲーム数")
    p.add_argument("--eval-every", type=int, default=100, help="何ゲームごとに評価するか")
    p.add_argument("--eval-games", type=int, default=100, help="評価時にランダム相手と戦うゲーム数")
    p.add_argument("--filters", type=int, default=64, help="畳み込みのフィルタ数")
    p.add_argument("--conv-layers", type=int, default=4, help="畳み込み層の数")
    p.add_argument("--buffer-size", type=int, default=50000, help="経験再生バッファの容量")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=5e-4, help="学習率")
    p.add_argument("--gamma", type=float, default=1.0, help="割引率(終局報酬のみなので1.0)")
    p.add_argument("--eps-start", type=float, default=1.0)
    p.add_argument("--eps-end", type=float, default=0.05)
    p.add_argument("--eps-decay-games", type=int, default=None,
                   help="εを減らしきるまでのゲーム数(未指定ならgamesの60%)")
    p.add_argument("--train-every", type=int, default=4, help="何手ごとに1回学習するか")
    p.add_argument("--target-update", type=int, default=1000,
                   help="何回の学習ステップごとにTarget Networkを更新するか")
    p.add_argument("--learn-start", type=int, default=1000,
                   help="バッファにこの数の手が貯まってから学習を始める")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="auto", help="auto / cpu / cuda")
    p.add_argument("--out-dir", type=str, default="checkpoints")
    p.add_argument("--resume", action="store_true", help="latest.ptから再開する")
    p.add_argument("--smoke", action="store_true", help="CPU向けの小さい設定で試運転する")
    args = p.parse_args()

    if args.smoke:
        args.games = 30
        args.eval_every = 10
        args.eval_games = 10
        args.filters = 16
        args.buffer_size = 5000 #経験バッファに貯めておく手の上限
        args.learn_start = 200
        args.target_update = 100
    if args.eps_decay_games is None:
        args.eps_decay_games = max(1, int(args.games * 0.6))
    return args


if __name__ == "__main__":
    train(parse_args())
