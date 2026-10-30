import pygame
import sys

#初期設定.
pygame.init()

CELL_SIZE = 60 #1マスの大きさ.
BOARD_SIZE = 10 #10マス/
WINDOW_SIZE = CELL_SIZE * BOARD_SIZE #全体の大きさ
INFO_HEIGHT = 60

screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE + INFO_HEIGHT))
pygame.display.set_caption("4色オセロ")
font = pygame.font.SysFont("meiryo", 28)

GREEN = (0, 128, 0)
BOARD_LINE = (0, 0, 0)
TEXT_COLOR = (255, 255, 255)
HIGHLIGHT = (255, 255, 0)
INFO_BG = (100, 100, 100)

STONE_WHITE = (255, 255, 255)
STONE_BLACK = (0, 0, 0)
STONE_BLUE = (30, 90, 220)
STONE_RED = (210, 30, 30)

#盤面上のデータ.
EMPTY = 0
WHITE = 1
BLACK = 2
BLUE = 3
RED = 4

STONE_COLOR_MAP = {
    WHITE: STONE_WHITE,
    BLACK: STONE_BLACK,
    BLUE: STONE_BLUE,
    RED: STONE_RED
}
STONE_NAME_MAP = {
    WHITE: "白",
    BLACK: "黒",
    BLUE: "青",
    RED: "赤"
}

#石の所有者の定義
PLAYER_A = "A"
PLAYER_B = "B"

OWNER= {
    WHITE: PLAYER_A,
    BLUE: PLAYER_A,
    BLACK: PLAYER_B,
    RED: PLAYER_B
}

PLAYER_NAME = {
    PLAYER_A: "あなた(白・青)",
    PLAYER_B: "相手(黒・赤)"
}

#手番の順番を記憶
TURN_SEQUENCE = [WHITE, BLACK, BLUE, RED]



board = [[EMPTY for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)] # 8 * 8マス盤面

#初期配置.
board[3][3] = BLUE
board[3][6] = RED
board[4][4] = WHITE
board[4][5] = BLACK
board[5][4] = RED
board[5][5] = BLUE
board[6][3] = BLACK
board[6][6] = WHITE

#どの色か判断
turn_index = 0
game_over = False
#4回パスで終了
pass_streak = 0


DIRECTIONS = [(-1, 1), (0, 1), (1, 1),
              (-1, 0),         (1, 0),
              (-1, -1), (0, -1), (1, -1)]

'''def opponent(player):#相手の石の種類を返すだけ/
    return WHITE_STONE if player == BLACK_STONE else BLACK_STONE
'''

def current_color():
    return TURN_SEQUENCE[turn_index % 4]

def get_flips(row, col, color):#石を置けるかどうかを判定
    if board[row][col] != EMPTY:
        return []#石あったらおけない

    #opp = opponent(player)#相手の石の色.
    flips_total = []

    for dr, dc in DIRECTIONS:
        r, c = row + dr, col + dc #全方位に向かって自分の石を探す
        flips_in_this_direction = []

        #while 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and board[r][c] == opp:
        while 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and board[r][c] != EMPTY and board[r][c] != color:
            flips_in_this_direction.append((r, c))#相手の石がある方向の位置をすべて保存.
            r += dr
            c += dc#同じ色なら、終わり.

        if flips_in_this_direction and 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and board[r][c] != EMPTY and board[r][c] == color:
            flips_total.extend(flips_in_this_direction) #一致する色がその先に存在する場合のみflip_totalに載せる
    return flips_total

def get_valid_moves(color):#おける位置をハイライトするための準備
    valid_moves = {}
    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            flips = get_flips(row, col, color)#すべておけるか確かめる.
            if flips:
                valid_moves[(row, col)] = flips #おける位置を辞書で保存
    return valid_moves

def place_stone(row, col, color, flips):#ひっくり返す作業.
    board[row][col] = color
    for r, c in flips:
        board[r][c] = color

def count_stones():#勝敗判定のカウント作業.
    counts = {PLAYER_A: 0, PLAYER_B: 0}
    for row in board:
        for stone in row:
            if stone != EMPTY:
                counts[OWNER[stone]] += 1
    return counts

    '''black_count = sum(row.count(BLACK_STONE) for row in board)
    white_count = sum(row.count(WHITE_STONE) for row in board)
    return black_count, white_count'''

#背景とグリッドを描画
def draw_board():
    screen.fill(GREEN, (0, 0, WINDOW_SIZE, WINDOW_SIZE))
    for i in range(BOARD_SIZE + 1):
        pygame.draw.line(screen, BLACK, (i * CELL_SIZE, 0), (i * CELL_SIZE, WINDOW_SIZE))
        pygame.draw.line(screen, BLACK, (0, i * CELL_SIZE), (WINDOW_SIZE, i * CELL_SIZE))

#いしの描画.
def draw_stones():
    radius = CELL_SIZE // 2 - 5
    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):#すべての盤面を走査
            stone = board[row][col]
            if stone == EMPTY:
                continue
            center_x = col * CELL_SIZE + CELL_SIZE // 2
            center_y = row * CELL_SIZE + CELL_SIZE // 2
            #color = BLACK if stone == BLACK_STONE else WHITE
            pygame.draw.circle(screen, STONE_COLOR_MAP[stone], (center_x, center_y), radius)
            #pygame.draw.circle(screen, (200, 200, 200), (center_x, center_y), radius, 2)

def draw_valid_moves(valid_moves): #おける位置をハイライト
    for (row, col) in valid_moves.keys():
        center_x = col * CELL_SIZE + CELL_SIZE // 2
        center_y = row * CELL_SIZE + CELL_SIZE // 2
        pygame.draw.circle(screen, HIGHLIGHT, (center_x, center_y), 8)

def draw_info(valid_moves):#下に情報を表示する.
    pygame.draw.rect(screen, INFO_BG, (0, WINDOW_SIZE, WINDOW_SIZE, INFO_HEIGHT))
    counts = count_stones()

    if game_over:
        if counts[PLAYER_A] > counts[PLAYER_B]:
            message = f"あなたの勝ち！ (A: {counts[PLAYER_A]}  B: {counts[PLAYER_B]})"
        elif counts[PLAYER_A] < counts[PLAYER_B]:
            message = f"相手の勝ち！ (A: {counts[PLAYER_A]}  B: {counts[PLAYER_B]})"
        else:
            message = f"引き分け (A: {counts[PLAYER_A]}  B: {counts[PLAYER_B]})"
    else:
        color = current_color()
        color_name = STONE_NAME_MAP[color]
        player_name = PLAYER_NAME[OWNER[color]]
        #turn_name = "黒" if current_turn == BLACK_STONE else "白"
        if valid_moves:
            message = f"{player_name}の番です: {color_name}を置いてください (A: {counts[PLAYER_A]}  B: {counts[PLAYER_B]})"
        else:
            message = f"{color_name}は置ける場所がないためスキップ (A:{counts[PLAYER_A]}  B:{counts[PLAYER_B]})"
    text_surface = font.render(message, True, WHITE)
    screen.blit(text_surface, (10, WINDOW_SIZE + 10))

def advance_turn_with_valid_moves():#おける場所があるかチェック。４回スキップで終了.
    global turn_index, game_over, pass_streak

    while True:
        color = current_color()
        valid_moves = get_valid_moves(color)
        if valid_moves:
            pass_streak = 0
            return valid_moves

        #おけない場合
        pass_streak += 1
        print(f"{STONE_NAME_MAP[color]}は置ける場所がないためスキップします")
        turn_index += 1

        if pass_streak >= 4:
            game_over = True
            return{}


def main():
    global turn_index, game_over

    clock = pygame.time.Clock()
    running = True
    valid_moves = advance_turn_with_valid_moves()#おける位置を取得.

    while running:
        '''
        if not game_over:
            valid_moves = get_valid_moves(current_turn)#おける位置を取得.

            if not valid_moves: #置く場所なし
                opp_moves = get_valid_moves(opponent(current_turn))
                if not opp_moves:#どちらも置けない
                    game_over = True
                else:
                    turn_name = "黒" if current_turn == BLACK_STONE else "白"
                    print(f"{turn_name}は置ける場所がないためパスします")
                    current_turn = opponent(current_turn)
                    valid_moves = get_valid_moves(current_turn)
        '''

        for event in pygame.event.get():#前回ループから今まで起きたイベントをすべて取り出す
            if event.type  == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN and not game_over:
                mouse_x, mouse_y = event.pos
                if mouse_y< WINDOW_SIZE:
                    col = mouse_x // CELL_SIZE
                    row = mouse_y // CELL_SIZE
                    if (row, col) in valid_moves:
                        color = current_color()
                        flips = valid_moves[(row, col)]#ひっくり返る位置
                        place_stone(row, col, color, flips)#石を置いてひっくり返る
                        turn_index += 1
                        if not game_over:
                            valid_moves = advance_turn_with_valid_moves()
                        #current_turn = opponent(current_turn)#相手のターン

        draw_board()
        draw_stones()
        if not game_over:
            draw_valid_moves(valid_moves)#次のターンどこが置けるか
        draw_info(valid_moves)

        pygame.display.flip()#表の画面と裏の画面を入れ替えて、処理を更新
        clock.tick(60)

    pygame.quit()
    sys.exit()#ループ抜けたら終わり

if __name__ == "__main__":
    main()
