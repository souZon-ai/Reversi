import pygame
import sys

#初期設定.
pygame.init()

CELL_SIZE = 75 #1マスの大きさ.
BOARD_SIZE = 8 #８マス/
WINDOW_SIZE = CELL_SIZE * BOARD_SIZE #全体の大きさ
INFO_HEIGHT = 50

screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE + INFO_HEIGHT))
pygame.display.set_caption("オセロ")
font = pygame.font.SysFont("msgothic", 36)

GREEN = (0, 128, 0)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
HIGHLIGHT = (255, 255, 0)
INFO_BG = (30, 30, 30)

#盤面上のデータ.
EMPTY = 0
BLACK_STONE = 1
WHITE_STONE = 2

board = [[EMPTY for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)] # 8 * 8マス盤面

#初期配置.
board[3][3] = WHITE_STONE
board[3][4] = BLACK_STONE
board[4][3] = BLACK_STONE
board[4][4] = WHITE_STONE

current_turn = BLACK_STONE #現在の手番
game_over = False

DIRECTIONS = [(-1, 1), (0, 1), (1, 1),
              (-1, 0),         (1, 0),
              (-1, -1), (0, -1), (1, -1)]

def opponent(player):#相手の石の種類を返すだけ/
    return WHITE_STONE if player == BLACK_STONE else BLACK_STONE

def get_flips(row, col, player):#石を置けるかどうかを判定
    if board[row][col]!= EMPTY:
        return []#石あったらおけない

    opp = opponent(player)#相手の石の色.
    flips_total = []

    for dr, dc in DIRECTIONS:
        r, c = row + dr, col + dc #全方位に向かって自分の石を探す
        flips_in_this_direction = []

        while 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and board[r][c] == opp:
            flips_in_this_direction.append((r, c))#相手の石がある方向の位置をすべて保存.
            r += dr
            c += dc#相手の色以外なら、終わり.

        if flips_in_this_direction and 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and board[r][c] == player:
            flips_total.extend(flips_in_this_direction)
    return flips_total

def get_valid_moves(player):#おける位置をハイライトするための準備
    valid_moves = {}
    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            flips = get_flips(row, col, player)#すべておけるか確かめる.
            if flips:
                valid_moves[(row, col)] = flips
    return valid_moves

def place_stone(row, col, player, flips):#ひっくり返す作業.
    board[row][col] = player
    for r, c in flips:
        board[r][c] = player

def count_stones():#勝敗判定時のカウント作業.
    black_count = sum(row.count(BLACK_STONE) for row in board)
    white_count = sum(row.count(WHITE_STONE) for row in board)
    return black_count, white_count

#背景とグリッドを描画
def draw_board():
    screen.fill(GREEN, (0, 0, WINDOW_SIZE, WINDOW_SIZE))
    for i in range(BOARD_SIZE + 1):
        pygame.draw.line(screen, BLACK, (i * CELL_SIZE, 0), (i * CELL_SIZE, WINDOW_SIZE))
        pygame.draw.line(screen, BLACK, (0, i * CELL_SIZE), (WINDOW_SIZE, i * CELL_SIZE))

#医師の描画.
def draw_stones():
    radius = CELL_SIZE // 2 - 5
    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):#すべての盤面を走査
            stone = board[row][col]
            if stone == EMPTY:
                continue
            center_x = col * CELL_SIZE + CELL_SIZE // 2
            center_y = row * CELL_SIZE + CELL_SIZE // 2
            color = BLACK if stone == BLACK_STONE else WHITE
            pygame.draw.circle(screen, color, (center_x, center_y), radius)

def draw_valid_moves(valid_moves): #おける位置をハイライト
    for (row, col) in valid_moves.keys():
        center_x = col * CELL_SIZE + CELL_SIZE // 2
        center_y = row * CELL_SIZE + CELL_SIZE // 2
        pygame.draw.circle(screen, HIGHLIGHT, (center_x, center_y), 8)

def draw_info():#下に情報を表示する.
    pygame.draw.rect(screen, INFO_BG, (0, WINDOW_SIZE, WINDOW_SIZE, INFO_HEIGHT))
    black_count, white_count = count_stones()

    if game_over:
        if black_count > white_count:
            message = f"黒の勝ち！ (黒: {black_count}  白: {white_count})"
        elif black_count < white_count:
            message = f"白の勝ち！ (黒: {black_count}  白: {white_count})"
        else:
            message = f"引き分け (黒: {black_count}  白: {white_count})"
    else:
        turn_name = "黒" if current_turn == BLACK_STONE else "白"
        message = f"{turn_name}の番です (黒: {black_count}  白: {white_count})"
    text_surface = font.render(message, True, WHITE)
    screen.blit(text_surface, (10, WINDOW_SIZE + 10))

def main():
    global current_turn, game_over

    clock = pygame.time.Clock()
    running = True
    valid_moves = get_valid_moves(current_turn)#おける位置を取得.

    while running:
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

        for event in pygame.event.get():#前回ループから今まで起きたイベントをすべて取り出す
            if event.type  == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mouse_x, mouse_y = event.pos
                if mouse_y< WINDOW_SIZE:
                    col = mouse_x // CELL_SIZE
                    row = mouse_y // CELL_SIZE
                    if (row, col) in valid_moves:
                        flips = valid_moves[(row, col)]#ひっくり返る位置
                        place_stone(row, col, current_turn, flips)#石を置いてひっくり返る
                        current_turn = opponent(current_turn)#相手のターン

        draw_board()
        draw_stones()
        if not game_over:
            draw_valid_moves(valid_moves)#次のターンどこが置けるか
        draw_info()

        pygame.display.flip()#表の画面と裏の画面を入れ替えて、処理を更新
        clock.tick(60)

    pygame.quit()
    sys.exit()#ループ抜けたら終わり

if __name__ == "__main__":
    main()