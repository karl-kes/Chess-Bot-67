# chess_bot.py
# Single-file chess bot. Exposes move(state, player).
# Pure Python, no external libraries.

import time
from collections import defaultdict

# === Configuration ===
MAX_DEPTH_LIMIT = 6
TIME_LIMIT_PER_MOVE = 0.95  # seconds (leave tiny safety margin)
QUIESCENCE_MAX = 20

# Piece values (white pieces are lowercase per contest spec; black uppercase).
# Positive score favors White.
PIECE_VALUES = {
    'p': 100, 'n': 320, 'b': 330, 'r': 500, 'q': 900, 'k': 20000,
    'P': -100, 'N': -320, 'B': -330, 'R': -500, 'Q': -900, 'K': -20000
}

# Piece-square tables (oriented rank1..rank8).
PST = {
    'P': [
        [0,  0,  0,  0,  0,  0,  0,  0],
        [50, 50, 50, 50, 50, 50, 50, 50],
        [10, 10, 20, 30, 30, 20, 10, 10],
        [5,  5, 10, 25, 25, 10,  5,  5],
        [0,  0,  0, 20, 20,  0,  0,  0],
        [5, -5,-10,  0,  0,-10, -5,  5],
        [5, 10, 10,-20,-20, 10, 10,  5],
        [0,  0,  0,  0,  0,  0,  0,  0]
    ],
    'N': [
        [-50,-40,-30,-30,-30,-30,-40,-50],
        [-40,-20,  0,  0,  0,  0,-20,-40],
        [-30,  0, 10, 15, 15, 10,  0,-30],
        [-30,  5, 15, 20, 20, 15,  5,-30],
        [-30,  0, 15, 20, 20, 15,  0,-30],
        [-30,  5, 10, 15, 15, 10,  5,-30],
        [-40,-20,  0,  5,  5,  0,-20,-40],
        [-50,-40,-30,-30,-30,-30,-40,-50]
    ],
    'B': [
        [-20,-10,-10,-10,-10,-10,-10,-20],
        [-10,  0,  0,  0,  0,  0,  0,-10],
        [-10,  0,  5, 10, 10,  5,  0,-10],
        [-10,  5,  5, 10, 10,  5,  5,-10],
        [-10,  0, 10, 10, 10, 10,  0,-10],
        [-10, 10, 10, 10, 10, 10, 10,-10],
        [-10,  5,  0,  0,  0,  0,  5,-10],
        [-20,-10,-10,-10,-10,-10,-10,-20]
    ],
    'R': [
        [0,  0,  0,  0,  0,  0,  0,  0],
        [5, 10, 10, 10, 10, 10, 10,  5],
        [-5,  0,  0,  0,  0,  0,  0, -5],
        [-5,  0,  0,  0,  0,  0,  0, -5],
        [-5,  0,  0,  0,  0,  0,  0, -5],
        [-5,  0,  0,  0,  0,  0,  0, -5],
        [-5,  0,  0,  0,  0,  0,  0, -5],
        [0,  0,  0,  5,  5,  0,  0,  0]
    ],
    'Q': [
        [-20,-10,-10, -5, -5,-10,-10,-20],
        [-10,  0,  0,  0,  0,  0,  0,-10],
        [-10,  0,  5,  5,  5,  5,  0,-10],
        [-5,  0,  5,  5,  5,  5,  0, -5],
        [0,  0,  5,  5,  5,  5,  0, -5],
        [-10,  5,  5,  5,  5,  5,  0,-10],
        [-10,  0,  5,  0,  0,  0,  0,-10],
        [-20,-10,-10, -5, -5,-10,-10,-20]
    ],
    'K': [
        [-30,-40,-40,-50,-50,-40,-40,-30],
        [-30,-40,-40,-50,-50,-40,-40,-30],
        [-30,-40,-40,-50,-50,-40,-40,-30],
        [-30,-40,-40,-50,-50,-40,-40,-30],
        [-20,-30,-30,-40,-40,-30,-30,-20],
        [-10,-20,-20,-20,-20,-20,-20,-10],
        [20, 20,  0,  0,  0,  0, 20, 20],
        [20, 30, 10,  0,  0, 10, 30, 20]
    ],
    'K_endgame': [
        [-50,-40,-30,-20,-20,-30,-40,-50],
        [-30,-20,-10,  0,  0,-10,-20,-30],
        [-30,-10, 20, 30, 30, 20,-10,-30],
        [-30,-10, 30, 40, 40, 30,-10,-30],
        [-30,-10, 30, 40, 40, 30,-10,-30],
        [-30,-10, 20, 30, 30, 20,-10,-30],
        [-30,-30,  0,  0,  0,  0,-30,-30],
        [-50,-30,-30,-30,-30,-30,-30,-50]
    ]
}

# === Helpers ===

def board_to_key(board, state):
    """Return a compact string key for transposition: board + castling + prev_move"""
    rows = [''.join(r) for r in board]
    key = '/'.join(rows)
    flags = ''.join(['1' if state.get(x, False) else '0' for x in ('E1K','E8K','A1R','A8R','H1R','H8R')])
    prev = state.get('prev_move') or ''
    return key + ' ' + flags + ' ' + prev

def clone_board(board):
    return [row[:] for row in board]

def inside(r, c):
    return 0 <= r < 8 and 0 <= c < 8

# === Move generation (largely reused but corrected for piece-case assumptions) ===

def generate_legal_moves(board, player, state):
    moves = []
    is_white = (player == 1)
    for row in range(8):
        for col in range(8):
            piece = board[row][col]
            if piece == '.':
                continue
            # skip opponent pieces
            if is_white and piece.isupper():
                continue
            if (not is_white) and piece.islower():
                continue
            t = piece.upper()
            if t == 'P':
                moves.extend(generate_pawn_moves(board, row, col, is_white, state))
            elif t == 'N':
                moves.extend(generate_knight_moves(board, row, col, is_white))
            elif t == 'B':
                moves.extend(generate_bishop_moves(board, row, col, is_white))
            elif t == 'R':
                moves.extend(generate_rook_moves(board, row, col, is_white))
            elif t == 'Q':
                moves.extend(generate_queen_moves(board, row, col, is_white))
            elif t == 'K':
                moves.extend(generate_king_moves(board, row, col, is_white, state))
    moves.extend(generate_castling_moves(board, is_white, state))
    # filter moves leaving king in check
    legal = []
    for m in moves:
        nb_board, nb_state = make_move(board, m, state)
        if not is_in_check(nb_board, is_white):
            legal.append(m)
    return legal

def generate_pawn_moves(board, row, col, is_white, state):
    moves = []
    direction = -1 if is_white else 1
    start_row = 6 if is_white else 1
    promotion_row = 0 if is_white else 7
    from_sq = chr(col + ord('a')) + str(8 - row)
    # one step
    nr = row + direction
    if inside(nr, col) and board[nr][col] == '.':
        to_sq = chr(col + ord('a')) + str(8 - nr)
        if nr == promotion_row:
            for promo in ['Q','R','B','N']:
                moves.append(f"P{from_sq}{to_sq}={promo}")
        else:
            moves.append(f"P{from_sq}{to_sq}")
        # two steps
        if row == start_row:
            nr2 = row + 2*direction
            if inside(nr2, col) and board[nr2][col] == '.':
                to_sq2 = chr(col + ord('a')) + str(8 - nr2)
                moves.append(f"P{from_sq}{to_sq2}")
    # captures
    for dc in (-1,1):
        nc = col + dc
        nr = row + direction
        if inside(nr, nc):
            target = board[nr][nc]
            if target != '.':
                # target must be opponent
                if (is_white and target.isupper()) or (not is_white and target.islower()):
                    to_sq = chr(nc + ord('a')) + str(8 - nr)
                    if nr == promotion_row:
                        for promo in ['Q','R','B','N']:
                            moves.append(f"P{from_sq}{to_sq}={promo}")
                    else:
                        moves.append(f"P{from_sq}{to_sq}")
    # en passant
    prev = state.get('prev_move')
    if prev and prev[0] == 'P' and len(prev) >= 5:
        # prev is like Pe7e5
        prev_from_file = ord(prev[1]) - ord('a')
        prev_from_rank = int(prev[2])
        prev_to_file = ord(prev[3]) - ord('a')
        prev_to_rank = int(prev[4])
        if abs(prev_from_rank - prev_to_rank) == 2:
            # pawn moved two squares last move
            if (is_white and row == 3) or (not is_white and row == 4):
                if abs(col - prev_to_file) == 1:
                    ep_row = row + direction
                    ep_col = prev_to_file
                    to_sq = chr(ep_col + ord('a')) + str(8 - ep_row)
                    moves.append(f"P{from_sq}{to_sq}")
    return moves

def generate_knight_moves(board, row, col, is_white):
    moves = []
    knight_moves = [(-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1)]
    from_sq = chr(col + ord('a')) + str(8 - row)
    for dr, dc in knight_moves:
        nr, nc = row + dr, col + dc
        if not inside(nr, nc):
            continue
        target = board[nr][nc]
        if target == '.':
            moves.append(f"N{from_sq}{chr(nc + ord('a'))}{8 - nr}")
        else:
            # capture if opponent
            if (is_white and target.isupper()) or (not is_white and target.islower()):
                moves.append(f"N{from_sq}{chr(nc + ord('a'))}{8 - nr}")
    return moves

def generate_sliding_moves(board, row, col, directions, is_white, pc):
    moves = []
    from_sq = chr(col + ord('a')) + str(8 - row)
    for dr, dc in directions:
        nr, nc = row + dr, col + dc
        while inside(nr, nc):
            target = board[nr][nc]
            to_sq = f"{chr(nc + ord('a'))}{8 - nr}"
            if target == '.':
                moves.append(f"{pc}{from_sq}{to_sq}")
            else:
                if (is_white and target.isupper()) or (not is_white and target.islower()):
                    moves.append(f"{pc}{from_sq}{to_sq}")
                break
            nr += dr; nc += dc
    return moves

def generate_bishop_moves(board, row, col, is_white):
    return generate_sliding_moves(board, row, col, [(-1,-1),(-1,1),(1,-1),(1,1)], is_white, 'B')

def generate_rook_moves(board, row, col, is_white):
    return generate_sliding_moves(board, row, col, [(-1,0),(1,0),(0,-1),(0,1)], is_white, 'R')

def generate_queen_moves(board, row, col, is_white):
    return generate_sliding_moves(board, row, col, [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)], is_white, 'Q')

def generate_king_moves(board, row, col, is_white, state):
    moves = []
    king_moves = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
    from_sq = chr(col + ord('a')) + str(8 - row)
    for dr, dc in king_moves:
        nr, nc = row + dr, col + dc
        if inside(nr, nc):
            target = board[nr][nc]
            if target == '.':
                moves.append(f"K{from_sq}{chr(nc + ord('a'))}{8 - nr}")
            else:
                if (is_white and target.isupper()) or (not is_white and target.islower()):
                    moves.append(f"K{from_sq}{chr(nc + ord('a'))}{8 - nr}")
    return moves

def generate_castling_moves(board, is_white, state):
    moves = []
    if is_white:
        # White king at row 7, col 4 (e1)
        if not state.get('E1K', False):
            # kingside
            if not state.get('H1R', False) and board[7][5]=='.' and board[7][6]=='.':
                if not is_square_attacked(board, 7, 4, False) and not is_square_attacked(board,7,5,False) and not is_square_attacked(board,7,6,False):
                    moves.append('O-O')
            # queenside
            if not state.get('A1R', False) and board[7][1]=='.' and board[7][2]=='.' and board[7][3]=='.':
                if not is_square_attacked(board,7,4,False) and not is_square_attacked(board,7,3,False) and not is_square_attacked(board,7,2,False):
                    moves.append('O-O-O')
    else:
        # Black king at row 0, col 4
        if not state.get('E8K', False):
            if not state.get('H8R', False) and board[0][5]=='.' and board[0][6]=='.':
                if not is_square_attacked(board,0,4,True) and not is_square_attacked(board,0,5,True) and not is_square_attacked(board,0,6,True):
                    moves.append('O-O')
            if not state.get('A8R', False) and board[0][1]=='.' and board[0][2]=='.' and board[0][3]=='.':
                if not is_square_attacked(board,0,4,True) and not is_square_attacked(board,0,3,True) and not is_square_attacked(board,0,2,True):
                    moves.append('O-O-O')
    return moves

# === Attack / check helpers ===

def is_square_attacked(board, row, col, by_white):
    # pawn attacks
    pawn_dir = -1 if by_white else 1
    for dc in (-1,1):
        rr = row - pawn_dir
        cc = col + dc
        if inside(rr, cc):
            piece = board[rr][cc]
            if piece == ('p' if by_white else 'P'):
                return True
    # knights
    knight_moves = [(-2,-1),(-2,1),(-1,-2),(-1,2),(1,-2),(1,2),(2,-1),(2,1)]
    for dr, dc in knight_moves:
        rr, cc = row + dr, col + dc
        if inside(rr, cc):
            piece = board[rr][cc]
            if piece == ('n' if by_white else 'N'):
                return True
    # diagonal (bishop/queen)
    for dr, dc in [(-1,-1),(-1,1),(1,-1),(1,1)]:
        rr, cc = row+dr, col+dc
        while inside(rr, cc):
            piece = board[rr][cc]
            if piece != '.':
                if by_white and piece in ('b','q'):
                    return True
                if (not by_white) and piece in ('B','Q'):
                    return True
                break
            rr += dr; cc += dc
    # straight (rook/queen)
    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
        rr, cc = row+dr, col+dc
        while inside(rr, cc):
            piece = board[rr][cc]
            if piece != '.':
                if by_white and piece in ('r','q'):
                    return True
                if (not by_white) and piece in ('R','Q'):
                    return True
                break
            rr += dr; cc += dc
    # king
    for dr in (-1,0,1):
        for dc in (-1,0,1):
            if dr==0 and dc==0:
                continue
            rr, cc = row+dr, col+dc
            if inside(rr, cc):
                piece = board[rr][cc]
                if piece == ('k' if by_white else 'K'):
                    return True
    return False

def is_in_check(board, is_white):
    king = 'k' if is_white else 'K'
    for r in range(8):
        for c in range(8):
            if board[r][c] == king:
                return is_square_attacked(board, r, c, not is_white)
    return False

# === Make move ===

def make_move(board, move_str, state):
    """
    Return (new_board, new_state) after applying move_str to board given state.
    new_state is a shallow copy of state with updated prev_move and basic castling flags.
    """
    new_board = clone_board(board)
    new_state = dict(state)  # shallow copy
    # default behavior: set prev_move
    new_state['prev_move'] = move_str

    # utility: parse from/to squares for normal moves TxyXY or promotions
    if move_str == 'O-O':
        # determine side by king piece on board
        # white
        if board[7][4] == 'k':
            new_board[7][4] = '.'
            new_board[7][6] = 'k'
            new_board[7][7] = '.'
            new_board[7][5] = 'r'
            # update castling rights
            new_state['E1K'] = True
            new_state['H1R'] = True
        else:
            # black
            new_board[0][4] = '.'
            new_board[0][6] = 'K'
            new_board[0][7] = '.'
            new_board[0][5] = 'R'
            new_state['E8K'] = True
            new_state['H8R'] = True
    elif move_str == 'O-O-O':
        if board[7][4] == 'k':
            new_board[7][4] = '.'
            new_board[7][2] = 'k'
            new_board[7][0] = '.'
            new_board[7][3] = 'r'
            new_state['E1K'] = True
            new_state['A1R'] = True
        else:
            new_board[0][4] = '.'
            new_board[0][2] = 'K'
            new_board[0][0] = '.'
            new_board[0][3] = 'R'
            new_state['E8K'] = True
            new_state['A8R'] = True
    else:
        # TxyXY or TxyXY=t
        piece_type = move_str[0]
        from_file = ord(move_str[1]) - ord('a')
        from_rank = 8 - int(move_str[2])
        to_file = ord(move_str[3]) - ord('a')
        to_rank = 8 - int(move_str[4])
        piece = board[from_rank][from_file]
        # remove from origin
        new_board[from_rank][from_file] = '.'

        # en passant: pawn moved diagonally to empty square (capture the pawn behind)
        if piece_type == 'P' and board[to_rank][to_file] == '.' and from_file != to_file:
            # determine which pawn is captured based on moving color
            if piece == 'p':  # white pawn capturing black
                # black pawn will be just below destination square (to_rank+1)
                new_board[to_rank+1][to_file] = '.'
            else:
                new_board[to_rank-1][to_file] = '.'

        # promotion
        if '=' in move_str:
            promo_piece = move_str[-1]
            # keep case consistent with moving piece
            new_board[to_rank][to_file] = promo_piece.lower() if piece.islower() else promo_piece.upper()
        else:
            # normal capture or move
            new_board[to_rank][to_file] = piece

        # Update castling rights: if king or rook moves, or rook is captured on original squares
        # detect white pieces (lowercase) and black (uppercase)
        # If a king moved from e1 or e8
        if piece.lower() == 'k':
            if piece.islower():  # white king moved
                new_state['E1K'] = True
            else:
                new_state['E8K'] = True
        # If a rook moved from one of starting squares:
        if piece.lower() == 'r':
            if piece.islower():
                if from_rank == 7 and from_file == 0:
                    new_state['A1R'] = True
                if from_rank == 7 and from_file == 7:
                    new_state['H1R'] = True
            else:
                if from_rank == 0 and from_file == 0:
                    new_state['A8R'] = True
                if from_rank == 0 and from_file == 7:
                    new_state['H8R'] = True

        # If a rook was captured on its starting square, update rights
        # examine original destination square on the original board
        captured = board[to_rank][to_file]
        if captured.lower() == 'r':
            if captured.islower():
                # white rook captured on a1/h1?
                if to_rank == 7 and to_file == 0:
                    new_state['A1R'] = True
                if to_rank == 7 and to_file == 7:
                    new_state['H1R'] = True
            else:
                if to_rank == 0 and to_file == 0:
                    new_state['A8R'] = True
                if to_rank == 0 and to_file == 7:
                    new_state['H8R'] = True

    # optionally update history and hdict to aid repetition detection (lightweight)
    # shallow-copy history list to avoid mutating original
    hist = state.get('history', None)
    if hist is not None:
        new_hist = hist[:]  # shallow copy
        new_hist.append(new_board)
        new_state['history'] = new_hist
    # prev_move already set
    return new_board, new_state

# === Evaluation ===

def evaluate_board(board, state):
    score = 0
    piece_count = 0
    # material + pst
    for r in range(8):
        for c in range(8):
            p = board[r][c]
            if p == '.':
                continue
            piece_count += 1
            score += PIECE_VALUES.get(p, 0)
            t = p.upper()
            if t in PST:
                # table indexed rank1..rank8; board uses 0=rank8,...,7=rank1
                # for white piece (lowercase) add table[7-r][c]; for black piece subtract table[r][c]
                if p.islower():  # white
                    table = PST[t]
                    score += table[7-r][c]
                else:  # black
                    table = PST[t]
                    score -= table[r][c]
    # mobility (lightweight)
    white_moves = len(generate_legal_moves(board, 1, state))
    black_moves = len(generate_legal_moves(board, 2, state))
    score += (white_moves - black_moves) * 10
    return score

# === Move ordering ===

def mvv_lva_score(board, move):
    # simple Most Valuable Victim - Least Valuable Attacker for captures
    if move in ('O-O','O-O-O'):
        return 1000000
    if len(move) >= 5:
        tf = ord(move[3]) - ord('a'); tr = 8 - int(move[4])
        victim = board[tr][tf]
        if victim != '.':
            victim_val = abs(PIECE_VALUES.get(victim, 0))
            attacker_file = ord(move[1]) - ord('a'); attacker_rank = 8 - int(move[2])
            attacker = board[attacker_rank][attacker_file]
            attacker_val = abs(PIECE_VALUES.get(attacker, 0))
            return 100000 + victim_val*100 - attacker_val
    # center and promotion prioritization
    score = 0
    if '=' in move:
        score += 90000
    if len(move) >= 5:
        tf = ord(move[3]) - ord('a'); tr = 8 - int(move[4])
        if 2 <= tf <= 5 and 2 <= tr <= 5:
            score += 50
    return score

# === Quiescence search for captures ===

def generate_captures(board, player, state):
    # returns only moves that capture (or promotions)
    caps = []
    all_moves = generate_legal_moves(board, player, state)
    for m in all_moves:
        if m in ('O-O','O-O-O'):
            continue
        if '=' in m:
            caps.append(m)
            continue
        to_file = ord(m[3]) - ord('a'); to_rank = 8 - int(m[4])
        if board[to_rank][to_file] != '.':
            caps.append(m)
    return caps

def quiescence(board, alpha, beta, player, state, start_time):
    # Stand pat
    stand = evaluate_board(board, state)
    if time.time() - start_time > TIME_LIMIT_PER_MOVE:
        return stand
    if stand >= beta:
        return beta
    if alpha < stand:
        alpha = stand
    # search captures
    captures = generate_captures(board, player, state)
    # order captures by MVV-LVA
    captures.sort(key=lambda m: mvv_lva_score(board, m), reverse=True)
    cnt = 0
    for mv in captures:
        if time.time() - start_time > TIME_LIMIT_PER_MOVE:
            break
        cnt += 1
        if cnt > QUIESCENCE_MAX:
            break
        nb_board, nb_state = make_move(board, mv, state)
        val = -quiescence(nb_board, -beta, -alpha, 3-player, nb_state, start_time)
        if val >= beta:
            return beta
        if val > alpha:
            alpha = val
    return alpha

# === Transposition table ===

class TTEntry:
    def __init__(self, depth, score, flag, best):
        self.depth = depth
        self.score = score
        self.flag = flag  # 'EXACT','LOWER','UPPER'
        self.best = best

trans_table = {}

# === Negamax with alpha-beta, iterative deepening wrapper ===

def negamax(board, depth, alpha, beta, player, state, start_time):
    # time check
    if time.time() - start_time > TIME_LIMIT_PER_MOVE:
        return evaluate_board(board, state)
    key = board_to_key(board, state)
    tt = trans_table.get(key)
    if tt and tt.depth >= depth:
        if tt.flag == 'EXACT':
            return tt.score
        elif tt.flag == 'LOWER':
            alpha = max(alpha, tt.score)
        elif tt.flag == 'UPPER':
            beta = min(beta, tt.score)
        if alpha >= beta:
            return tt.score
    if depth == 0:
        return quiescence(board, alpha, beta, player, state, start_time)
    best_score = -10**9
    best_move = None
    moves = generate_legal_moves(board, player, state)
    if not moves:
        # terminal: checkmate or stalemate
        if is_in_check(board, player==1):
            return -999999  # mate-ish
        return 0
    # move ordering
    moves.sort(key=lambda m: mvv_lva_score(board, m), reverse=True)

    original_alpha = alpha
    for m in moves:
        if time.time() - start_time > TIME_LIMIT_PER_MOVE:
            break
        nb_board, nb_state = make_move(board, m, state)
        val = -negamax(nb_board, depth-1, -beta, -alpha, 3-player, nb_state, start_time)
        if val > best_score:
            best_score = val
            best_move = m
        alpha = max(alpha, val)
        if alpha >= beta:
            break
    # store in tt (use original alpha/beta)
    flag = 'EXACT'
    if best_score <= original_alpha:
        flag = 'UPPER'
    elif best_score >= beta:
        flag = 'LOWER'
    trans_table[key] = TTEntry(depth, best_score, flag, best_move)
    return best_score

# === Top-level move function ===

def order_moves_root(moves, board):
    return sorted(moves, key=lambda m: mvv_lva_score(board, m), reverse=True)

def move(state, player):
    """
    Main entrypoint used by the judge.
    state: dict described in prompt
    player: 1 (white) or 2 (black)
    """
    start_time = time.time()
    board = state["board"]
    legal = generate_legal_moves(board, player, state)
    if not legal:
        return None
    if len(legal) == 1:
        return legal[0]
    # iterative deepening
    best_move = legal[0]
    best_score = -10**9
    max_d = 1
    # root ordering
    legal = order_moves_root(legal, board)
    # iterative deepening loop
    while max_d <= MAX_DEPTH_LIMIT and (time.time() - start_time) < TIME_LIMIT_PER_MOVE:
        try:
            alpha = -10**9
            beta = 10**9
            current_best = None
            current_best_score = -10**9
            for mv in legal:
                if time.time() - start_time > TIME_LIMIT_PER_MOVE:
                    break
                nb_board, nb_state = make_move(board, mv, state)
                score = -negamax(nb_board, max_d-1, -beta, -alpha, 3-player, nb_state, start_time)
                if score > current_best_score:
                    current_best_score = score
                    current_best = mv
                if score > alpha:
                    alpha = score
            if current_best is not None:
                best_move = current_best
                best_score = current_best_score
        except Exception:
            break
        max_d += 1
    return best_move

# If run as a script, you can test with a simple initial position.
if __name__ == "__main__":
    # small smoke test: starting board from prompt (white lowercase)
    board = [
        ["R","N","B","Q","K","B","N","R"],
        ["P","P","P","P","P","P","P","P"],
        [".",".",".",".",".",".",".","."],
        [".",".",".",".",".",".",".","."],
        [".",".",".",".",".",".",".","."],
        [".",".",".",".",".",".",".","."],
        [".",".",".",".",".",".",".","."],
        ["r","n","b","q","k","b","n","r"]
    ]
    st = {"board": board, "E1K": False, "E8K": False, "A1R": False, "A8R": False, "H1R": False, "H8R": False, "prev_move": None, "history": [board], "hdict": {}, "movelist": []}
    print("Bot picks:", move(st, 1))