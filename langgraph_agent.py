"""
langgraph_agent.py
==================
LangGraph 기반 3-FSM Baba Is You 에이전트

설계 구조
─────────
                  ┌─────────────┐
            ┌────▶│  규칙 FSM   │◀────┐
            │     │ SCAN→EMIT   │     │ re-scan (BLOCKED)
            │     └──────┬──────┘     │
            │            │ rule_context│
            │     ┌──────▼──────┐     │
            │     │  목표 FSM   │     │
            │     │ EVAL→GOAL   │─────┘
            │     └──────┬──────┘
            │            │ target_pos
            │     ┌──────▼──────┐
            │     │  공간 FSM   │
            │     │ IDLE→MOVE   │
            │     └──────┬──────┘
            │            │ 이동 완료
            └────────────┘  (매 스텝 규칙 재확인)

FSM 노드 목록
─────────────
규칙 FSM : scan_rules → parse_rules → emit_context
목표 FSM : evaluate_goal → (break_rule | make_rule | find_target)
공간 FSM : check_bounds → moving | blocked → (목표 재평가)

공유 상태 (AgentState / TypedDict)
──────────────────────────────────
  grid_map, active_rules, rule_blocks_pos,
  agent_pos, current_goal, target_pos,
  blocked_dirs, move_history, win_achieved,
  loop_count, fsm_phase, done
"""

from __future__ import annotations

import os
import heapq
from typing import Any, Dict, List, Optional, Set, Tuple, TypedDict

import numpy as np

# LangGraph
try:
    from langgraph.graph import StateGraph, END
except ImportError:
    raise ImportError(
        "LangGraph가 설치되지 않았습니다.\n"
        "  pip install langgraph\n"
        "을 실행하세요."
    )

from baba.llm_converter import StateConverter


# ══════════════════════════════════════════════════════════════════════════════
# 공유 상태 정의
# ══════════════════════════════════════════════════════════════════════════════

class AgentState(TypedDict):
    # ── 환경 스냅샷 ──────────────────────────────────────────────────────────
    env: Any                             # BabaIsYouEnv 참조 (직렬화 불필요)
    grid_width: int
    grid_height: int
    agent_pos: Tuple[int, int]           # (x, y)

    # ── 규칙 FSM 출력 ────────────────────────────────────────────────────────
    active_rules: List[Dict]             # extract_ruleset 결과의 _rule_ 리스트
    rule_blocks_pos: Dict[Tuple, str]    # {(x,y): block_name}
    win_objects: List[str]               # WIN 속성 오브젝트 타입명
    stop_objects: List[str]              # STOP 속성 오브젝트 타입명
    you_objects: List[str]               # YOU 속성 오브젝트 타입명

    # ── 목표 FSM 출력 ────────────────────────────────────────────────────────
    current_goal: str                    # "BREAK" | "MAKE" | "FIND" | "NONE"
    target_pos: Optional[Tuple[int, int]]
    target_desc: str                     # 사람이 읽는 목표 설명

    # ── 공간 FSM ─────────────────────────────────────────────────────────────
    blocked_dirs: List[str]              # ["up","down","left","right"] 부분집합
    move_history: List[str]             # 최근 이동 방향 기록

    # ── 제어 ─────────────────────────────────────────────────────────────────
    win_achieved: bool
    loop_count: int
    fsm_phase: str                       # 현재 어느 FSM 노드에 있는지 (디버그)
    done: bool
    pending_action: Optional[str]        # "up"|"down"|"left"|"right" — 다음 실행할 액션


# 방향 → (dx, dy)
DIR_VEC: Dict[str, Tuple[int, int]] = {
    "up":    (0, -1),
    "down":  (0,  1),
    "left":  (-1, 0),
    "right": (1,  0),
}
ALL_DIRS = list(DIR_VEC.keys())


# ══════════════════════════════════════════════════════════════════════════════
# 유틸: 그리드 파싱 / 경로 탐색
# ══════════════════════════════════════════════════════════════════════════════

def _parse_env(env) -> Dict:
    """env에서 필요한 정보를 모두 추출해 dict로 반환."""
    from baba.world_object import RuleBlock, FlexibleWorldObj, Wall, name_mapping
    from baba.rule import extract_ruleset

    grid  = env.grid
    w, h  = grid.width, grid.height
    apos  = tuple(env.agent_pos)

    rule_blocks: Dict[Tuple, str] = {}
    objects: Dict[Tuple, str]     = {}  # {(x,y): obj_type}

    for j in range(h):
        for i in range(w):
            cell = grid.get(i, j)
            if cell is None:
                continue
            if isinstance(cell, RuleBlock):
                rule_blocks[(i, j)] = cell.name
            elif isinstance(cell, (FlexibleWorldObj, Wall)):
                objects[(i, j)] = cell.type

    # 활성 규칙 추출
    ruleset     = extract_ruleset(grid)
    rules_list  = ruleset.get("_rule_", [])

    win_objects  = [k for k, v in ruleset.get("is_goal",   {}).items() if v is True]
    stop_objects = [k for k, v in ruleset.get("is_stop",   {}).items() if v is True]
    you_objects  = [k for k, v in ruleset.get("is_agent",  {}).items() if v is True]

    return {
        "rule_blocks_pos": rule_blocks,
        "active_rules":    rules_list,
        "win_objects":     win_objects,
        "stop_objects":    stop_objects,
        "you_objects":     you_objects,
        "objects":         objects,
        "agent_pos":       apos,
        "grid_width":      w,
        "grid_height":     h,
    }


def _is_wall_or_stop(env, x: int, y: int, stop_objects: List[str]) -> bool:
    """(x, y) 셀이 이동 불가(외벽/STOP 오브젝트)인지 확인."""
    grid = env.grid
    if x < 0 or y < 0 or x >= grid.width or y >= grid.height:
        return True
    cell = grid.get(x, y)
    if cell is None:
        return False
    # 외벽
    from baba.world_object import Wall
    if type(cell).__name__ == "Wall" and not hasattr(cell, "_ruleset"):
        return True
    # STOP 속성
    if hasattr(cell, "type") and cell.type in stop_objects:
        return True
    return False


def _bfs(
    env,
    start: Tuple[int, int],
    goal: Tuple[int, int],
    stop_objects: List[str],
) -> Optional[List[str]]:
    """
    BFS로 start → goal 최단 경로(방향 문자열 리스트)를 반환.
    경로가 없으면 None.
    인접 RuleBlock 셀은 PUSH 가능하므로 통과 가능으로 간주.
    """
    from collections import deque
    from baba.world_object import RuleBlock

    grid = env.grid
    w, h = grid.width, grid.height

    visited = {start}
    queue   = deque([(start, [])])

    while queue:
        pos, path = queue.popleft()
        if pos == goal:
            return path

        for dname, (dx, dy) in DIR_VEC.items():
            nx, ny = pos[0] + dx, pos[1] + dy
            if (nx, ny) in visited:
                continue
            if nx < 0 or ny < 0 or nx >= w or ny >= h:
                continue

            cell = grid.get(nx, ny)
            # 외벽 경계는 막힘
            from baba.world_object import Wall
            if isinstance(cell, Wall) and not hasattr(cell, "_ruleset"):
                continue
            # STOP 오브젝트는 막힘 (단, goal이라면 통과)
            if cell is not None and hasattr(cell, "type"):
                if cell.type in stop_objects and (nx, ny) != goal:
                    continue

            visited.add((nx, ny))
            queue.append(((nx, ny), path + [dname]))

    return None  # 경로 없음


def _find_object_positions(env, obj_types: List[str]) -> List[Tuple[int, int]]:
    """env 그리드에서 해당 타입의 오브젝트 위치 목록 반환."""
    from baba.world_object import FlexibleWorldObj
    grid = env.grid
    positions = []
    for j in range(grid.height):
        for i in range(grid.width):
            cell = grid.get(i, j)
            if cell and hasattr(cell, "type") and cell.type in obj_types:
                positions.append((i, j))
    return positions


def _find_rule_triple_positions(
    env,
    rule_blocks_pos: Dict[Tuple, str],
) -> List[Tuple[Tuple, str]]:
    """
    현재 규칙 트리플 [SUBJECT]-[IS]-[PROPERTY]을 형성하는 블록 중
    이동시켜 규칙을 완성할 수 있는 IS 블록 위치 탐색.
    (MAKE_RULE 목표: IS 블록 중 좌우 또는 상하에 SUBJECT, PROPERTY 가 없는 것)
    """
    # IS 블록 위치 수집
    is_positions = [(pos, name) for pos, name in rule_blocks_pos.items() if name == "is"]
    return is_positions


# ══════════════════════════════════════════════════════════════════════════════
# FSM 노드 구현
# ══════════════════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────────────────────
# 규칙 FSM
# ─────────────────────────────────────────────────────────────────────────────

def node_scan_rules(state: AgentState) -> AgentState:
    """
    규칙 FSM — SCAN_RULES
    그리드를 전수 순회해 RuleBlock 위치 수집 + 환경 스냅샷 갱신.
    """
    env  = state["env"]
    info = _parse_env(env)

    print(f"  [규칙FSM] SCAN  win={info['win_objects']}  stop={info['stop_objects']}")

    return {
        **state,
        "fsm_phase":       "SCAN_RULES",
        "rule_blocks_pos": info["rule_blocks_pos"],
        "active_rules":    info["active_rules"],
        "win_objects":     info["win_objects"],
        "stop_objects":    info["stop_objects"],
        "you_objects":     info["you_objects"],
        "agent_pos":       info["agent_pos"],
        "grid_width":      info["grid_width"],
        "grid_height":     info["grid_height"],
    }


def node_emit_context(state: AgentState) -> AgentState:
    """
    규칙 FSM — EMIT_CONTEXT
    규칙 컨텍스트를 확정하고 목표 FSM으로 넘김.
    win_achieved 여부도 여기서 최종 체크.
    """
    env         = state["env"]
    win_objects = state["win_objects"]

    # WIN 오브젝트가 agent 위치와 겹치면 승리
    apos = state["agent_pos"]
    win_achieved = False
    if win_objects:
        from baba.world_object import FlexibleWorldObj
        cell = env.grid.get(*apos)
        if cell and hasattr(cell, "type") and cell.type in win_objects:
            win_achieved = True

    print(f"  [규칙FSM] EMIT  win_achieved={win_achieved}")

    return {
        **state,
        "fsm_phase":    "EMIT_CONTEXT",
        "win_achieved": win_achieved,
        "done":         win_achieved,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 목표 FSM
# ─────────────────────────────────────────────────────────────────────────────

def node_evaluate_goal(state: AgentState) -> AgentState:
    """
    목표 FSM — EVALUATE
    우선순위: BREAK(STOP 장애) → MAKE(WIN 조건 불완전) → FIND(WIN 대상 이동)
    """
    env         = state["env"]
    win_objects = state["win_objects"]
    stop_objects = state["stop_objects"]
    agent_pos   = state["agent_pos"]

    # ── FIND: WIN 조건이 완성 + 이동 가능 경로 존재 ──────────────────────────
    if win_objects:
        win_positions = _find_object_positions(env, win_objects)
        for wpos in win_positions:
            path = _bfs(env, agent_pos, wpos, stop_objects)
            if path is not None:
                print(f"  [목표FSM] EVALUATE → FIND  target={wpos}  path_len={len(path)}")
                return {
                    **state,
                    "fsm_phase":   "EVALUATE",
                    "current_goal": "FIND",
                    "target_pos":  wpos,
                    "target_desc": f"WIN 오브젝트({win_objects[0]}) @ {wpos} 로 이동",
                }
        # WIN 오브젝트가 있지만 경로가 STOP 에 막힘 → BREAK
        if win_positions and stop_objects:
            print(f"  [목표FSM] EVALUATE → BREAK  (STOP이 WIN 경로 차단)")
            return {
                **state,
                "fsm_phase":    "EVALUATE",
                "current_goal": "BREAK",
                "target_pos":   None,
                "target_desc":  f"STOP 규칙 제거 (막힌 오브젝트: {stop_objects})",
            }

    # ── MAKE: WIN 조건 없음 → 규칙 생성 필요 ────────────────────────────────
    if not win_objects:
        print(f"  [목표FSM] EVALUATE → MAKE  (WIN 규칙 없음)")
        return {
            **state,
            "fsm_phase":    "EVALUATE",
            "current_goal": "MAKE",
            "target_pos":   None,
            "target_desc":  "WIN 규칙 생성 (IS 블록 정렬)",
        }

    # ── BREAK: STOP이 경로 차단 ──────────────────────────────────────────────
    if stop_objects:
        print(f"  [목표FSM] EVALUATE → BREAK  (STOP={stop_objects})")
        return {
            **state,
            "fsm_phase":    "EVALUATE",
            "current_goal": "BREAK",
            "target_pos":   None,
            "target_desc":  f"STOP 규칙 제거: {stop_objects}",
        }

    # 여기 도달하면 WIN 조건 있지만 WIN 오브젝트가 그리드에 없음
    print(f"  [목표FSM] EVALUATE → NONE  (WIN 오브젝트가 그리드에 없음)")
    return {
        **state,
        "fsm_phase":    "EVALUATE",
        "current_goal": "NONE",
        "target_pos":   None,
        "target_desc":  "목표 없음 (WIN 오브젝트 미존재)",
    }


def _find_breakable_stop_block(state: AgentState) -> Optional[Tuple[int, int]]:
    """
    BREAK 목표: STOP 규칙을 형성하는 RuleBlock 중 Baba가 밀 수 있는 것을 탐색.
    IS 블록 또는 STOP(is_stop) 블록을 밀 수 있는 위치 반환.
    """
    rule_blocks = state["rule_blocks_pos"]
    agent_pos   = state["agent_pos"]
    stop_objects = state["stop_objects"]

    # 규칙 트리플에서 STOP 관련 블록 찾기
    # 형태: [OBJECT IS stop] — "stop" 블록 또는 "is" 블록을 미는 것으로 파괴
    candidates = []
    for (x, y), name in rule_blocks.items():
        if name in ("stop", "is"):
            candidates.append((x, y))

    # 에이전트에서 가장 가까운 것 반환
    if not candidates:
        return None
    candidates.sort(key=lambda p: abs(p[0]-agent_pos[0]) + abs(p[1]-agent_pos[1]))

    # 해당 블록에 인접한 셀 (밀기 위해 도달해야 할 위치)
    for cpos in candidates:
        for dname, (dx, dy) in DIR_VEC.items():
            push_from = (cpos[0] - dx, cpos[1] - dy)  # 블록 맞은편에서 밀기
            return push_from  # 첫 번째 유효 후보 반환

    return None


def node_break_rule(state: AgentState) -> AgentState:
    """
    목표 FSM — BREAK_RULE
    STOP 규칙을 형성하는 블록을 밀기 위해 이동 대상 설정.
    'is' 블록 또는 'stop' property 블록에 인접하는 것을 목표로.
    """
    env          = state["env"]
    agent_pos    = state["agent_pos"]
    rule_blocks  = state["rule_blocks_pos"]
    stop_objects = state["stop_objects"]

    # STOP 규칙 관련 블록(is / stop) 중 에이전트와 가장 가까운 것 탐색
    best_target = None
    best_dist   = float("inf")

    for (x, y), name in rule_blocks.items():
        if name in ("stop", "is"):
            dist = abs(x - agent_pos[0]) + abs(y - agent_pos[1])
            if dist < best_dist:
                best_dist   = dist
                best_target = (x, y)

    # 블록 바로 옆으로 이동 (밀 수 있도록 인접)
    push_adj = None
    if best_target:
        tx, ty = best_target
        for dname, (dx, dy) in DIR_VEC.items():
            adj = (tx - dx, ty - dy)   # 블록과 반대편에서 접근
            path = _bfs(env, agent_pos, adj, state["stop_objects"])
            if path is not None:
                push_adj = adj
                break

    target = push_adj if push_adj else best_target

    print(f"  [목표FSM] BREAK_RULE  target={target}  block={best_target}")

    return {
        **state,
        "fsm_phase":    "BREAK_RULE",
        "target_pos":   target,
        "target_desc":  f"STOP 블록({best_target}) 밀기 위한 이동",
    }


def node_make_rule(state: AgentState) -> AgentState:
    """
    목표 FSM — MAKE_RULE
    WIN 규칙 생성: IS 블록이 OBJECT IS WIN 트리플을 만들 수 있는 위치로 이동.
    단순화: IS 블록에 인접하거나, WIN property 블록에 인접하는 위치 탐색.
    """
    env         = state["env"]
    agent_pos   = state["agent_pos"]
    rule_blocks = state["rule_blocks_pos"]

    # 'win' property 블록 위치 탐색
    win_blocks = [(pos, name) for pos, name in rule_blocks.items() if name == "win"]
    is_blocks  = [(pos, name) for pos, name in rule_blocks.items() if name == "is"]

    target = None

    # win 블록 기준으로 IS 블록을 밀어 IS WIN을 완성하는 위치 탐색
    for (wx, wy), _ in win_blocks:
        # IS 블록을 win 블록 왼쪽으로 밀어야 함 → IS 블록 우측에서 접근
        adj = (wx - 1, wy)  # win의 왼쪽 = IS 블록 목표 위치
        path = _bfs(env, agent_pos, adj, state["stop_objects"])
        if path is not None:
            target = adj
            break

    # 없으면 IS 블록에 그냥 인접
    if target is None and is_blocks:
        for (ix, iy), _ in is_blocks:
            for dname, (dx, dy) in DIR_VEC.items():
                adj = (ix - dx, iy - dy)
                path = _bfs(env, agent_pos, adj, state["stop_objects"])
                if path is not None:
                    target = adj
                    break
            if target:
                break

    print(f"  [목표FSM] MAKE_RULE  target={target}")

    return {
        **state,
        "fsm_phase":    "MAKE_RULE",
        "target_pos":   target,
        "target_desc":  f"WIN 규칙 완성을 위한 블록 정렬 이동 → {target}",
    }


def node_find_target(state: AgentState) -> AgentState:
    """
    목표 FSM — FIND_TARGET
    WIN 오브젝트로 향하는 최단 경로 설정.
    """
    env         = state["env"]
    agent_pos   = state["agent_pos"]
    win_objects = state["win_objects"]
    stop_objects = state["stop_objects"]

    win_positions = _find_object_positions(env, win_objects)
    best_path     = None
    best_pos      = None
    best_len      = float("inf")

    for wpos in win_positions:
        path = _bfs(env, agent_pos, wpos, stop_objects)
        if path and len(path) < best_len:
            best_len  = len(path)
            best_path = path
            best_pos  = wpos

    print(f"  [목표FSM] FIND_TARGET  win_pos={best_pos}  path_len={best_len if best_path else '없음'}")

    return {
        **state,
        "fsm_phase":   "FIND_TARGET",
        "target_pos":  best_pos,
        "target_desc": f"WIN 오브젝트 @ {best_pos} 로 이동 (경로: {best_len}칸)",
    }


# ─────────────────────────────────────────────────────────────────────────────
# 공간 FSM
# ─────────────────────────────────────────────────────────────────────────────

def node_check_bounds(state: AgentState) -> AgentState:
    """
    공간 FSM — CHECK_BOUNDS
    target_pos 방향 중 실제 이동 가능한 방향 계산.
    그리드 경계 및 STOP 충돌 여부 확인.
    """
    env          = state["env"]
    agent_pos    = state["agent_pos"]
    target_pos   = state["target_pos"]
    stop_objects = state["stop_objects"]

    blocked = []
    valid_dirs = []

    ax, ay = agent_pos
    for dname, (dx, dy) in DIR_VEC.items():
        nx, ny = ax + dx, ay + dy
        if _is_wall_or_stop(env, nx, ny, stop_objects):
            blocked.append(dname)
        else:
            valid_dirs.append(dname)

    print(f"  [공간FSM] CHECK_BOUNDS  valid={valid_dirs}  blocked={blocked}")

    return {
        **state,
        "fsm_phase":    "CHECK_BOUNDS",
        "blocked_dirs": blocked,
    }


def node_moving(state: AgentState) -> AgentState:
    """
    공간 FSM — MOVING
    target_pos를 향해 BFS 경로의 첫 번째 스텝을 pending_action에 설정.
    실제 env.step()은 FSMAgent.act()에서 호출.
    """
    env          = state["env"]
    agent_pos    = state["agent_pos"]
    target_pos   = state["target_pos"]
    stop_objects = state["stop_objects"]
    blocked_dirs = state["blocked_dirs"]

    action = None

    if target_pos is not None:
        path = _bfs(env, agent_pos, target_pos, stop_objects)
        if path:
            action = path[0]

    # BFS 실패 시 막히지 않은 랜덤 방향
    if action is None:
        available = [d for d in ALL_DIRS if d not in blocked_dirs]
        if available:
            # 방문 기록을 피해 새로운 방향 우선
            history = state.get("move_history", [])
            new_dirs = [d for d in available if d not in history[-2:]]
            action = new_dirs[0] if new_dirs else available[0]
        else:
            action = "up"  # 폴백

    print(f"  [공간FSM] MOVING  action={action}  target={target_pos}")

    history = list(state.get("move_history", []))[-9:] + [action]

    return {
        **state,
        "fsm_phase":      "MOVING",
        "pending_action": action,
        "move_history":   history,
    }


def node_blocked(state: AgentState) -> AgentState:
    """
    공간 FSM — BLOCKED
    모든 방향이 막혔거나 target_pos에 도달 불가 시 진입.
    루프 카운터 증가 + 규칙 재스캔 트리거.
    """
    loop = state.get("loop_count", 0) + 1
    print(f"  [공간FSM] BLOCKED  loop_count={loop}  → 규칙 FSM 재스캔")

    return {
        **state,
        "fsm_phase":      "BLOCKED",
        "pending_action": None,
        "loop_count":     loop,
        "current_goal":   "NONE",   # 목표 초기화 → 재평가
        "target_pos":     None,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 라우터 (조건부 엣지)
# ══════════════════════════════════════════════════════════════════════════════

def route_after_emit(state: AgentState) -> str:
    """emit_context 후: WIN이면 END, 아니면 evaluate_goal."""
    if state.get("done") or state.get("win_achieved"):
        return "end"
    return "evaluate_goal"


def route_after_evaluate(state: AgentState) -> str:
    """evaluate_goal 후: 목표에 따라 분기."""
    goal = state.get("current_goal", "NONE")
    if goal == "BREAK":
        return "break_rule"
    elif goal == "MAKE":
        return "make_rule"
    elif goal == "FIND":
        return "find_target"
    else:
        # NONE → 규칙 재스캔
        return "scan_rules"


def route_after_goal(state: AgentState) -> str:
    """break/make/find 후 항상 check_bounds."""
    return "check_bounds"


def route_after_bounds(state: AgentState) -> str:
    """check_bounds 후: BLOCKED면 blocked 노드, 아니면 moving."""
    blocked = state.get("blocked_dirs", [])
    # 4방향 모두 막혔거나 target_pos가 None이면 BLOCKED
    if len(blocked) >= 4 or state.get("target_pos") is None:
        return "blocked"
    return "moving"


def route_after_moving(state: AgentState) -> str:
    """moving 후: pending_action이 있으면 END(액션 실행 대기), 없으면 blocked."""
    if state.get("pending_action"):
        return "end"   # FSMAgent.act()가 pending_action을 게임에 적용
    return "blocked"


def route_after_blocked(state: AgentState) -> str:
    """blocked 후: 루프 한도 초과면 END, 아니면 scan_rules."""
    if state.get("loop_count", 0) >= 30:
        return "end"
    return "scan_rules"


# ══════════════════════════════════════════════════════════════════════════════
# LangGraph 빌드
# ══════════════════════════════════════════════════════════════════════════════

def build_fsm_graph() -> StateGraph:
    """3-FSM LangGraph 그래프를 빌드하고 컴파일된 그래프를 반환."""
    g = StateGraph(AgentState)

    # ── 노드 등록 ──────────────────────────────────────────────────────────
    # 규칙 FSM
    g.add_node("scan_rules",    node_scan_rules)
    g.add_node("emit_context",  node_emit_context)
    # 목표 FSM
    g.add_node("evaluate_goal", node_evaluate_goal)
    g.add_node("break_rule",    node_break_rule)
    g.add_node("make_rule",     node_make_rule)
    g.add_node("find_target",   node_find_target)
    # 공간 FSM
    g.add_node("check_bounds",  node_check_bounds)
    g.add_node("moving",        node_moving)
    g.add_node("blocked",       node_blocked)

    # ── 엔트리 포인트 ──────────────────────────────────────────────────────
    g.set_entry_point("scan_rules")

    # ── 엣지 ──────────────────────────────────────────────────────────────
    # 규칙 FSM 내부
    g.add_edge("scan_rules", "emit_context")

    # 규칙 FSM → 목표 FSM
    g.add_conditional_edges(
        "emit_context",
        route_after_emit,
        {"end": END, "evaluate_goal": "evaluate_goal"},
    )

    # 목표 FSM 내부 분기
    g.add_conditional_edges(
        "evaluate_goal",
        route_after_evaluate,
        {
            "break_rule":  "break_rule",
            "make_rule":   "make_rule",
            "find_target": "find_target",
            "scan_rules":  "scan_rules",
        },
    )

    # 목표 노드 → 공간 FSM
    for goal_node in ("break_rule", "make_rule", "find_target"):
        g.add_conditional_edges(
            goal_node,
            route_after_goal,
            {"check_bounds": "check_bounds"},
        )

    # 공간 FSM 내부
    g.add_conditional_edges(
        "check_bounds",
        route_after_bounds,
        {"moving": "moving", "blocked": "blocked"},
    )
    g.add_conditional_edges(
        "moving",
        route_after_moving,
        {"end": END, "blocked": "blocked"},
    )
    g.add_conditional_edges(
        "blocked",
        route_after_blocked,
        {"scan_rules": "scan_rules", "end": END},
    )

    return g.compile()


# ══════════════════════════════════════════════════════════════════════════════
# FSMAgent: 에이전트 클래스
# ══════════════════════════════════════════════════════════════════════════════

class FSMAgent:
    """
    LangGraph 3-FSM 에이전트.

    매 게임 스텝마다:
      1. 현재 env 상태로 AgentState 초기화
      2. LangGraph 그래프 실행 (규칙 → 목표 → 공간 FSM 순환)
      3. 그래프가 END에 도달할 때 pending_action을 읽어 env.step() 호출

    react_steps / history_len 등 기존 LLMAgent와 같은 인터페이스 유지.
    """

    def __init__(
        self,
        env,
        verbose: bool = True,
        max_loop: int = 30,
    ):
        self.env       = env
        self.verbose   = verbose
        self.max_loop  = max_loop
        self.graph     = build_fsm_graph()
        self.converter = StateConverter()

        self._step = 0

        self._action_map = {
            "up":    env.actions.up,
            "down":  env.actions.down,
            "left":  env.actions.left,
            "right": env.actions.right,
        }

        print("[FSMAgent] LangGraph 3-FSM 초기화 완료")

    def reset(self):
        self._step = 0

    def act(self, obs) -> int:
        """
        LangGraph FSM을 실행해 다음 액션을 반환.
        """
        self._step += 1
        env = self.env

        if self.verbose:
            print(f"\n{'─'*60}")
            print(f"[게임 스텝 {self._step}]")
            print(self.converter.human_ascii(env))

        # ── 초기 상태 구성 ────────────────────────────────────────────────
        info = _parse_env(env)
        initial_state: AgentState = {
            "env":             env,
            "grid_width":      info["grid_width"],
            "grid_height":     info["grid_height"],
            "agent_pos":       info["agent_pos"],
            "active_rules":    info["active_rules"],
            "rule_blocks_pos": info["rule_blocks_pos"],
            "win_objects":     info["win_objects"],
            "stop_objects":    info["stop_objects"],
            "you_objects":     info["you_objects"],
            "current_goal":    "NONE",
            "target_pos":      None,
            "target_desc":     "",
            "blocked_dirs":    [],
            "move_history":    [],
            "win_achieved":    False,
            "loop_count":      0,
            "fsm_phase":       "INIT",
            "done":            False,
            "pending_action":  None,
        }

        # ── LangGraph 실행 ────────────────────────────────────────────────
        final_state = self.graph.invoke(initial_state)

        action_str = final_state.get("pending_action") or "up"
        env_action = self._action_map.get(action_str, env.actions.up)

        if self.verbose:
            print(f"\n  [FSMAgent] 선택 액션: {action_str}")
            print(f"  [FSMAgent] 목표: {final_state.get('current_goal')}  "
                  f"({final_state.get('target_desc', '')})")

        return env_action


# ══════════════════════════════════════════════════════════════════════════════
# 편의 함수: 에피소드 실행
# ══════════════════════════════════════════════════════════════════════════════

def run_episode(
    env,
    agent: FSMAgent,
    render: bool = False,
) -> Tuple[float, int, bool]:
    agent.reset()
    obs          = env.reset()
    total_reward = 0.0
    steps        = 0
    done         = False

    while not done:
        if render:
            try:
                env.render(mode="rgb_array")
            except Exception:
                pass

        action              = agent.act(obs)
        obs, reward, done, *_ = env.step(action)
        total_reward       += reward
        steps              += 1

    success = total_reward > 0
    status  = "✓ 성공" if success else "✗ 실패"
    print(f"\n{'═'*60}")
    print(f"  에피소드 종료  {status}  reward={total_reward:.3f}  steps={steps}")
    print(f"{'═'*60}\n")
    return total_reward, steps, success


# ══════════════════════════════════════════════════════════════════════════════
# CLI 진입점
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LangGraph FSM Agent for Baba Is You")
    parser.add_argument("--env",       type=str, default="env/you_win", help="환경 ID")
    parser.add_argument("--episodes",  type=int, default=3,             help="에피소드 수")
    parser.add_argument("--max_loop",  type=int, default=30,            help="BLOCKED 허용 횟수")
    parser.add_argument("--quiet",     action="store_true",             help="상세 출력 숨기기")
    args = parser.parse_args()

    try:
        from baba import make
    except ImportError:
        raise ImportError("baba 패키지를 찾을 수 없습니다. PYTHONPATH를 확인하세요.")

    env   = make(args.env)
    agent = FSMAgent(env, verbose=not args.quiet, max_loop=args.max_loop)

    print(f"\n{'━'*60}")
    print(f"  환경     : {args.env}")
    print(f"  에이전트 : LangGraph 3-FSM")
    print(f"  에피소드 : {args.episodes}")
    print(f"{'━'*60}\n")

    results = []
    for ep in range(args.episodes):
        print(f"\n[에피소드 {ep+1}/{args.episodes}]")
        reward, steps, success = run_episode(env, agent)
        results.append((reward, steps, success))

    successes = sum(r[2] for r in results)
    avg_steps = sum(r[1] for r in results) / len(results)
    print(f"\n{'━'*60}")
    print(f"  성공률   : {successes}/{args.episodes} ({100*successes/args.episodes:.1f}%)")
    print(f"  평균 스텝: {avg_steps:.1f}")
    print(f"{'━'*60}\n")
