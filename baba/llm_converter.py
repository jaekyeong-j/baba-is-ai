"""
state_converter.py
==================
BabaIsYouEnv 상태를 두 가지 형태로 변환하는 모듈

용어 정의
─────────
- Baba     : LLM이 조종하는 플레이어 캐릭터
- Object   : 게임 맵에 존재하는 실체 오브젝트 (baba, key, ball, door, flag 등)
- RuleBlock: 규칙을 구성하는 텍스트 블록 (BABA, IS, YOU, WIN, STOP 등)

출력 방식
─────────
- human_ascii(env) : 콘솔 출력용 ASCII 맵 (사람이 읽는 용)
- llm_text(env)    : LLM 프롬프트용 좌표 기반 텍스트 (LLM이 읽는 용)
                     → 활성 규칙 + 규칙 파괴 분석 포함
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple


class StateConverter:
    """
    BabaIsYouEnv 상태를 두 가지 형태로 변환합니다.

    오브젝트 분류
    ─────────────
    - Object   (실체): baba, key, ball, door, flag, rock ... → 실제로 움직이거나 상호작용
    - RuleBlock(텍스트): BABA, IS, YOU, WIN, STOP, PUSH ... → 규칙 구성 블록, 밀어서 규칙 변경
    """

    _BABA_SYM   = "@"
    _EMPTY_SYM  = "."
    _BORDER_SYM = "#"
    _IWALL_SYM  = "█"

    # ── 공개 API ──────────────────────────────────────────────────────────────

    def human_ascii(self, env) -> str:
        """콘솔 출력용 ASCII 맵 + 범례 + 규칙 요약"""
        objects, rule_blocks = self._parse_grid(env)
        baba_pos = tuple(env.agent_pos)
        w, h = env.grid.width, env.grid.height

        ascii_map, legend = self._build_ascii(objects, rule_blocks, baba_pos, w, h)
        rules_text = self._format_rules(env)
        step_info  = f"남은 스텝: {env.steps_remaining} / {env.max_steps}"

        return "\n".join([
            "## [사람용] 게임 상태 (ASCII 맵)",
            ascii_map,
            "",
            "### 심볼 범례",
            legend,
            "",
            "### 활성 규칙",
            rules_text,
            "",
            f"### {step_info}",
        ])

    def llm_text(self, env) -> str:
        """
        LLM 프롬프트용 좌표 기반 상태 텍스트.

        포함 섹션:
          1. 맵 이동 가능 영역
          2. Baba 위치
          3. Objects 목록 (좌표 + 종류 + 색상)
          4. RuleBlocks 목록 (좌표 + 이름)
          5. 현재 활성 규칙
          6. 규칙 파괴 분석 — 어떤 블록을 밀면 어떤 규칙이 사라지는지
          7. 남은 스텝
        """
        objects, rule_blocks = self._parse_grid(env)
        baba_pos = tuple(env.agent_pos)
        w, h = env.grid.width, env.grid.height

        inner_x_min, inner_x_max = 1, w - 2
        inner_y_min, inner_y_max = 1, h - 2

        lines: List[str] = [
            "## 게임 상태",
            "",
            "### 맵 이동 가능 영역",
            f"  x: {inner_x_min} ~ {inner_x_max},  y: {inner_y_min} ~ {inner_y_max}",
            "  (이 범위를 벗어나는 이동은 불가능합니다)",
            "",
            "### Baba 위치  ← 당신(LLM)이 조종하는 캐릭터",
            f"  ({baba_pos[0]}, {baba_pos[1]})",
            "",
        ]

        # Objects
        lines.append("### Objects  (실제로 존재하는 오브젝트 — baba 제외)")
        lines.append("  형식: (x, y)  종류  색상")
        if objects:
            for (cx, cy), (obj_type, color) in sorted(objects.items()):
                lines.append(f"  ({cx}, {cy})  {obj_type}  [{color}]")
        else:
            lines.append("  (없음)")
        lines.append("")

        # RuleBlocks
        lines.append("### RuleBlocks  (밀어서 규칙을 변경하는 텍스트 블록)")
        lines.append("  형식: (x, y)  블록이름")
        if rule_blocks:
            for (cx, cy), name in sorted(rule_blocks.items()):
                lines.append(f"  ({cx}, {cy})  {name}")
        else:
            lines.append("  (없음)")
        lines.append("")

        # 활성 규칙
        lines.append("### 현재 활성 규칙")
        lines.append(self._format_rules(env))
        lines.append("")

        # 규칙 파괴 분석
        lines.append("### 규칙 파괴 분석  (Baba가 RuleBlock을 밀면 사라지는 규칙)")
        lines.append(self._format_rule_break_analysis(env, rule_blocks))
        lines.append("")

        # 스텝 정보
        lines.append(f"### 남은 스텝: {env.steps_remaining} / {env.max_steps}")

        return "\n".join(lines)

    # ── 내부: 그리드 파싱 ────────────────────────────────────────────────────

    def _parse_grid(
        self, env
    ) -> Tuple[Dict[Tuple[int, int], Tuple[str, str]], Dict[Tuple[int, int], str]]:
        """
        그리드를 순회하여 두 딕셔너리로 분리 반환.

        Returns
        -------
        objects    : {(x, y): (type_name, color)}  — 실체 오브젝트
        rule_blocks: {(x, y): block_name}           — 규칙 텍스트 블록
        """
        from baba.world_object import RuleBlock, Wall

        objects: Dict[Tuple[int, int], Tuple[str, str]] = {}
        rule_blocks: Dict[Tuple[int, int], str] = {}
        baba_pos = tuple(env.agent_pos)

        for j in range(1, env.grid.height - 1):
            for i in range(1, env.grid.width - 1):
                cell = env.grid.get(i, j)
                if cell is None:
                    continue

                if isinstance(cell, RuleBlock):
                    rule_blocks[(i, j)] = cell.name
                elif isinstance(cell, Wall):
                    objects[(i, j)] = ("wall", "grey")
                else:
                    name  = getattr(cell, "name",  cell.type)
                    color = getattr(cell, "color", "grey")
                    # Baba 오브젝트는 @ 심볼로만 표시 — objects에서 제외
                    if (i, j) == baba_pos:
                        continue
                    objects[(i, j)] = (name, color)

        return objects, rule_blocks

    # ── 내부: ASCII 맵 생성 ──────────────────────────────────────────────────

    def _build_ascii(
        self,
        objects: Dict,
        rule_blocks: Dict,
        baba_pos: Tuple[int, int],
        width: int,
        height: int,
    ) -> Tuple[str, str]:
        """
        심볼 할당 규칙 — 모든 심볼은 반드시 1글자
        ────────────────────────────────────────────
        - Baba 오브젝트  : @ (고정)
        - 내부 벽        : █ (고정)
        - RuleBlock      : 대문자 1글자 (A~Z 풀)
        - Object (기타)  : 소문자 1글자 (a~z·0~9 풀)
        """
        obj_symbol_map: Dict[str, str]  = {}
        obj_legend: List[str]           = []
        rule_symbol_map: Dict[str, str] = {}
        rule_legend: List[str]          = []

        used_obj  = {self._BABA_SYM, self._EMPTY_SYM, self._BORDER_SYM, self._IWALL_SYM}
        obj_pool  = iter("abcdefghijklmnopqrstuvwxyz0123456789")
        used_rule = {self._BABA_SYM, self._EMPTY_SYM, self._BORDER_SYM, self._IWALL_SYM}
        rule_pool = iter("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

        def assign_obj(key: str, label: str) -> str:
            if key not in obj_symbol_map:
                sym = next(obj_pool, "?")
                while sym in used_obj:
                    sym = next(obj_pool, "?")
                used_obj.add(sym)
                obj_symbol_map[key] = sym
                obj_legend.append(f"  {sym}  = {label}")
            return obj_symbol_map[key]

        def assign_rule(name: str) -> str:
            if name not in rule_symbol_map:
                sym = next(rule_pool, "?")
                while sym in used_rule:
                    sym = next(rule_pool, "?")
                used_rule.add(sym)
                rule_symbol_map[name] = sym
                rule_legend.append(f"  {sym}  = [Rule] {name}")
            return rule_symbol_map[name]

        grid = [[self._BORDER_SYM] * width for _ in range(height)]
        for j in range(1, height - 1):
            for i in range(1, width - 1):
                grid[j][i] = self._EMPTY_SYM

        bx, by = baba_pos
        grid[by][bx] = self._BABA_SYM

        for (cx, cy), (obj_type, color) in objects.items():
            if obj_type == "wall":
                grid[cy][cx] = self._IWALL_SYM
                continue
            grid[cy][cx] = assign_obj(f"obj_{obj_type}_{color}", f"[Object] {obj_type} ({color})")

        for (cx, cy), name in rule_blocks.items():
            grid[cy][cx] = assign_rule(name)

        legend_lines = [
            f"  {self._BABA_SYM}  = [Baba] LLM이 조종하는 캐릭터  ← 당신",
            f"  {self._IWALL_SYM}  = [Object] 내부 벽 (통과 불가)",
        ]
        if obj_legend:
            legend_lines.append("  --- Objects ---")
            legend_lines += obj_legend
        if rule_legend:
            legend_lines.append("  --- RuleBlocks ---")
            legend_lines += rule_legend
        legend_lines += [
            f"  {self._BORDER_SYM}  = 외곽 테두리 (맵 경계)",
            f"  {self._EMPTY_SYM}  = 빈 칸",
        ]

        col_header = "    " + " ".join(str(i % 10) for i in range(width))
        rows = [col_header] + [f"{j:2d}  " + " ".join(row) for j, row in enumerate(grid)]

        return "\n".join(rows), "\n".join(legend_lines)

    # ── 내부: 규칙 포맷 ──────────────────────────────────────────────────────

    def _format_rules(self, env) -> str:
        """
        활성 규칙을 'WALL IS STOP' 형태로 반환.
        name_mapping으로 내부 키 → 게임 표시명 변환, 중복 제거.
        """
        try:
            from baba.world_object import name_mapping

            ruleset = env.get_ruleset()
            rules   = ruleset.get("_rule_", [])
            if not rules:
                return "  (활성 규칙 없음 — RuleBlock을 밀어 규칙을 만드세요)"

            prop_notes = {
                "you":  "← 당신이 조종하는 오브젝트",
                "win":  "← 승리 조건 오브젝트",
                "lose": "← 위험! 닿으면 패배",
                "stop": "← 통과 불가",
                "push": "← 밀 수 있음",
                "pull": "← 당길 수 있음",
                "move": "← 스스로 움직임",
            }

            def display(raw: str) -> str:
                return name_mapping.get(raw, raw).upper()

            seen:  set  = set()
            lines: List[str] = []
            for rule in rules:
                if "property" in rule and "object" in rule:
                    obj       = display(rule["object"])
                    prop      = display(rule["property"])
                    color     = rule.get("obj_color", "")
                    color_str = f" [{color}]" if color else ""
                    note      = prop_notes.get(prop.lower(), "")
                    line      = f"  {obj}{color_str}  IS  {prop}  {note}".rstrip()
                elif "object1" in rule and "object2" in rule:
                    line = f"  {display(rule['object1'])}  IS  {display(rule['object2'])}  ← 변환 규칙"
                else:
                    continue
                if line not in seen:
                    seen.add(line)
                    lines.append(line)

            return "\n".join(lines) if lines else "  (활성 규칙 없음)"

        except Exception:
            return "  (규칙 파싱 실패)"

    # ── 내부: 규칙 파괴 분석 ─────────────────────────────────────────────────

    def _format_rule_break_analysis(
        self,
        env,
        rule_blocks: Dict[Tuple[int, int], str],
    ) -> str:
        """
        현재 맵에서 RuleBlock 하나를 밀었을 때 깨지는 규칙을 분석해 반환.

        동작 원리
        ─────────
        활성 규칙은 [OBJECT][IS][PROPERTY] 3블록이 가로 또는 세로로 연속할 때 성립.
        그 중 하나라도 움직이면 규칙이 깨진다.
        → 각 RuleBlock 위치에서 그것이 속한 트리플을 찾아
          "이 블록을 밀면 X IS Y 규칙이 사라진다" 는 정보를 LLM에 전달.

        출력 예시
        ─────────
          (2,1) [is] 를 밀면:  WALL IS STOP 규칙이 사라짐
              → 효과: WALL이 통과 불가 속성을 잃음 (자유롭게 통과 가능)
          (3,1) [win] 를 밀면:  FLAG IS WIN 규칙이 사라짐
              → 효과: FLAG가 승리 조건을 잃음 (더 이상 FLAG에 닿아도 승리 안 함)
        """
        try:
            from baba.world_object import name_mapping

            grid   = env.grid
            w, h   = grid.width, grid.height

            prop_effects = {
                "you":  "가 플레이어 조종 불가가 됨 (게임 오버 위험!)",
                "win":  "이 승리 조건을 잃음 (더 이상 승리 불가)",
                "lose": "이 LOSE 속성을 잃음 (안전해짐)",
                "stop": "이 통과 불가 속성을 잃음 (자유롭게 통과 가능)",
                "push": "이 PUSH 속성을 잃음",
                "pull": "이 PULL 속성을 잃음",
                "move": "이 자동 이동 속성을 잃음",
            }

            def display(raw: str) -> str:
                return name_mapping.get(raw, raw).upper()

            def get_name(x: int, y: int) -> Optional[str]:
                """좌표의 셀 이름 반환 (RuleBlock이면 name, 아니면 None)"""
                from baba.world_object import RuleBlock
                if not (0 <= x < w and 0 <= y < h):
                    return None
                cell = grid.get(x, y)
                if cell is None or not isinstance(cell, RuleBlock):
                    return None
                return cell.name  # 이미 name_mapping 적용된 표시명

            results: List[str] = []
            seen_rules: set     = set()

            for (bx, by), bname in rule_blocks.items():
                broken: List[str] = []

                # 가로 방향: 이 블록이 트리플의 첫/중간/끝일 수 있음
                for dx in range(-2, 1):  # offset -2, -1, 0
                    x0, x1, x2 = bx + dx, bx + dx + 1, bx + dx + 2
                    n0 = get_name(x0, by)
                    n1 = get_name(x1, by)
                    n2 = get_name(x2, by)
                    if n0 and n1 == "is" and n2:
                        rule_str = f"{display(n0)} IS {display(n2)}"
                        if rule_str not in seen_rules:
                            seen_rules.add(rule_str)
                            prop_key = n2.lower()
                            effect   = prop_effects.get(prop_key, f"이 {display(n2)} 속성을 잃음")
                            broken.append((rule_str, display(n0), effect))

                # 세로 방향
                for dy in range(-2, 1):
                    y0, y1, y2 = by + dy, by + dy + 1, by + dy + 2
                    n0 = get_name(bx, y0)
                    n1 = get_name(bx, y1)
                    n2 = get_name(bx, y2)
                    if n0 and n1 == "is" and n2:
                        rule_str = f"{display(n0)} IS {display(n2)}"
                        if rule_str not in seen_rules:
                            seen_rules.add(rule_str)
                            prop_key = n2.lower()
                            effect   = prop_effects.get(prop_key, f"이 {display(n2)} 속성을 잃음")
                            broken.append((rule_str, display(n0), effect))

                for (rule_str, obj_name, effect) in broken:
                    results.append(
                        f"  ({bx},{by}) [{bname}] 을 밀면:  {rule_str} 규칙이 사라짐\n"
                        f"      → 효과: {obj_name}{effect}"
                    )

            if not results:
                return "  (현재 밀어서 규칙을 깰 수 있는 블록 없음)"
            return "\n".join(results)

        except Exception as e:
            return f"  (규칙 파괴 분석 실패: {e})"