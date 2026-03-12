"""
llm_agent.py
============
Claude / GPT / Ollama API를 호출하는 LLM 기반 Baba Is You 에이전트
ReAct (Reasoning + Acting) 패턴 적용

ReAct 동작 방식
───────────────
한 게임 스텝 안에서 LLM이 Thought → Action → Observation 루프를 반복:

  [Thought]   현재 상태를 분석하고 계획을 세움
  [Action]    up / down / left / right 중 하나를 선택 (또는 DONE으로 종료)
  [Observation] 실제 게임에서 해당 액션을 실행한 결과를 LLM에게 전달
  → 루프를 최대 react_steps 회 반복 후 마지막 Action을 게임에 적용

용어 정의
─────────
- Baba     : LLM이 조종하는 플레이어 캐릭터 (코드 내 "LLMAgent" 클래스와 구분)
- Object   : 게임 맵에 존재하는 실체 오브젝트 (baba, key, ball, door, flag 등)
- RuleBlock: 규칙을 구성하는 텍스트 블록 (BABA, IS, YOU, WIN, STOP 등)

출력 방식
─────────
- 콘솔(사람용) : ASCII 맵으로 시각적 게임 상황 표시  → state_converter.py
- LLM 프롬프트 : 좌표 기반 텍스트로 상태 전달        → state_converter.py

지원 모델
─────────
- Claude : claude-opus-4-6, claude-sonnet-4-6, claude-haiku-4-5-20251001
- OpenAI : gpt-4o, gpt-4-turbo, gpt-3.5-turbo
- Ollama : llama3, llama3.1, llama3.2, mistral, gemma3 등 (로컬 무료)
"""

from __future__ import annotations

import os
import re
import time
from collections import deque
from typing import List, Optional, Tuple

import numpy as np

from llm_converter import StateConverter


# ──────────────────────────────────────────────────────────────────────────────
# LLM 클라이언트
# ──────────────────────────────────────────────────────────────────────────────

class LLMClient:
    """Claude / OpenAI / Ollama API를 통합 호출하는 클라이언트."""

    # ── ReAct 시스템 프롬프트 ────────────────────────────────────────────────
    SYSTEM_PROMPT = """\
You are Baba, the player character in 'Baba Is You'. You reason and act using the ReAct framework.

=== HOW RULES WORK ===
A rule is activated when three RuleBlocks are consecutive horizontally OR vertically:
  [OBJECT] [IS] [PROPERTY]

Examples:
  BABA IS YOU    -> You control the BABA object.
  FLAG IS WIN    -> Touching a FLAG object wins the level.
  ROCK IS PUSH   -> ROCK objects can be pushed.
  WALL IS STOP   -> WALL objects block movement.
  SKULL IS LOSE  -> Touching a SKULL object kills you.

A rule is ONLY active if the three blocks are aligned in a straight line.
If the alignment is broken, the rule deactivates immediately.
You can push RuleBlocks to create, destroy, or modify rules.

=== HOW TO WIN ===
Move Baba onto an object that currently has the WIN property.
Check the active rules each turn — if no WIN rule exists, push RuleBlocks to form one first.

=== HOW TO LOSE ===
Baba touches an object with the LOSE/DEFEAT property, or you run out of steps.

=== KEY DISTINCTIONS ===
  - Objects    : physical entities on the map (baba, key, ball, door, flag, rock, wall ...)
  - RuleBlocks : text tiles that form rules when aligned. They can be pushed like objects.

=== ReAct FORMAT ===
You MUST follow this exact format every turn:

[Thought]
Analyze the current state step by step:
- What is the current WIN rule? Which object do I need to reach?
- What obstacles or STOP/LOSE rules are blocking the path?
- What RuleBlocks can I push, and what rules would that create or break?
- What is the best next move?

[Action]
Exactly one of: up / down / left / right

Do NOT include anything after [Action]. The game will tell you what happened next."""

    def __init__(self, provider: str, model: str, api_key: str):
        self.provider = provider.lower()
        self.model    = model
        self.api_key  = api_key
        self._client  = None
        self._init_client()

    def _init_client(self):
        if self.provider == "claude":
            try:
                import anthropic
                self._client = anthropic.Anthropic(api_key=self.api_key)
            except ImportError:
                raise ImportError("Claude 사용을 위해 `pip install anthropic` 을 실행하세요.")
        elif self.provider == "openai":
            try:
                import openai
                self._client = openai.OpenAI(api_key=self.api_key)
            except ImportError:
                raise ImportError("OpenAI 사용을 위해 `pip install openai` 을 실행하세요.")
        elif self.provider == "ollama":
            try:
                import requests
                self._client = requests
            except ImportError:
                raise ImportError("`pip install requests` 가 필요합니다.")
        else:
            raise ValueError(
                f"지원하지 않는 provider: '{self.provider}'. "
                "'claude' / 'openai' / 'ollama' 중 하나를 사용하세요."
            )

    def ask(self, messages: List[dict], max_retries: int = 3) -> str:
        """멀티턴 messages 리스트를 받아 LLM 응답을 반환."""
        for attempt in range(max_retries):
            try:
                if self.provider == "claude":
                    return self._ask_claude(messages)
                elif self.provider == "openai":
                    return self._ask_openai(messages)
                else:
                    return self._ask_ollama(messages)
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                wait = 2 ** attempt
                print(f"  [LLM] API 오류 ({e}), {wait}초 후 재시도...")
                time.sleep(wait)

    def _ask_claude(self, messages: List[dict]) -> str:
        response = self._client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=self.SYSTEM_PROMPT,
            messages=messages,
        )
        return response.content[0].text

    def _ask_openai(self, messages: List[dict]) -> str:
        full = [{"role": "system", "content": self.SYSTEM_PROMPT}] + messages
        response = self._client.chat.completions.create(
            model=self.model,
            max_tokens=1024,
            messages=full,
        )
        return response.choices[0].message.content

    def _ask_ollama(self, messages: List[dict], host: str = "http://localhost:11434") -> str:
        full = [{"role": "system", "content": self.SYSTEM_PROMPT}] + messages
        payload = {
            "model": self.model,
            "messages": full,
            "stream": False,
            "options": {"num_predict": 1024},
        }
        resp = self._client.post(f"{host}/api/chat", json=payload, timeout=60)
        resp.raise_for_status()
        return resp.json()["message"]["content"]


# ──────────────────────────────────────────────────────────────────────────────
# LLM 에이전트 (ReAct)
# ──────────────────────────────────────────────────────────────────────────────

class LLMAgent:
    """
    ReAct 패턴으로 Baba Is You를 플레이하는 LLM 에이전트.

    한 게임 스텝마다:
      1. 현재 상태를 LLM에 전달 → [Thought] + [Action] 수신
      2. [Action]을 게임에 실제 적용
      3. 변화된 상태를 [Observation]으로 LLM에 전달 → 다음 [Thought] + [Action]
      4. react_steps 횟수만큼 반복 후 마지막 Action을 게임에 반영

    react_steps=1 이면 기존 single-turn 방식과 동일.
    """

    def __init__(
        self,
        env,
        provider: str = "claude",
        model: str = "claude-sonnet-4-6",
        api_key: Optional[str] = None,
        history_len: int = 5,
        react_steps: int = 3,       # ReAct 루프 반복 횟수 (1 = 기존 single-turn)
        verbose: bool = True,
        fallback_random: bool = True,
    ):
        self.env             = env
        self.verbose         = verbose
        self.fallback_random = fallback_random
        self.history_len     = history_len
        self.react_steps     = max(1, react_steps)

        if api_key is None:
            if provider == "ollama":
                api_key = ""
            else:
                env_var = "ANTHROPIC_API_KEY" if provider == "claude" else "OPENAI_API_KEY"
                api_key = os.environ.get(env_var, "")
                if not api_key:
                    raise ValueError(
                        f"API 키가 없습니다. 인자로 전달하거나 환경변수 {env_var}를 설정하세요."
                    )

        self.llm       = LLMClient(provider=provider, model=model, api_key=api_key)
        self.converter = StateConverter()

        self._action_history: deque = deque(maxlen=history_len)
        self._step = 0

        self._action_map = {
            "up":    env.actions.up,
            "down":  env.actions.down,
            "left":  env.actions.left,
            "right": env.actions.right,
        }

        print(f"[LLMAgent] 초기화 완료  provider={provider}  model={model}  react_steps={react_steps}")

    # ── 공개 API ──────────────────────────────────────────────────────────────

    def act(self, obs) -> int:
        """
        ReAct 루프를 실행하고 최종 액션을 반환.
        react_steps=1 이면 LLM을 한 번만 호출 (기존 방식).
        react_steps>1 이면 중간 액션을 실제로 env에 적용하고 Observation을 LLM에 전달.
        """
        self._step += 1

        if self.verbose:
            print(f"\n{'─'*60}")
            print(f"[게임 스텝 {self._step}]")
            print(self.converter.human_ascii(self.env))

        # ReAct 대화 히스토리 (이 게임 스텝 안에서만 유지)
        messages: List[dict] = []
        last_action = None

        for react_turn in range(self.react_steps):
            # ── User 메시지 구성 ──────────────────────────────────────────
            state_text = self.converter.llm_text(self.env)

            if react_turn == 0:
                # 첫 턴: 전체 상태 + 행동 히스토리
                history_text = self._format_history()
                user_content = state_text
                if history_text:
                    user_content += f"\n\n### 최근 행동 기록\n{history_text}"
                user_content += "\n\nNow reason and decide your action."
            else:
                # 이후 턴: Observation (직전 액션 후 변화된 상태)
                user_content = (
                    f"[Observation]\n"
                    f"You moved {last_action}. Here is the updated state:\n\n"
                    f"{state_text}\n\n"
                    f"Continue your reasoning and decide the next action."
                )

            messages.append({"role": "user", "content": user_content})

            if self.verbose:
                tag = "첫 상태" if react_turn == 0 else f"Observation (turn {react_turn})"
                print(f"\n[ReAct Turn {react_turn+1}/{self.react_steps}  {tag}]")
                print(state_text)

            # ── LLM 호출 ─────────────────────────────────────────────────
            try:
                response = self.llm.ask(messages)
            except Exception as e:
                print(f"  [LLMAgent] API 오류: {e}")
                if self.fallback_random:
                    last_action = self._random_action()
                    print(f"  → 폴백 랜덤 행동: {last_action}")
                    break
                else:
                    raise

            if self.verbose:
                print(f"\n[LLM 응답 — Turn {react_turn+1}]\n{response}")

            # ── 응답을 messages에 추가 (멀티턴 컨텍스트 유지) ────────────
            messages.append({"role": "assistant", "content": response})

            # ── Action 파싱 ───────────────────────────────────────────────
            action = self._parse_action(response)
            if action is None:
                if self.fallback_random:
                    action = self._random_action()
                    print(f"  → 파싱 실패, 랜덤 행동: {action}")
                else:
                    action = "up"

            last_action = action

            if self.verbose:
                print(f"  → 선택 액션: {action}")

            # ── 마지막 턴이 아니면 env에 실제 적용해 Observation 생성 ────
            if react_turn < self.react_steps - 1:
                env_action = self._action_map.get(action, self.env.actions.up)
                _, _, done, *_ = self.env.step(env_action)
                if done:
                    # 게임이 끝났으면 루프 중단
                    break

        self._action_history.append(last_action)
        return self._action_map.get(last_action, self.env.actions.up)

    def reset(self):
        self._action_history.clear()
        self._step = 0

    # ── 내부 헬퍼 ─────────────────────────────────────────────────────────────

    def _parse_action(self, response: str) -> Optional[str]:
        """[Action] 섹션에서 방향 단어 1개를 추출."""
        # 1순위: [Action] 바로 뒤 방향 단어
        m = re.search(r"\[Action\]\s*(up|down|left|right)", response, re.IGNORECASE)
        if m:
            return m.group(1).lower()

        # 2순위: 응답 전체에서 방향 단어 탐색
        for word in ["up", "down", "left", "right"]:
            if re.search(rf"\b{word}\b", response, re.IGNORECASE):
                return word

        return None

    def _format_history(self) -> str:
        if not self._action_history:
            return ""
        return "  이전 행동: " + " → ".join(self._action_history)

    def _random_action(self) -> str:
        return np.random.choice(["up", "down", "left", "right"])


# ──────────────────────────────────────────────────────────────────────────────
# 편의 함수: 에피소드 실행
# ──────────────────────────────────────────────────────────────────────────────

def run_episode(
    env,
    agent: LLMAgent,
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

        action             = agent.act(obs)
        obs, reward, done, *_ = env.step(action)
        total_reward      += reward
        steps             += 1

    success = total_reward > 0
    status  = "✓ 성공" if success else "✗ 실패"
    print(f"\n{'═'*60}")
    print(f"  에피소드 종료  {status}  reward={total_reward:.3f}  steps={steps}")
    print(f"{'═'*60}\n")
    return total_reward, steps, success


# ──────────────────────────────────────────────────────────────────────────────
# CLI 진입점
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LLM Agent (ReAct) for Baba Is You")
    parser.add_argument("--env",          type=str, default="env/you_win",      help="환경 ID")
    parser.add_argument("--provider",     type=str, default="claude",            help="claude | openai | ollama")
    parser.add_argument("--model",        type=str, default="claude-sonnet-4-6", help="모델명")
    parser.add_argument("--api_key",      type=str, default=None,                help="API 키 (ollama는 불필요)")
    parser.add_argument("--episodes",     type=int, default=3,                   help="에피소드 수")
    parser.add_argument("--history",      type=int, default=5,                   help="행동 히스토리 길이")
    parser.add_argument("--react_steps",  type=int, default=3,                   help="ReAct 루프 횟수 (1=기존 방식)")
    parser.add_argument("--quiet",        action="store_true",                   help="상세 출력 숨기기")
    args = parser.parse_args()

    try:
        from baba import make
    except ImportError:
        raise ImportError("baba 패키지를 찾을 수 없습니다. PYTHONPATH를 확인하세요.")

    env = make(args.env)

    agent = LLMAgent(
        env,
        provider=args.provider,
        model=args.model,
        api_key=args.api_key,
        history_len=args.history,
        react_steps=args.react_steps,
        verbose=not args.quiet,
    )

    print(f"\n{'━'*60}")
    print(f"  환경        : {args.env}")
    print(f"  LLM         : {args.provider} / {args.model}")
    print(f"  ReAct steps : {args.react_steps}")
    print(f"  에피소드    : {args.episodes}")
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