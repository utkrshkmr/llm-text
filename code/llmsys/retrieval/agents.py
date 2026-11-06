"""Tool use and agent loops as systems (Chapter 30).

An agent turn is: generate -> parse a tool call out of the text -> validate it against the tool's
JSON schema -> execute it (here: Python in the sandbox of :mod:`llmsys.rlsys.sandbox`) -> append the
result -> generate again. Two systems facts shape the implementation:

* The conversation is **append-only**. Turn t+1's prompt is turn t's prompt plus what the model
  generated plus the tool result, so with prefix caching (:mod:`llmsys.serving`) only the new tokens
  are prefilled. ``TurnCost`` records both the cached and the uncached cost of every turn.
* Tool output is **untrusted input**. It is parsed, bounded in size and placed in the context as
  data; the sandbox limits CPU time, memory, files and wall clock, and is still not a security
  boundary (the RL-systems chapter says what a production one looks like).
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

from ..rlsys.sandbox import _limits

# ----------------------------------------------------------------------------- tool specs
@dataclass
class ToolSpec:
    """One callable tool: an OpenAI/Qwen-style function schema plus the Python that implements it."""
    name: str
    description: str
    parameters: Dict[str, Any]                 # JSON schema of the arguments object
    fn: Callable[..., Any]

    def schema(self) -> Dict[str, Any]:
        """The ``tools=[...]`` entry a chat API would receive; also what we render into the prompt."""
        return {"type": "function",
                "function": {"name": self.name, "description": self.description, "parameters": self.parameters}}


_JSON_TYPES = {"string": str, "integer": int, "number": (int, float), "boolean": bool,
               "array": list, "object": dict, "null": type(None)}


def validate_args(args: Any, schema: Dict[str, Any], path: str = "$") -> List[str]:
    """A small JSON-schema checker: type, required, properties, additionalProperties, enum, items.

    Model output is text; a call that parses as JSON can still name a missing argument or pass a
    string where an integer is expected. Rejecting it here, with a message the model can read, is
    cheaper than letting the tool raise and cheaper still than a wrong side effect.
    """
    errors: List[str] = []
    t = schema.get("type")
    if t is not None:
        allowed = tuple(_JSON_TYPES[x] for x in (t if isinstance(t, list) else [t]))
        if not isinstance(args, allowed) or (isinstance(args, bool) and bool not in allowed):
            return [f"{path}: expected {t}, got {type(args).__name__}"]
    if "enum" in schema and args not in schema["enum"]:
        errors.append(f"{path}: {args!r} is not one of {schema['enum']}")
    if isinstance(args, dict):
        props = schema.get("properties", {})
        for name in schema.get("required", []):
            if name not in args:
                errors.append(f"{path}.{name}: required")
        for name, value in args.items():
            if name in props:
                errors += validate_args(value, props[name], f"{path}.{name}")
            elif schema.get("additionalProperties", True) is False:
                errors.append(f"{path}.{name}: unexpected argument")
    if isinstance(args, list) and "items" in schema:
        for i, v in enumerate(args):
            errors += validate_args(v, schema["items"], f"{path}[{i}]")
    return errors


# ----------------------------------------------------------------------------- call format
@dataclass
class ToolCall:
    name: str
    arguments: Dict[str, Any]
    raw: str = ""
    error: Optional[str] = None               # set when the call text could not be parsed


_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def parse_tool_calls(text: str) -> List[ToolCall]:
    """Extract ``<tool_call>{"name": ..., "arguments": {...}}</tool_call>`` blocks (Qwen's format).

    A fenced JSON object with the same two keys is accepted as a fallback; anything else is a
    parse error attached to the call so the loop can feed it back to the model instead of crashing.
    """
    blocks = _CALL_RE.findall(text) or _FENCE_RE.findall(text)
    calls: List[ToolCall] = []
    for raw in blocks:
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError as e:
            calls.append(ToolCall("", {}, raw, f"invalid JSON: {e.msg} at char {e.pos}"))
            continue
        if not isinstance(obj, dict) or "name" not in obj:
            calls.append(ToolCall("", {}, raw, "a tool call is an object with 'name' and 'arguments'"))
            continue
        args = obj.get("arguments", {})
        if isinstance(args, str):                     # some models double-encode the arguments
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                calls.append(ToolCall(str(obj["name"]), {}, raw, "arguments is not a JSON object"))
                continue
        calls.append(ToolCall(str(obj["name"]), args if isinstance(args, dict) else {}, raw))
    return calls


def format_tool_result(name: str, result: str, max_chars: int = 2000) -> str:
    """Tool output goes back as a bounded, delimited block: a tool, not the user, said this."""
    if len(result) > max_chars:
        result = result[: max_chars // 2] + f"\n... [{len(result) - max_chars} chars truncated] ...\n" + result[-max_chars // 2:]
    return f"<tool_response>\n{json.dumps({'name': name, 'content': result})}\n</tool_response>"


def render_tools(tools: Sequence[ToolSpec]) -> str:
    """The tools section of a system prompt: schemas plus the call format we expect back."""
    lines = ["You may call tools. Available tools:"]
    lines += [json.dumps(t.schema()) for t in tools]
    lines.append('To call one, write <tool_call>{"name": "<tool>", "arguments": {...}}</tool_call>.')
    return "\n".join(lines)


# ----------------------------------------------------------------------------- sandboxed python
@dataclass
class SandboxOutput:
    status: str          # "ok" | "error" | "timeout"
    stdout: str
    stderr: str
    elapsed: float


def run_python_sandboxed(code: str, timeout: float = 2.0, mem_mb: int = 512, max_output: int = 4000) -> SandboxOutput:
    """Run ``code`` the way :func:`llmsys.rlsys.sandbox.run_python_reward` does, returning its stdout.

    Same recipe: a fresh ``-I -S`` interpreter in an empty temp directory with an empty environment,
    rlimits on CPU, address space, file descriptors and file size, a wall-clock timeout, and the
    process group killed on timeout. Output is truncated so a ``print`` loop cannot flood the context.
    """
    t0 = time.perf_counter()
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "tool.py")
        with open(path, "w") as f:
            f.write(code)
        try:
            p = subprocess.run([sys.executable, "-I", "-S", path], cwd=tmp, env={}, capture_output=True,
                               text=True, timeout=timeout, preexec_fn=_limits(int(timeout) + 1, mem_mb))
        except subprocess.TimeoutExpired:
            return SandboxOutput("timeout", "", f"timed out after {timeout}s", time.perf_counter() - t0)
    status = "ok" if p.returncode == 0 else "error"
    return SandboxOutput(status, p.stdout[:max_output], p.stderr[-max_output:], time.perf_counter() - t0)


def python_tool(timeout: float = 2.0) -> ToolSpec:
    """The one tool every coding agent has: run a Python snippet, return what it printed."""
    def run(code: str) -> str:
        out = run_python_sandboxed(code, timeout=timeout)
        return out.stdout if out.status == "ok" else f"[{out.status}] {out.stderr.strip()}"
    return ToolSpec("python", "Execute Python code in a sandbox and return its standard output.",
                    {"type": "object", "properties": {"code": {"type": "string", "description": "Python source"}},
                     "required": ["code"], "additionalProperties": False}, run)


# ----------------------------------------------------------------------------- the loop
@dataclass
class TurnCost:
    """Tokens the engine had to process for one turn, with and without prefix caching."""
    prompt_tokens: int          # full conversation so far
    new_prompt_tokens: int      # tokens appended since the previous turn (what a cached engine prefills)
    generated_tokens: int
    seconds: float

    @property
    def uncached_prefill(self) -> int:
        return self.prompt_tokens

    @property
    def cached_prefill(self) -> int:
        return self.new_prompt_tokens


@dataclass
class Turn:
    text: str
    calls: List[ToolCall]
    results: List[str]
    cost: TurnCost


@dataclass
class AgentTrace:
    turns: List[Turn] = field(default_factory=list)
    final_text: str = ""
    stopped: str = ""                       # "answer" | "max_turns" | "error"

    @property
    def total_uncached(self) -> int:
        return sum(t.cost.uncached_prefill for t in self.turns)

    @property
    def total_cached(self) -> int:
        return sum(t.cost.cached_prefill for t in self.turns)


class EngineGenerator:
    """``generate_fn`` backed by an :class:`llmsys.serving.LLMEngine`: each turn is one request.

    The engine looks the conversation prefix up in its block hash table, so a turn whose prompt
    extends the previous one by a few hundred tokens prefills only those (``prefix_hit_rate`` in
    ``engine.metrics()`` shows it).
    """

    def __init__(self, engine, max_new_tokens: int = 64, eos_id: Optional[int] = None):
        from ..serving import SamplingParams
        self.engine = engine
        self.params = SamplingParams(max_new_tokens=max_new_tokens, eos_id=eos_id)

    def __call__(self, prompt: List[int]) -> List[int]:
        return self.engine.generate([prompt], self.params)[0]


class ScriptedGenerator:
    """A stand-in policy that returns pre-written responses in order (tests and demonstrations)."""

    def __init__(self, tokenizer, responses: Sequence[str]):
        self.tok, self.responses, self.i = tokenizer, list(responses), 0
        self.prompts: List[List[int]] = []

    def __call__(self, prompt: List[int]) -> List[int]:
        self.prompts.append(list(prompt))
        text = self.responses[min(self.i, len(self.responses) - 1)]
        self.i += 1
        return self.tok.encode(text) + [self.tok.eos_id]


class AgentLoop:
    """generate -> parse -> validate -> execute -> append, until the model answers without a call."""

    def __init__(self, generate_fn: Callable[[List[int]], List[int]], tokenizer, tools: Sequence[ToolSpec],
                 system_prompt: str = "You are a helpful assistant.", max_turns: int = 6,
                 max_result_chars: int = 2000):
        self.generate_fn, self.tok = generate_fn, tokenizer
        self.tools = {t.name: t for t in tools}
        self.max_turns, self.max_result_chars = max_turns, max_result_chars
        self.system_ids = tokenizer.encode(system_prompt + "\n" + render_tools(tools))

    def execute(self, call: ToolCall) -> str:
        """Validate then run one call; every failure becomes text the model sees on the next turn."""
        if call.error:
            return f"[parse error] {call.error}"
        spec = self.tools.get(call.name)
        if spec is None:
            return f"[unknown tool] {call.name!r}; available: {sorted(self.tools)}"
        errors = validate_args(call.arguments, spec.parameters)
        if errors:
            return "[invalid arguments] " + "; ".join(errors)
        try:
            return str(spec.fn(**call.arguments))
        except Exception as e:                   # a tool bug must not take the agent down
            return f"[tool error] {type(e).__name__}: {e}"

    def run(self, user_message: str) -> AgentTrace:
        trace = AgentTrace()
        prompt = self.system_ids + self.tok.encode("User: " + user_message + "\nAssistant:")
        prev_len = 0
        for _ in range(self.max_turns):
            t0 = time.perf_counter()
            out = self.generate_fn(prompt)
            text = self.tok.decode([t for t in out if t != self.tok.eos_id])
            calls = parse_tool_calls(text)
            results = [format_tool_result(c.name, self.execute(c), self.max_result_chars) for c in calls]
            cost = TurnCost(len(prompt), len(prompt) - prev_len, len(out), time.perf_counter() - t0)
            trace.turns.append(Turn(text, calls, results, cost))
            if not calls:
                trace.final_text, trace.stopped = text, "answer"
                return trace
            prev_len = len(prompt)
            # Append-only: the next prompt extends this one, so a prefix-caching engine reuses it.
            prompt = prompt + out + self.tok.encode("\n" + "\n".join(results) + "\nAssistant:")
        trace.final_text, trace.stopped = trace.turns[-1].text, "max_turns"
        return trace


def turn_cost_table(turns: int, system_tokens: int, tokens_per_turn: int) -> Dict[str, int]:
    """Prefill tokens over a conversation of ``turns`` turns, cached versus re-prefilled every turn.

    Without caching the t-th turn re-prefills S + t * n tokens, a sum that is quadratic in turns;
    with caching each turn prefills its n new tokens, so the total is S + turns * n.
    """
    uncached = sum(system_tokens + t * tokens_per_turn for t in range(turns))
    cached = system_tokens + turns * tokens_per_turn
    return dict(uncached=uncached, cached=cached, ratio=uncached / max(1, cached))
