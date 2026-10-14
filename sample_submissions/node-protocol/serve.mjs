// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// A harness in JavaScript that speaks the serve protocol (docs/protocol.md) itself, with no taskbridge and no SDK.
//
// The evaluator opens one WebSocket per task and sends the task. The harness sends requests (exec, read_file,
// write_file), each with an id of its own, and the evaluator answers each with a result that carries the same id.
// log and usage messages get no answer. The harness ends the task with done, also when the evaluator sends cancel.
import { randomUUID } from "node:crypto";
import { setTimeout as sleep } from "node:timers/promises";
import { WebSocketServer } from "ws";

const MODEL = process.env.MODEL_NAME;
// The model's highest reasoning level: max for Claude, high for the others (Gemini refuses max). REASONING_EFFORT
// overrides it.
const EFFORT = process.env.REASONING_EFFORT || (MODEL.toLowerCase().includes("claude") ? "max" : "high");
const CHAT_URL = `${process.env.MODEL_BASE_URL.replace(/\/$/, "")}/chat/completions`;
const WRAP_UP_SEC = 240;            // ask for the summary this long before the time limit
const RESULT_CHARS = 20_000;        // a longer tool result loses its middle
const MAX_NUDGES = 3;               // times in a row an empty or cut-off reply gets a "continue" before the run stops
const MAX_OUTPUT_TOKENS = Number(process.env.MAX_OUTPUT_TOKENS || 32000);   // lower it for a small context window
const MAX_FRAME = 16 << 20;         // the protocol's largest message, either way

const SYSTEM = (workdir) => `You are an expert engineer working on a task in a Linux container with no internet \
access. Your tools act inside it, and bash starts in ${workdir}. Read the task and the files it points to before you \
start. Compute with code, not by hand. Write each deliverable exactly where and in the format the task asks, then \
open it again to check it. When you are done, reply without calling a tool: a short summary of each deliverable, \
where it is, and the key results.`;

const tool = (name, description, ...params) => ({
  type: "function",
  function: { name, description, parameters: {
    type: "object", properties: Object.fromEntries(params.map((p) => [p, { type: "string" }])), required: params } },
});
const TOOLS = [
  tool("bash", "Run a bash command (120 s limit) and return its exit code and output.", "command"),
  tool("read_file", "Read a text file.", "path"),
  tool("write_file", "Create or overwrite a file; parent folders are created.", "path", "content"),
];

class Cancelled extends Error {}
class ModelError extends Error {}   // an error the endpoint sent inside its stream; the call is not tried again

function clip(text) {
  if (text.length <= RESULT_CHARS) return text;
  const half = RESULT_CHARS / 2;
  return `${text.slice(0, half)}\n[... ${text.length - RESULT_CHARS} characters cut ...]\n${text.slice(-half)}`;
}

// The evaluator counts an empty command or path, or one holding a NUL byte or an unpaired surrogate, as a malformed
// message.
function required(args, name) {
  const value = args[name];
  if (typeof value !== "string" || value === "" || value.includes("\0") || !value.isWellFormed()) {
    throw new Error(`${name} must be non-empty text without a NUL byte or an unpaired surrogate`);
  }
  return value;
}

class Task {
  constructor(ws, message) {
    this.ws = ws;
    this.message = message;
    this.summary = "";
    this.waiting = new Map();
    this.ids = 0;
    this.abort = new AbortController();
    this.finished = false;
    const budget = message.limits?.agent_timeout_sec;
    this.deadline = budget ? Date.now() + budget * 1000 : null;
  }

  secondsLeft() {
    return this.deadline === null ? null : (this.deadline - Date.now()) / 1000;
  }

  send(message) {
    if (this.ws.readyState === this.ws.OPEN) this.ws.send(JSON.stringify(message));
  }

  request(message) {
    return new Promise((resolve, reject) => {
      if (this.abort.signal.aborted) return reject(new Cancelled());
      const id = String(++this.ids);
      const frame = JSON.stringify({ ...message, id });
      if (Buffer.byteLength(frame) > MAX_FRAME) return reject(new Error("the request is over the 16 MiB limit"));
      this.waiting.set(id, { resolve, reject });
      this.ws.send(frame);
    });
  }

  receive(message) {
    if (message?.type === "result" && this.waiting.has(message.id)) {
      this.waiting.get(message.id).resolve(message);
      this.waiting.delete(message.id);
    } else if (message?.type === "cancel") {
      this.cancel();
    }
  }

  // The time is up, or the socket closed: stop the model call and every request, and send done while the socket is
  // open. The evaluator closes the socket 10 s after cancel, and a done sent before then still counts.
  cancel() {
    this.abort.abort();
    for (const { reject } of this.waiting.values()) reject(new Cancelled());
    this.waiting.clear();
    this.finish({ type: "done", final_message: this.summary || "Stopped: the time is up." });
  }

  finish(message) {
    if (!this.finished) this.send(message);
    this.finished = true;
  }
}

// One tool call, `fn` being its name and arguments. The arguments are kept as canonical JSON, or {} when they are not a
// JSON object, since some endpoints refuse a conversation holding invalid arguments.
async function runTool(task, fn) {
  let args;
  try {
    args = JSON.parse(fn.arguments || "{}");
    if (typeof args !== "object" || args === null || Array.isArray(args)) throw new Error("not a JSON object");
    fn.arguments = JSON.stringify(args);
  } catch (error) {
    fn.arguments = "{}";
    return `error: the arguments are not a JSON object (${error.message}); send the call again`;
  }
  try {
    if (fn.name === "bash") {
      const r = await task.request({ type: "exec", command: required(args, "command"), timeout_sec: 120 });
      return r.ok === false ? `error: ${r.error}` : clip(`exit code ${r.exit_code}\n${r.output}`);
    }
    if (fn.name === "read_file") {
      const r = await task.request({ type: "read_file", path: required(args, "path") });
      if (!r.ok) return `error: ${r.error}`;
      const note = r.truncated ? `\n[the file has ${r.size} bytes; a read returns at most 10 MB]` : "";
      return clip(Buffer.from(r.content_b64, "base64").toString()) + note;
    }
    if (fn.name === "write_file") {
      if (typeof args.content !== "string") throw new Error("content must be text");
      const r = await task.request({ type: "write_file", path: required(args, "path"),
        content_b64: Buffer.from(args.content).toString("base64") });
      return r.ok ? `wrote ${r.size} bytes to ${args.path}` : `error: ${r.error}`;
    }
    return `error: there is no tool named ${fn.name}`;
  } catch (error) {
    if (error instanceof Cancelled) throw error;
    return `error: ${error.message}`;
  }
}

const isObject = (value) => typeof value === "object" && value !== null && !Array.isArray(value);
const present = (object) => Object.fromEntries(Object.entries(object).filter(([, value]) => value != null));

// One delta's value of a reply field added to what came before: text joins, objects merge key by key, lists join, and
// anything else is replaced (a signature comes whole). Claude's thinking blocks, as LiteLLM streams them, come in
// pieces that join until a signature (or a redacted block's data) closes the block.
function merged(old, value, key) {
  if (["content", "reasoning_content", "reasoning"].includes(key) && typeof value === "string") {
    return (old ?? "") + value;
  }
  if (key === "thinking_blocks" && Array.isArray(value)) {
    const blocks = Array.isArray(old) ? old : [];
    for (const block of value) {
      const last = blocks.at(-1);
      if (!last || last.signature || last.data || block?.data || last.type !== block?.type) blocks.push(block);
      else {
        last.thinking = (last.thinking ?? "") + (block.thinking ?? "");
        last.signature = block.signature;
      }
    }
    return blocks;
  }
  if (isObject(old) && isObject(value)) {
    return { ...old, ...Object.fromEntries(Object.entries(value).map(([k, v]) => [k, merged(old[k], v, k)])) };
  }
  return Array.isArray(old) && Array.isArray(value) ? [...old, ...value] : value;
}

// A streamed reply put together as one assistant message. Streaming keeps bytes moving on a long generation (Node's
// fetch gives up after 5 minutes without a response). The message and its tool calls keep the fields the endpoint
// adds, so they go back with it: DeepSeek refuses a tool-call turn without its reasoning_content, Gemini a tool call
// without the thought signature in its extra_content, and Claude behind LiteLLM keeps its reasoning in thinking_blocks.
// `onText` gets the reply's text so far each time more of it arrives.
async function readReply(body, onText) {
  const message = { role: "assistant", content: "" };
  const calls = [], byIndex = new Map(), byId = new Map();
  let usage, finish, last, buffer = "";
  const decoder = new TextDecoder();
  for await (const bytes of body) {
    const lines = (buffer + decoder.decode(bytes, { stream: true })).split("\n");
    buffer = lines.pop();
    for (const line of lines) {
      if (!line.startsWith("data:")) continue;
      let chunk;
      try {
        chunk = JSON.parse(line.slice(5));
      } catch {
        continue;                       // [DONE], with or without a space, or anything else that is not JSON
      }
      if (chunk?.error) throw new ModelError(`model: ${JSON.stringify(chunk.error).slice(0, 300)}`);
      usage = chunk?.usage ?? usage;
      const choice = chunk?.choices?.[0];
      finish = choice?.finish_reason ?? finish;
      const { tool_calls: parts, ...fields } = choice?.delta ?? {};
      for (const [key, value] of Object.entries(present(fields))) message[key] = merged(message[key], value, key);
      if (fields.content) onText(message.content);
      for (const { index, id, function: fn, ...callFields } of parts ?? []) {
        // by index, or by id when there is none (Gemini) or one index is reused for another call; a delta with
        // neither continues the call being streamed
        let call = byId.get(id) ?? (index == null ? (id ? undefined : last) : byIndex.get(index));
        if (call?.id && id && call.id !== id) call = undefined;
        if (!call) calls.push((call = { id: "", type: "function", function: { name: "", arguments: "" } }));
        if (index != null) byIndex.set(index, call);
        if (id) byId.set(id, call);
        last = call;
        const { name, arguments: args, ...functionFields } = fn ?? {};
        Object.assign(call, present(callFields));
        Object.assign(call.function, present(functionFields));
        call.id ||= id ?? "";
        call.function.name ||= name ?? "";
        call.function.arguments += args ?? "";
      }
    }
  }
  message.content ||= null;
  for (const call of calls) call.id ||= `call_${randomUUID()}`;     // endpoints refuse a tool result for call ""
  if (calls.length) message.tool_calls = calls;
  // Claude refuses a thinking block that comes back without its signature
  const signed = (blocks) => blocks.filter((block) => block?.signature || block?.data);
  for (const holder of [message, message.provider_specific_fields]) {
    if (Array.isArray(holder?.thinking_blocks)) holder.thinking_blocks = signed(holder.thinking_blocks);
  }
  return { message, usage, finish };
}

// One model call, tried up to 5 times in all after a rate limit, a server error or a dropped connection.
async function chat(messages, toolChoice, signal, onText) {
  const body = JSON.stringify({ model: MODEL, messages, tools: TOOLS, tool_choice: toolChoice,
    reasoning_effort: EFFORT, max_completion_tokens: MAX_OUTPUT_TOKENS, stream: true,
    stream_options: { include_usage: true } });
  for (let attempt = 1; ; attempt++) {
    let error, retry;
    try {
      const response = await fetch(CHAT_URL, { method: "POST", body, signal, headers: {
        "Content-Type": "application/json", Authorization: `Bearer ${process.env.MODEL_API_KEY}` } });
      if (response.ok) return await readReply(response.body, onText);
      retry = response.status === 429 || response.status >= 500;
      error = new Error(`model: HTTP ${response.status}: ${(await response.text()).slice(0, 300)}`);
    } catch (caught) {
      [error, retry] = [caught, !signal.aborted && !(caught instanceof ModelError)];
    }
    if (!retry || attempt === 5) throw error;
    await sleep(2000 * 2 ** attempt, undefined, { signal });
  }
}

async function solve(task) {
  const { workdir, instruction } = task.message;
  const messages = [{ role: "system", content: SYSTEM(workdir) }, { role: "user", content: instruction }];
  let nudges = 0, head = "";                     // head: the start of an answer cut off at the output limit
  while (true) {
    const left = task.secondsLeft();             // null when the task has no time limit
    const wrapUp = left !== null && left < WRAP_UP_SEC;
    if (wrapUp) messages.push({ role: "user", content: "Time is almost up. Stop and reply with your summary." });
    // a call still running when the wrap-up starts is stopped
    const signal = left === null || wrapUp ? task.abort.signal
      : AbortSignal.any([task.abort.signal, AbortSignal.timeout(Math.round((left - WRAP_UP_SEC) * 1000))]);
    let reply;
    try {
      // the text streamed so far is the summary, so a reply cut off by the time limit still counts
      reply = await chat(messages, wrapUp ? "none" : "auto", signal, (content) => { task.summary = head + content; });
    } catch (error) {
      if (task.abort.signal.aborted || !signal.aborted) throw error;
      continue;
    }
    const { message, usage, finish } = reply;
    if (usage) {
      task.send({ type: "usage", model: MODEL, input_tokens: usage.prompt_tokens ?? 0,
        output_tokens: usage.completion_tokens ?? 0,
        cache_read_tokens: usage.prompt_tokens_details?.cached_tokens ?? 0 });
    }
    const text = head + (message.content ?? "");
    if (wrapUp || !message.tool_calls) {
      const cut = finish === "length";
      if (wrapUp || (message.content && !cut) || nudges === MAX_NUDGES) return;
      nudges++;
      head = text;
      if (message.content) messages.push(message);      // cut off at the output limit
      messages.push({ role: "user", content: cut ? "Your reply was cut off; continue where it stopped."
        : "Continue the task, and finish with your summary." });
      continue;
    }
    nudges = 0;
    head = "";
    messages.push(message);
    for (const call of message.tool_calls) {
      task.send({ type: "log", text: `${call.function.name} ${call.function.arguments.slice(0, 200)}` });
      messages.push({ role: "tool", tool_call_id: call.id, content: await runTool(task, call.function) });
    }
    if (left !== null) messages.at(-1).content += `\n[${Math.floor(task.secondsLeft() / 60)} minutes left]`;
  }
}

const server = new WebSocketServer({ host: "0.0.0.0", port: Number(process.env.PORT), maxPayload: MAX_FRAME });
server.on("connection", (ws) => {
  let task;
  ws.on("close", () => task?.cancel());
  ws.on("message", (data) => {
    let message;
    try {
      message = JSON.parse(data);
    } catch {
      return;
    }
    if (task) return task.receive(message);
    if (message?.type !== "task") return ws.send(JSON.stringify({ type: "error", message: "expected a task first" }));
    task = new Task(ws, message);
    solve(task)
      .catch((error) => {
        if (task.finished) return;
        console.error(error);
        task.send({ type: "log", text: `harness error: ${error.message}`.slice(0, 2000) });
        if (!task.summary) task.finish({ type: "error", message: `${error.message}`.slice(0, 2000) });
      })
      .finally(() => task.finish({ type: "done", final_message: task.summary }));
  });
});
console.log(`listening on ${process.env.PORT}`);
