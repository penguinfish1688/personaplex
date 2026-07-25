#!/usr/bin/env node

// Minimal protocol client used to verify cross-server history behavior without
// starting a model turn. Requires Node.js with the built-in WebSocket client.

const [url, command = "list", argument = ""] = process.argv.slice(2);

if (!url || !["create", "turn", "list", "resume", "read"].includes(command)) {
  console.error(
    "usage: node codex/app_server_smoke.mjs ws://127.0.0.1:PORT " +
      "[create [CWD]|turn [CWD]|list [THREAD_ID]|resume THREAD_ID|read THREAD_ID]",
  );
  process.exit(2);
}

if (typeof WebSocket === "undefined") {
  console.error("This Node.js runtime does not provide a built-in WebSocket client.");
  process.exit(2);
}

const socket = new WebSocket(url);
const pending = new Map();
let nextId = 1;
let finished = false;
let turnCompleted = null;

const timeout = setTimeout(() => {
  console.error("Timed out waiting for app-server.");
  socket.close();
  process.exit(1);
}, command === "turn" ? 120000 : 15000);

function send(message) {
  socket.send(JSON.stringify(message));
}

function request(method, params) {
  const id = nextId++;
  send({ method, id, params });
  return new Promise((resolve, reject) => pending.set(id, { resolve, reject }));
}

socket.addEventListener("message", (event) => {
  const message = JSON.parse(String(event.data));
  if (message.method === "turn/completed" && turnCompleted) {
    turnCompleted(message.params);
    turnCompleted = null;
    return;
  }
  if (message.id === undefined || !pending.has(message.id)) return;
  const { resolve, reject } = pending.get(message.id);
  pending.delete(message.id);
  if (message.error) reject(new Error(JSON.stringify(message.error)));
  else resolve(message.result);
});

socket.addEventListener("error", (event) => {
  if (finished) return;
  console.error("WebSocket error:", event.message || event.type || event);
});

socket.addEventListener("open", async () => {
  try {
    await request("initialize", {
      clientInfo: {
        name: "codex_remote_smoke",
        title: "Codex remote smoke test",
        version: "1.0.0",
      },
    });
    send({ method: "initialized", params: {} });

    let result;
    if (command === "create" || command === "turn") {
      result = await request("thread/start", {
        cwd: argument || process.cwd(),
      });
      const threadId = result.thread.id;
      if (command === "turn") {
        const completion = new Promise((resolve) => {
          turnCompleted = resolve;
        });
        await request("turn/start", {
          threadId,
          input: [
            {
              type: "text",
              text: "Reply exactly ORCD_SHARED_HISTORY_SMOKE_OK. Do not use tools.",
            },
          ],
        });
        const completed = await completion;
        console.log(
          JSON.stringify({
            threadId,
            command,
            status:
              (completed.turn && completed.turn.status) ||
              completed.status ||
              "completed",
          }),
        );
      } else {
        console.log(JSON.stringify({ threadId, command }));
      }
    } else if (command === "resume") {
      if (!argument) throw new Error("resume requires a thread id");
      result = await request("thread/resume", { threadId: argument });
      console.log(JSON.stringify({ threadId: result.thread.id, command }));
    } else if (command === "read") {
      if (!argument) throw new Error("read requires a thread id");
      result = await request("thread/read", {
        threadId: argument,
        includeTurns: true,
      });
      console.log(JSON.stringify({ threadId: result.thread.id, command }));
    } else {
      result = await request("thread/list", {
        limit: 100,
        useStateDbOnly: false,
      });
      const ids = (result.data || []).map((thread) => thread.id);
      console.log(
        JSON.stringify({
          command,
          count: ids.length,
          found: argument ? ids.includes(argument) : undefined,
          ids,
        }),
      );
    }
    finished = true;
    clearTimeout(timeout);
    socket.close();
  } catch (error) {
    finished = true;
    clearTimeout(timeout);
    console.error(error.stack || String(error));
    socket.close();
    process.exitCode = 1;
  }
});
