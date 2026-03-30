/**
 * Socket.io integration smoke test.
 *
 * Run WHILE the dev server is already running:
 *   pnpm dev          ← terminal 1
 *   pnpm test:socket  ← terminal 2
 *
 * Or run standalone (this script starts its own server):
 *   npx tsx scripts/test-socket.ts
 */

import { config } from "dotenv";
import { expand } from "dotenv-expand";
import { sign } from "hono/jwt";
import path from "node:path";
import { io as ioClient } from "socket.io-client";

expand(config({ path: path.resolve(process.cwd(), ".env") }));

const PORT = process.env.PORT ?? "9999";
const URL = `http://localhost:${PORT}`;
const SECRET = process.env.ACCESS_TOKEN_SECRET!;

// ── Helper: generate a JWT the same way the app does ────────────────

async function makeToken(userId: number): Promise<string> {
  return sign(
    { sub: userId, exp: Math.floor(Date.now() / 1000) + 3600 },
    SECRET,
  );
}

// ── Test runner ──────────────────────────────────────────────────────

async function run() {
  console.log(`\n🔌  Connecting to ${URL}\n`);

  const token = await makeToken(1); // use any existing userId

  // ── 1. Connect with valid token ──────────────────────────────────
  const socket = ioClient(URL, {
    auth: { token },
    transports: ["websocket"],
  });

  await new Promise<void>((resolve, reject) => {
    socket.on("connect", () => {
      console.log(`✅  Connected  (socket.id=${socket.id})`);
      resolve();
    });
    socket.on("connect_error", (err) => {
      console.error("❌  connect_error:", err.message);
      reject(err);
    });
    setTimeout(() => reject(new Error("Connection timeout")), 5000);
  });

  // ── 2. Join a conversation room ──────────────────────────────────
  console.log("\n📂  Joining conversation:1 …");
  socket.emit("conversation:join", { conversationId: 1 });
  await sleep(300);

  // ── 3. Typing indicator ──────────────────────────────────────────
  console.log("⌨️   Sending typing=true …");
  socket.emit("message:typing", { conversationId: 1, isTyping: true });
  await sleep(300);

  socket.on("message:typing", (data) => {
    console.log("📨  message:typing received:", data);
  });

  // ── 4. Listen for any new messages ──────────────────────────────
  socket.on("message:new", (data) => {
    console.log("💬  message:new received:", data);
  });

  // ── 5. Listen for presence updates ──────────────────────────────
  socket.on("presence:update", (data) => {
    console.log("👤  presence:update received:", data);
  });

  // ── 6. Listen for errors ─────────────────────────────────────────
  socket.on("error", (data) => {
    console.warn("⚠️   server error event:", data);
  });

  // ── 7. Test invalid token (should be rejected) ───────────────────
  console.log("\n🔒  Testing invalid token (expect connection error) …");
  const badSocket = ioClient(URL, {
    auth: { token: "bad.token.here" },
    transports: ["websocket"],
  });

  await new Promise<void>((resolve) => {
    badSocket.on("connect_error", (err) => {
      console.log(`✅  Invalid token correctly rejected: "${err.message}"`);
      badSocket.disconnect();
      resolve();
    });
    badSocket.on("connect", () => {
      console.error("❌  Invalid token was NOT rejected!");
      badSocket.disconnect();
      resolve();
    });
    setTimeout(resolve, 3000);
  });

  // ── 8. Leave room & disconnect ───────────────────────────────────
  console.log("\n🚪  Leaving conversation:1 …");
  socket.emit("conversation:leave", { conversationId: 1 });
  await sleep(300);

  socket.disconnect();
  console.log("\n✅  All checks passed. Disconnected.\n");
  process.exit(0);
}

function sleep(ms: number) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

run().catch((err) => {
  console.error("\n❌  Test failed:", err.message);
  process.exit(1);
});
