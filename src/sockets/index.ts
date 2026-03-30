import type { Server as HttpServer, IncomingMessage, ServerResponse } from "node:http";

import { verify } from "hono/jwt";
import { Server } from "socket.io";

import db from "@/db";
import env from "@/env";

import type { ClientToServerEvents, InterServerEvents, ServerToClientEvents, SocketData, TypedServer, TypedSocket } from "./types";

import { conversationJoinSchema, conversationLeaveSchema, conversationRoom, userRoom } from "./events";
import { registerMessageHandlers } from "./handlers/message.socket";
import { registerPresenceHandlers } from "./handlers/presence.socket";
import { registerTypingHandlers } from "./handlers/typing.socket";

// ── Singleton so the REST layer can import `getIO()` ────────────────

let ioInstance: TypedServer | null = null;

export function getIO(): TypedServer {
 if (!ioInstance) {
  throw new Error("Socket.io has not been initialised yet");
 }
 return ioInstance;
}

// ── Bootstrap ───────────────────────────────────────────────────────

export function initSocketIO(httpServer: HttpServer<typeof IncomingMessage, typeof ServerResponse>): TypedServer {
 const io: TypedServer = new Server<ClientToServerEvents, ServerToClientEvents, InterServerEvents, SocketData>(httpServer, {
  cors: { origin: "*" },
 });

 ioInstance = io;

 // ── Auth middleware — reuse same JWT verification ──────────────────
 io.use(async (socket, next) => {
  try {
   const token = socket.handshake.auth?.token || socket.handshake.headers?.authorization?.replace("Bearer ", "");

   if (!token) {
    return next(new Error("Authentication required"));
   }

   // Try as admin user FIRST — prevents admin tokens from being
   // misidentified as contacts when both JWT secrets are identical.
   try {
    const payload = await verify(token, env.ACCESS_TOKEN_SECRET);
    const subId = payload.sub as number;

    // Verify this sub actually exists in the users table
    const user = await db.query.users.findFirst({
     where(fields, ops) {
      return ops.eq(fields.id, subId);
     },
    });

    if (user) {
     socket.data.userId = subId;
     socket.data.senderRole = "admin";
     return next();
    }
    // sub is not a user — fall through to contact verification
   } catch {
    // Token not valid with admin secret — fall through
   }

   // Try as a public contact token
   try {
    const payload = await verify(token, env.ACCESS_TOKEN_SECRET_PUBLIC);
    const subId = payload.sub as number;

    const contact = await db.query.contacts.findFirst({
     where(fields, ops) {
      return ops.eq(fields.id, subId);
     },
    });

    if (contact) {
     socket.data.contactId = subId;
     socket.data.userId = contact.userId;
     socket.data.senderRole = "contact";
     return next();
    }
    return next(new Error("Invalid token"));
   } catch {
    return next(new Error("Invalid token"));
   }
  } catch {
   return next(new Error("Authentication failed"));
  }
 });

 // ── Connection handler ────────────────────────────────────────────
 io.on("connection", (socket: TypedSocket) => {
  console.log(`[socket] connected: ${socket.id} (userId=${socket.data.userId})`);

  // Admin sockets (no contactId) join a user-specific room so they
  // can receive conversation:new and message:new events for ALL their
  // conversations, not just the currently active one.
  if (!socket.data.contactId && socket.data.userId) {
   const room = userRoom(socket.data.userId);
   socket.join(room);
   console.log(`[socket] ${socket.id} joined ${room}`);
  }
  // ── Room: join ──────────────────────────────────────────────────
  socket.on("conversation:join", (raw) => {
   const parsed = conversationJoinSchema.safeParse(raw);
   if (!parsed.success) {
    socket.emit("error", {
     event: "conversation:join",
     message: parsed.error.issues.map((i) => i.message).join(", "),
    });
    return;
   }
   const room = conversationRoom(parsed.data.conversationId);
   socket.join(room);
   console.log(`[socket] ${socket.id} joined ${room}`);
  });

  // ── Room: leave ─────────────────────────────────────────────────
  socket.on("conversation:leave", (raw) => {
   const parsed = conversationLeaveSchema.safeParse(raw);
   if (!parsed.success) {
    socket.emit("error", {
     event: "conversation:leave",
     message: parsed.error.issues.map((i) => i.message).join(", "),
    });
    return;
   }
   const room = conversationRoom(parsed.data.conversationId);
   socket.leave(room);
   console.log(`[socket] ${socket.id} left ${room}`);
  });

  // ── Register domain handlers ────────────────────────────────────
  registerMessageHandlers(io, socket);
  registerTypingHandlers(io, socket);
  registerPresenceHandlers(io, socket);

  // ── Disconnect logging ──────────────────────────────────────────
  socket.on("disconnect", (reason) => {
   console.log(`[socket] disconnected: ${socket.id} reason=${reason}`);
  });
 });

 return io;
}
