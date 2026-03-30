import type { Server, Socket } from "socket.io";
import type { z } from "zod";

import type { conversationJoinSchema, conversationLeaveSchema, messageSendSchema, messageTypingSchema } from "./events";

// ── Payload types inferred from Zod schemas ─────────────────────────

export type ConversationJoinPayload = z.infer<typeof conversationJoinSchema>;
export type ConversationLeavePayload = z.infer<typeof conversationLeaveSchema>;
export type MessageSendPayload = z.infer<typeof messageSendSchema>;
export type MessageTypingPayload = z.infer<typeof messageTypingSchema>;

// ── Server → Client events ──────────────────────────────────────────

export interface ServerToClientEvents {
 "message:new": (data: { id: number; text: string; conversationId: number; userId: number; isBot: boolean; senderRole: "admin" | "bot" | "contact"; createdAt: Date | null; updatedAt: Date | null }) => void;

 "message:typing": (data: { conversationId: number; userId: number; isTyping: boolean }) => void;

 "presence:update": (data: { userId: number; contactId?: number; isOnline: boolean }) => void;

 "conversation:autoReplyUpdated": (data: { conversationId: number; autoReply: boolean }) => void;

 "conversation:new": (data: {
  id: number;
  userId: number;
  chatbotId: number;
  contactId: number;
  autoReply: boolean;
  createdAt: Date | null;
  updatedAt: Date | null;
  lastMessage: null;
  contact: { id: number; name: string; email: string; phone: string | null; userId: number; createdAt: Date | null; updatedAt: Date | null };
 }) => void;

 error: (data: { event: string; message: string }) => void;
}

// ── Client → Server events ──────────────────────────────────────────

export interface ClientToServerEvents {
 "conversation:join": (data: ConversationJoinPayload) => void;
 "conversation:leave": (data: ConversationLeavePayload) => void;
 "message:send": (data: MessageSendPayload) => void;
 "message:typing": (data: MessageTypingPayload) => void;
}

// ── Inter-server events (unused for now but typed for completeness) ─

export interface InterServerEvents {}

// ── Per-socket data ─────────────────────────────────────────────────

export interface SocketData {
 userId: number;
 contactId?: number;
 senderRole?: "admin" | "contact";
}

// ── Typed server & socket aliases ───────────────────────────────────

export type TypedServer = Server<ClientToServerEvents, ServerToClientEvents, InterServerEvents, SocketData>;

export type TypedSocket = Socket<ClientToServerEvents, ServerToClientEvents, InterServerEvents, SocketData>;
