import { z } from "zod";

// ── Client → Server payload schemas ─────────────────────────────────

export const conversationJoinSchema = z.object({
 conversationId: z.number().int().positive(),
});

export const conversationLeaveSchema = z.object({
 conversationId: z.number().int().positive(),
});

export const messageSendSchema = z.object({
 conversationId: z.number().int().positive(),
 chatbotId: z.number().int().positive(),
 text: z.string().min(1).max(500),
 senderRole: z.enum(["admin", "contact"]).optional(),
});

export const messageTypingSchema = z.object({
 conversationId: z.number().int().positive(),
 isTyping: z.boolean(),
});

// ── Room helpers ────────────────────────────────────────────────────

export function conversationRoom(conversationId: number): string {
 return `conversation:${conversationId}`;
}

export function userRoom(userId: number): string {
 return `user:${userId}`;
}
