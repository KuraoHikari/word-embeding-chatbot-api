import type { TypedServer, TypedSocket } from "@/sockets/types";

import { getRedisInstance } from "@/db/redish";
import { conversationRoom, messageTypingSchema } from "@/sockets/events";

const TYPING_TTL_SECONDS = 5;

function typingKey(conversationId: number, userId: number): string {
  return `typing:${conversationId}:${userId}`;
}

export function registerTypingHandlers(io: TypedServer, socket: TypedSocket): void {
  socket.on("message:typing", async (raw) => {
    // 1. Validate
    const parsed = messageTypingSchema.safeParse(raw);
    if (!parsed.success) {
      socket.emit("error", {
        event: "message:typing",
        message: parsed.error.issues.map(i => i.message).join(", "),
      });
      return;
    }

    const { conversationId, isTyping } = parsed.data;
    const userId = socket.data.userId;
    const room = conversationRoom(conversationId);

    try {
      const redis = getRedisInstance();
      const key = typingKey(conversationId, userId);

      if (isTyping) {
        // Store typing state with a 5-second TTL
        await redis.set(key, "1", { ex: TYPING_TTL_SECONDS });
      }
      else {
        await redis.del(key);
      }

      // Broadcast to other users in the room
      socket.to(room).emit("message:typing", {
        conversationId,
        userId,
        isTyping,
      });
    }
    catch (err) {
      console.error("[socket] message:typing error", err);
    }
  });
}
