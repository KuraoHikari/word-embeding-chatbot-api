import { and, eq } from "drizzle-orm";

import db from "@/db";
import { conversations } from "@/db/schema";

/**
 * Validate that a conversation exists and optionally check ownership.
 */
export async function findConversation(conversationId: number) {
  return db.query.conversations.findFirst({
    where(fields, operators) {
      return operators.eq(fields.id, conversationId);
    },
  });
}

/**
 * Find a conversation that belongs to a specific user.
 */
export async function findUserConversation(conversationId: number, userId: number) {
  return db.query.conversations.findFirst({
    where(fields, operators) {
      return operators.and(
        operators.eq(fields.id, conversationId),
        operators.eq(fields.userId, userId),
      );
    },
  });
}

/**
 * Update the autoReply field of a conversation.
 */
export async function updateAutoReply(conversationId: number, autoReply: boolean) {
  const [updated] = await db
    .update(conversations)
    .set({ autoReply })
    .where(eq(conversations.id, conversationId))
    .returning();

  return updated;
}
