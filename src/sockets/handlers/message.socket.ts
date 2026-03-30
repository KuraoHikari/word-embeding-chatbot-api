import { eq } from "drizzle-orm";

import type { TypedServer, TypedSocket } from "@/sockets/types";

import db from "@/db";
import { conversations } from "@/db/schema";
import { findConversation } from "@/services/conversation.service";
import { saveMessage, sendAndSaveBotReply } from "@/services/message.service";
import { conversationRoom, messageSendSchema, userRoom } from "@/sockets/events";

export function registerMessageHandlers(io: TypedServer, socket: TypedSocket): void {
 socket.on("message:send", async (raw) => {
  // 1. Validate payload
  const parsed = messageSendSchema.safeParse(raw);
  if (!parsed.success) {
   socket.emit("error", {
    event: "message:send",
    message: parsed.error.issues.map((i) => i.message).join(", "),
   });
   return;
  }

  const { conversationId, chatbotId, text } = parsed.data;

  try {
   // 2. Verify the conversation exists
   const conversation = await findConversation(conversationId);
   if (!conversation) {
    socket.emit("error", {
     event: "message:send",
     message: "Conversation not found",
    });
    return;
   }

   // Always use the conversation's owner for message attribution.
   // contacts are NOT in the users table, so we must use a valid users.id FK.
   // For admin sockets: conversation.userId === socket.data.userId (they own it).
   // For contact sockets: conversation.userId === the chatbot owner.
   const messageUserId = conversation.userId;

   // Use senderRole from client payload if provided (admin ChatInput
   // explicitly sends "admin"), otherwise derive from socket identity.
   const senderRole: "admin" | "contact" = parsed.data.senderRole ?? (socket.data.contactId ? "contact" : "admin");

   const userMessage = await saveMessage({
    userId: messageUserId,
    conversationId,
    text,
    isBot: false,
    senderRole,
   });

   // Update conversation updatedAt so it sorts to top
   await db.update(conversations).set({ updatedAt: new Date() }).where(eq(conversations.id, conversationId));

   const room = conversationRoom(conversationId);
   const ownerRoom = userRoom(conversation.userId);

   // 4. Broadcast user message to the room AND to the conversation
   //    owner's user room (so admin gets notified even for non-active
   //    conversations). Socket.IO deduplicates across .to() calls.
   io.to(room).to(ownerRoom).emit("message:new", userMessage);

   // 5. If auto-reply is on, generate & broadcast bot response
   if (conversation.autoReply) {
    const chatbot = await db.query.chatbots.findFirst({
     where(fields, operators) {
      return operators.eq(fields.id, chatbotId);
     },
    });

    if (chatbot) {
     const botMessage = await sendAndSaveBotReply({
      chatbot: {
       id: chatbot.id,
       userId: chatbot.userId,
       embeddingModel: chatbot.embeddingModel,
       systemPrompt: chatbot.systemPrompt,
       pdfTitle: chatbot.pdfTitle,
       aiModel: chatbot.aiModel,
       maxTokens: chatbot.maxTokens,
       temperature: chatbot.temperature,
       isProposedModel: chatbot.isProposedModel,
      },
      conversationId,
      userMessageText: text,
     });

     io.to(room).to(ownerRoom).emit("message:new", botMessage);
    }
   }
  } catch (err) {
   console.error("[socket] message:send error", err);
   socket.emit("error", {
    event: "message:send",
    message: "Failed to process message",
   });
  }
 });
}
