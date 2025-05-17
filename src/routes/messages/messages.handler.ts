import { eq } from "drizzle-orm";
import * as HttpStatusCodes from "stoker/http-status-codes";
import * as HttpStatusPhrases from "stoker/http-status-phrases";

import type { AppRouteHandler } from "@/lib/types";

import db from "@/db";
import { messages } from "@/db/schema";
import env from "@/env";
import { sendQueryRequestWithoutRetry } from "@/lib/send-query-request-without-retry";

import type { CreateRoute } from "./messages.routes";

export const create: AppRouteHandler<CreateRoute> = async (c) => {
  const contactId = c.get("contactId");
  if (!contactId) {
    return c.json(
      { message: HttpStatusPhrases.UNAUTHORIZED },
      HttpStatusCodes.UNAUTHORIZED,
    );
  }
  const message = c.req.valid("json");

  // find chatbot by id
  const chatbot = await db.query.chatbots.findFirst({
    where(fields, operators) {
      return operators.eq(fields.id, message.chatbotId);
    },
  });
  if (!chatbot) {
    return c.json(
      { message: HttpStatusPhrases.NOT_FOUND },
      HttpStatusCodes.NOT_FOUND,
    );
  }
  // const newMessage = await db.insert(messages).values({
  //     userId: chatbot.userId,
  //     conversationId: message.conversationId,
  //     isBot: false,
  //     text: message.text,
  // }).returning();

  // if (!newMessage) {
  //     return c.json(
  //         { message: HttpStatusPhrases.INTERNAL_SERVER_ERROR },
  //         HttpStatusCodes.INTERNAL_SERVER_ERROR,
  //     );
  // }
  // Example usage in a handler
  const response = await sendQueryRequestWithoutRetry(
    {
      query: message.text,
      userId: String(chatbot.userId),
      chatbotId: String(chatbot.id),
      modelType: chatbot.embedingModel,
      pdfTitle: chatbot.pdfTitle,
      topK: 10,
      similarityThreshold: 0.8,
    },
    env.API_PASSWORD,
  );
  console.log("🚀 ~ constcreate:AppRouteHandler<CreateRoute>= ~ response:", response);

  const results = await response.json();
  return c.json({
    ...results,
  });
};
