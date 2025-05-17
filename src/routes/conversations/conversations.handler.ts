import { eq } from "drizzle-orm";
import * as HttpStatusCodes from "stoker/http-status-codes";
import * as HttpStatusPhrases from "stoker/http-status-phrases";

import type { AppRouteHandler } from "@/lib/types";

import db from "@/db";
import { conversations } from "@/db/schema";

import type { CreateRoute, GetOneRoute, ListRoute, RemoveRoute } from "./conversations.routes";

export const list: AppRouteHandler<ListRoute> = async (c) => {
  const userId = c.get("userId");

  if (!userId) {
    return c.json(
      { message: HttpStatusPhrases.UNAUTHORIZED },
      HttpStatusCodes.UNAUTHORIZED,
    );
  }

  const conversationsList = await db.query.conversations.findMany({
    where(fields, operators) {
      return operators.eq(fields.userId, userId);
    },
  });

  return c.json(conversationsList, HttpStatusCodes.OK);
};
export const create: AppRouteHandler<CreateRoute> = async (c) => {
  const contactId = c.get("contactId");

  if (!contactId) {
    return c.json(
      { message: HttpStatusPhrases.UNAUTHORIZED },
      HttpStatusCodes.UNAUTHORIZED,
    );
  }

  const conversation = c.req.valid("json");

  // find chatbot by id
  const chatbot = await db.query.chatbots.findFirst({
    where(fields, operators) {
      return operators.eq(fields.id, conversation.chatbotId);
    },
  });

  if (!chatbot) {
    return c.json(
      { message: HttpStatusPhrases.NOT_FOUND },
      HttpStatusCodes.NOT_FOUND,
    );
  }

  const newConversation = await db.insert(conversations).values({
    userId: chatbot.userId,
    chatbotId: conversation.chatbotId,
    contactId,
  }).returning();

  return c.json(newConversation[0], HttpStatusCodes.CREATED);
};

export const getOne: AppRouteHandler<GetOneRoute> = async (c) => {
  const { id } = c.req.valid("param");

  const contactId = c.get("contactId");

  if (!contactId) {
    return c.json(
      { message: HttpStatusPhrases.UNAUTHORIZED },
      HttpStatusCodes.UNAUTHORIZED,
    );
  }
  // find conversation by id without userId
  const conversation = await db.query.conversations.findFirst({
    where(fields, operators) {
      return operators.eq(fields.id, id);
    },
  });

  if (!conversation) {
    return c.json(
      { message: HttpStatusPhrases.NOT_FOUND },
      HttpStatusCodes.NOT_FOUND,
    );
  }

  return c.json(conversation, HttpStatusCodes.OK);
};

export const remove: AppRouteHandler<RemoveRoute> = async (c) => {
  const userId = c.get("userId");

  if (!userId) {
    return c.json(
      { message: HttpStatusPhrases.UNAUTHORIZED },
      HttpStatusCodes.UNAUTHORIZED,
    );
  }

  const { id } = c.req.valid("param");

  // Check if the conversation exists
  const conversation = await db.query.conversations.findFirst({
    where(fields, operators) {
      return operators.and(
        operators.eq(fields.id, id),
        operators.eq(fields.userId, userId),
      );
    },
  });

  if (!conversation) {
    return c.json(
      { message: HttpStatusPhrases.NOT_FOUND },
      HttpStatusCodes.NOT_FOUND,
    );
  }

  // Delete the conversation
  await db.delete(conversations).where(eq(conversations.id, id));

  return c.json({ message: HttpStatusPhrases.OK }, HttpStatusCodes.OK);
};
