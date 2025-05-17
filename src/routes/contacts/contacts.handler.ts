import { eq } from "drizzle-orm";
import * as HttpStatusCodes from "stoker/http-status-codes";
import * as HttpStatusPhrases from "stoker/http-status-phrases";

import type { AppRouteHandler } from "@/lib/types";

import db from "@/db";
import { contacts } from "@/db/schema";
import { generatePublicContactAccessToken } from "@/lib/token";

import type { CreateRoute, ListRoute, RemoveRoute } from "./contacts.routes";

export const list: AppRouteHandler<ListRoute> = async (c) => {
  const userId = c.get("userId");

  if (!userId) {
    return c.json(
      { message: HttpStatusPhrases.UNAUTHORIZED },
      HttpStatusCodes.UNAUTHORIZED,
    );
  }

  const contacts = await db.query.contacts.findMany({
    where(fields, operators) {
      return operators.eq(fields.userId, userId);
    },
  });

  return c.json(contacts, HttpStatusCodes.OK);
};
export const create: AppRouteHandler<CreateRoute> = async (c) => {
  const contact = c.req.valid("json");

  // find chatbot by id
  const chatbot = await db.query.chatbots.findFirst({
    where(fields, operators) {
      return operators.eq(fields.id, contact.chatbotId);
    },
  });

  if (!chatbot) {
    return c.json(
      { message: HttpStatusPhrases.NOT_FOUND },
      HttpStatusCodes.NOT_FOUND,
    );
  }

  // find contact by email and userId
  const existingContact = await db.query.contacts.findFirst({
    where(fields, operators) {
      return operators.and(
        operators.eq(fields.email, contact.email),
        operators.eq(fields.userId, chatbot.userId),
      );
    },
  });
  // if contact with the same email already exists, in that chatbot just send token
  if (existingContact) {
    const access_token = await generatePublicContactAccessToken(existingContact.id);

    return c.json({ access_token }, HttpStatusCodes.OK);
  }
  else {
    // Create contact
    const insertContact = await db.insert(contacts).values({
      ...contact,
      userId: chatbot.userId,
    }).returning();

    // Generate access token
    const access_token = await generatePublicContactAccessToken(insertContact[0].id);

    return c.json({ access_token }, HttpStatusCodes.OK);
  }
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

  // Check if the contact exists
  const contact = await db.query.contacts.findFirst({
    where(fields, operators) {
      return operators.and(
        operators.eq(fields.id, id),
        operators.eq(fields.userId, userId),
      );
    },
  });
  if (!contact) {
    return c.json(
      {
        message: HttpStatusPhrases.NOT_FOUND,
      },
      HttpStatusCodes.NOT_FOUND,
    );
  }

  await db.delete(contacts).where(eq(contacts.id, id));

  return c.json({
    message: "Contact deleted successfully",
  }, HttpStatusCodes.OK);
};
